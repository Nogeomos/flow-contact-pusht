"""Stage 2: episode-separated flow -> contact / flow -> action experiments.

Future motion is allowed ONLY for offline supervision. Online callers provide
goal-planner flow. Lengths are simulator units, not metres. No test-based tuning.
"""
from pathlib import Path
import argparse, hashlib, json, math, sys, time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from reconstruct_contacts import boundary_points, OUTLINE


def rotation(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def normals(n):
    p = boundary_points(n)
    edges = np.roll(OUTLINE, -1, axis=0) - OUTLINE
    lengths = np.linalg.norm(edges, axis=1)
    seg = np.searchsorted(np.cumsum(lengths), np.arange(n) * lengths.sum() / n, side='right')
    # Clockwise in screen coordinates: the polygon interior is on the left
    # in numerical x-y coordinates. A vertex uses its outgoing edge normal.
    return np.stack([-edges[seg, 1], edges[seg, 0]], -1) / lengths[seg, None]


def features(state, flow_world, n=64):
    """Accept [B,5], [B,8,N,2]; returns geometry[B,N,13], flow[B,N,8,2]."""
    state, flow_world = np.asarray(state), np.asarray(flow_world)
    single = state.ndim == 1
    if single:
        state, flow_world = state[None], flow_world[None]
    theta = state[:, 4]
    c, s = np.cos(theta), np.sin(theta)
    rot = np.stack([c, -s, s, c], -1).reshape(-1, 2, 2)
    pusher = np.einsum('bi,bij->bj', state[:, :2] - state[:, 2:4], rot)
    local = np.broadcast_to(boundary_points(n)[None], (len(state), n, 2))
    normal = np.broadcast_to(normals(n)[None], local.shape)
    rel = pusher[:, None] - local
    global_context = np.stack([state[:, 0]/256-1, state[:, 1]/256-1,
                               state[:, 2]/256-1, state[:, 3]/256-1, c, s], -1)
    geo = np.concatenate([local/100, normal, rel/100,
                          np.linalg.norm(rel, axis=-1, keepdims=True)/100,
                          np.broadcast_to(global_context[:, None], (len(state), n, 6))], -1)
    fl = np.einsum('bhni,bij->bnhj', flow_world, rot) / 30.0
    return geo.astype('float32'), fl.astype('float32')


def load_dataset(n=64, label='soft'):
    """Build all eligible windows from fixed manifests, without crossing episodes."""
    manifest = json.loads((ROOT/'data/derived/windows.json').read_text())
    all_records = [(split, *rec) for split, records in manifest['windows'].items() for rec in records]
    count = len(all_records)
    geo = np.empty((count,n,13),np.float32)
    flow = np.empty((count,n,8,2),np.float32)
    targets = np.zeros((count,n),np.float32)
    distances = np.full((count,n),512,np.float32)
    action = np.empty((count,8,2),np.float32)
    states = np.empty((count,5),np.float32)
    ids = np.empty((count,2),np.int32)
    masks = {key: [] for key in manifest['windows']}
    local = boundary_points(n)
    prev, z = None, None
    for i,(split,ep,t) in enumerate(all_records):
        if ep != prev:
            if z is not None: z.close()
            z = np.load(ROOT/f'data/derived/replayed_contacts/episode_{ep:06d}.npz')
            st, acts, ev = z['states'], z['actions'], z['contact_events']
            th=st[:,4]; c=np.cos(th); s=np.sin(th)
            rots=np.stack([c,-s,s,c],-1).reshape(-1,2,2)
            world=np.einsum('nj,tij->tni',local,rots)+st[:,None,2:4]
            prev=ep
        future = world[t+1:t+9] - world[t]
        g,f=features(st[t],future,n)
        geo[i],flow[i],states[i]=g[0],f[0],st[t]
        selected=ev[(ev['action_index']>=t)&(ev['action_index']<t+8)&ev['on_exterior']]
        if len(selected):
            dist=np.linalg.norm(local[:,None]-selected['point_local'][None],axis=-1).min(1)
            distances[i]=dist
            targets[i]=np.exp(-.5*(dist/5.)**2) if label=='soft' else (dist<=7.5)
        # Predict target relative to CURRENT pusher, expressed in anchor body frame.
        action[i]=(acts[t:t+8]-st[t,:2])@rotation(st[t,4])/100.
        ids[i]=[ep,t]; masks[split].append(i)
    if z is not None:z.close()
    return dict(geometry=geo,flow=flow,target=targets,distance=distances,action=action,
                states=states,ids=ids,splits={k:np.array(v) for k,v in masks.items()})


def make_model(mode='sequence', task='contact'):
    import torch
    from torch import nn
    class Network(nn.Module):
        def __init__(self):
            super().__init__()
            self.mode=mode; self.task=task
            dim={'sequence':16,'endpoint':2,'none':0,'gru':32}[mode]
            if mode=='gru':self.temporal=nn.GRU(2,32,batch_first=True)
            self.point=nn.Sequential(nn.Linear(13+dim,96),nn.SiLU(),nn.Linear(96,96),nn.SiLU())
            self.head=nn.Sequential(nn.Linear(192,96),nn.SiLU(),nn.Linear(96,1))
            self.global_head=nn.Sequential(nn.Linear(96,96),nn.SiLU(),nn.Linear(96,1 if task=='contact' else 16))
        def forward(self,g,f):
            if self.mode=='gru':
                _,h=self.temporal(f.reshape(-1,8,2)); ff=h[0].reshape(*f.shape[:2],32)
            elif self.mode=='sequence':ff=f.flatten(2)
            elif self.mode=='endpoint':ff=f[:,:,-1]
            else:ff=f[...,:0,0]
            x=self.point(torch.cat([g,ff],-1)); pooled=x.max(1).values
            globalout=self.global_head(pooled)
            if self.task=='action':return globalout.reshape(-1,8,2)
            return self.head(torch.cat([x,pooled[:,None].expand_as(x)],-1)).squeeze(-1),globalout[:,0]
    return Network()


def average_precision(target, scores):
    order=np.argsort(-scores,kind='stable'); y=target[order].astype(float)
    if y.sum()==0:return None
    return float((np.cumsum(y)/np.arange(1,len(y)+1)*y).sum()/y.sum())


def contact_metrics(prob, presence, dist, episode_ids):
    yes=dist.min(1)<511; choice=prob.argmax(1); error=dist[np.arange(len(dist)),choice]
    top5=np.take_along_axis(dist,np.argsort(-prob,axis=1)[:,:5],axis=1).min(1)
    per=[]
    for ep in np.unique(episode_ids):
        keep=(episode_ids==ep)&yes
        if keep.any():per.append(float((error[keep]<=10).mean()))
    p=(presence>=.5); tp=(p&yes).sum();fp=(p&~yes).sum();fn=(~p&yes).sum()
    return dict(windows=len(dist),contact_windows=int(yes.sum()),
                top1_within10=float((error[yes]<=10).mean()),top5_within10=float((top5[yes]<=10).mean()),
                mean_contact_distance=float(error[yes].mean()),median_contact_distance=float(np.median(error[yes])),
                episode_macro_top1_within10=float(np.mean(per)),
                point_average_precision=average_precision((dist<=7.5).ravel(),prob.ravel()),
                presence_f1=float(2*tp/max(2*tp+fp+fn,1)),
                presence_balanced_accuracy=float(.5*((p[yes]).mean()+(~p[~yes]).mean())) if (~yes).any() else None,
                no_contact_false_positive_rate=float(p[~yes].mean()) if (~yes).any() else None)


def evaluate(model,data,idx,device,task):
    import torch
    model.eval(); pred=[]; pres=[]
    with torch.no_grad():
        for start in range(0,len(idx),256):
            rows=idx[start:start+256]
            out=model(torch.as_tensor(data['geometry'][rows],device=device),torch.as_tensor(data['flow'][rows],device=device))
            if task=='contact':
                pred.append(out[0].sigmoid().cpu().numpy());pres.append(out[1].sigmoid().cpu().numpy())
            else:pred.append(out.cpu().numpy())
    prediction=np.concatenate(pred)
    if task=='contact':
        presence=np.concatenate(pres)
        return contact_metrics(prediction,presence,data['distance'][idx],data['ids'][idx,0]),prediction,presence
    errors=np.linalg.norm((prediction-data['action'][idx])*100,axis=-1)
    return dict(windows=len(idx),first_action_target_error=float(errors[:,0].mean()),
                action_sequence_target_error=float(errors.mean())),prediction,None


def train(args):
    import torch
    from torch.nn import functional as F
    torch.set_num_threads(4);torch.manual_seed(args.seed);np.random.seed(args.seed)
    torch.backends.cudnn.benchmark=False
    device='cuda' if torch.cuda.is_available() else 'cpu'
    output=Path(args.output);output.mkdir(parents=True,exist_ok=False)
    config=vars(args).copy();config.update(device=device,torch=torch.__version__,
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            split_sha256=hashlib.sha256((ROOT/'data/derived/splits.json').read_bytes()).hexdigest(),
            label_semantics='union of geometric contacts during next eight control intervals',
            input='current geometry, pusher, 8 future material-point displacements in anchor body frame',
            online_flow='must come from current-state goal planner, never held-out future trajectory')
    (output/'config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    data=load_dataset(args.points,args.label);trainidx=data['splits']['train'];validx=data['splits']['validation']
    model=make_model(args.mode,args.task).to(device);opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-4)
    gen=np.random.default_rng(args.seed);best=float('inf');history=[];start=time.time()
    for epoch in range(1,args.epochs+1):
        model.train(); losses=[]
        for rows in np.array_split(gen.permutation(trainidx),math.ceil(len(trainidx)/args.batch)):
            g=torch.as_tensor(data['geometry'][rows],device=device);f=torch.as_tensor(data['flow'][rows],device=device)
            opt.zero_grad(set_to_none=True);out=model(g,f)
            if args.task=='contact':
                y=torch.as_tensor(data['target'][rows],device=device)
                has=torch.as_tensor((data['distance'][rows].min(1)<511).astype('float32'),device=device)
                loss=F.binary_cross_entropy_with_logits(out[0],y,pos_weight=torch.tensor(5.,device=device))+0.25*F.binary_cross_entropy_with_logits(out[1],has)
            else:loss=F.smooth_l1_loss(out,torch.as_tensor(data['action'][rows],device=device))
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);opt.step();losses.append(loss.item())
        metrics,_,_=evaluate(model,data,validx,device,args.task)
        criterion=metrics['mean_contact_distance'] if args.task=='contact' else metrics['first_action_target_error']
        entry=dict(epoch=epoch,train_loss=float(np.mean(losses)),validation=metrics,elapsed_seconds=time.time()-start)
        history.append(entry)
        if criterion<best:
            best=criterion
            torch.save(dict(model=model.state_dict(),config=config,epoch=epoch,validation=metrics),output/'best.pt')
        (output/'history.json').write_text(json.dumps(history,indent=2),encoding='utf-8')
        print(json.dumps(entry),flush=True)
    print('FINISHED '+str(output),flush=True)


class LearnedPolicy:
    def __init__(self,path,device='cpu'):
        import torch
        torch.set_num_threads(2)
        self.device=device;ck=torch.load(path,map_location=device,weights_only=False)
        self.config=ck['config'];self.model=make_model(self.config['mode'],self.config['task']).to(device)
        self.model.load_state_dict(ck['model']);self.model.eval()
    def __call__(self,state,desired_flow_world,points_world=None):
        import torch
        g,f=features(state,desired_flow_world,self.config['points'])
        with torch.no_grad():out=self.model(torch.as_tensor(g,device=self.device),torch.as_tensor(f,device=self.device))
        if self.config['task']=='contact':return out[0][0].sigmoid().cpu().numpy()
        local=out[0,0].cpu().numpy()*100
        return np.clip(np.asarray(state[:2])+local@rotation(state[4]).T,0,512)


def load_policy(checkpoint, device='cpu', seed=0):
    return LearnedPolicy(checkpoint, device=device)


def final_evaluation(args):
    import torch
    torch.set_num_threads(4)
    path=Path(args.checkpoint);ck=torch.load(path,map_location='cpu',weights_only=False);cfg=ck['config']
    data=load_dataset(cfg['points'],cfg['label']);model=make_model(cfg['mode'],cfg['task']);model.load_state_dict(ck['model'])
    metrics,pred,pres=evaluate(model,data,data['splits'][args.split],'cpu',cfg['task'])
    result=dict(split=args.split,checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                chosen_epoch=ck['epoch'],config=cfg,metrics=metrics)
    out=path.parent/f'{args.split}_metrics.json'
    if out.exists():raise FileExistsError(out)
    out.write_text(json.dumps(result,indent=2),encoding='utf-8')
    idx=data['splits'][args.split]
    np.savez_compressed(path.parent/f'{args.split}_predictions.npz',prediction=pred,
                        presence=pres if pres is not None else np.array([]),distance=data['distance'][idx],
                        target=data['target'][idx],ids=data['ids'][idx],states=data['states'][idx])
    print(json.dumps(result),flush=True)


def main():
    p=argparse.ArgumentParser();s=p.add_subparsers(dest='command',required=True)
    t=s.add_parser('train');t.add_argument('--output',required=True);t.add_argument('--points',type=int,default=64)
    t.add_argument('--mode',choices=['sequence','endpoint','none','gru'],default='sequence')
    t.add_argument('--label',choices=['soft','hard'],default='soft');t.add_argument('--task',choices=['contact','action'],default='contact')
    t.add_argument('--seed',type=int,default=0);t.add_argument('--epochs',type=int,default=40)
    t.add_argument('--batch',type=int,default=128);t.add_argument('--lr',type=float,default=0.001)
    e=s.add_parser('evaluate');e.add_argument('--checkpoint',required=True);e.add_argument('--split',choices=['validation','test'],default='test')
    args=p.parse_args();train(args) if args.command=='train' else final_evaluation(args)


if __name__=='__main__':main()
