"""Sequential GPU training, persistent command logs, and independent test runs."""
from pathlib import Path
import argparse, datetime, hashlib, json, subprocess, sys, time

ROOT=Path(__file__).resolve().parents[1]


def run(command, out, name):
    out.mkdir(parents=True,exist_ok=True)
    for suffix in ('stdout.txt','stderr.txt','command.json'):
        if (out/f'{name}.{suffix}').exists():
            raise FileExistsError(f'Preserve command evidence: {out/name}.{suffix}')
    before=datetime.datetime.now().isoformat()
    started=time.time()
    with (out/f'{name}.stdout.txt').open('w',encoding='utf-8') as stdout, (out/f'{name}.stderr.txt').open('w',encoding='utf-8') as stderr:
        print('START '+name,flush=True)
        result=subprocess.run(command,cwd=ROOT,stdout=stdout,stderr=stderr)
    record=dict(command=command,cwd=str(ROOT),started=before,seconds=time.time()-started,returncode=result.returncode,
                stdout=str(out/f'{name}.stdout.txt'),stderr=str(out/f'{name}.stderr.txt'))
    (out/f'{name}.command.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
    print(json.dumps(record),flush=True)
    if result.returncode:raise RuntimeError(name+' failed; see saved stderr')


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',default='runs/stages234_20261022')
    p.add_argument('--phase',choices=['contact','ablation','evaluate'],required=True);p.add_argument('--epochs',type=int,default=40)
    args=p.parse_args();base=Path(args.output);logs=base/'logs';py=sys.executable
    configs=[('contact_main',64,'sequence','soft','contact',0),
             ('flow_action',64,'sequence','soft','action',0),
             ('contact_seed1',64,'sequence','soft','contact',1),
             ('contact_seed2',64,'sequence','soft','contact',2),
             ('contact_32',32,'sequence','soft','contact',0),
             ('contact_128',128,'sequence','soft','contact',0),
             ('contact_hard',64,'sequence','hard','contact',0),
             ('contact_endpoint',64,'endpoint','soft','contact',0),
             ('contact_gru',64,'gru','soft','contact',0),
             ('contact_no_flow',64,'none','soft','contact',0)]
    selected=configs[:2] if args.phase=='contact' else configs[2:]
    if args.phase=='evaluate':
        selected=configs
        for name,*_ in selected:
            run([py,'research/contact_learning.py','evaluate','--checkpoint',str(base/name/'best.pt')],logs,'offline_'+name+'_test')
    else:
        for name,n,mode,label,task,seed in selected:
            run([py,'research/contact_learning.py','train','--output',str(base/name),'--points',str(n),
                 '--mode',mode,'--label',label,'--task',task,'--seed',str(seed),'--epochs',str(args.epochs)],logs,name)


if __name__=='__main__':main()
