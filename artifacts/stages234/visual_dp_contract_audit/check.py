from pathlib import Path
import sys,json,hashlib
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from research.visual_dp import ImageWindows,Config,load_policy,DEFAULT_CACHE
from research.evaluate_closed_loop import run_episode
from research.controller import ContactReplayEnv
from scripts.reconstruct_contacts import initial_state,block_mask,overlap
import pyarrow.parquet as pq
import av
import torch
torch.set_num_threads(2)

OUTPUT=Path(__file__).resolve().parent
PROTECTED=[ROOT/'research/visual_dp.py',ROOT/'research/evaluate_closed_loop.py',
           ROOT/'research/controller.py',ROOT/'runs/stages234_20261022/visual_dp/best.pt',
           ROOT/'runs/stages234_20261022/visual_dp/last.pt']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
before={str(p.relative_to(ROOT)):sha(p) for p in PROTECTED}
table=pq.read_table(ROOT/'data/raw/lerobot_pusht/data/chunk-000/file-000.parquet')
raw_actions=np.asarray(table['action'].to_pylist(),dtype=np.float32)
raw_agents=np.asarray(table['observation.state'].to_pylist(),dtype=np.float32)
train=ImageWindows(DEFAULT_CACHE,'train',Config())
validation=ImageWindows(DEFAULT_CACHE,'validation',Config())
assert np.array_equal(train.actions,raw_actions)
assert np.array_equal(train.agents,raw_agents)
window_checks=[]
for dataset in [train,validation]:
    for episode in dataset.episode_ids[:3]:
        start,end=dataset.bounds[episode]
        for anchor in [start,min(start+1,end-1),(start+end)//2,end-1]:
            i=int(np.flatnonzero(dataset.anchors==anchor)[0])
            item=dataset[i]
            observed=np.maximum(start,[anchor-1,anchor])
            action_indices=np.arange(anchor-1,anchor+15)
            valid=(action_indices>=start)&(action_indices<end)
            target=raw_actions[np.clip(action_indices,start,end-1)]
            assert np.array_equal(item['actions'],target)
            assert np.array_equal(item['valid'],valid)
            assert np.array_equal(item['images'].transpose(0,2,3,1),dataset.images[observed])
            assert np.array_equal(item['agents'],raw_agents[observed])
            assert np.array_equal(item['actions'][1],raw_actions[anchor])
            window_checks.append(dict(split=dataset.split,episode=int(episode),anchor=anchor,frame=anchor-start))
assert set(train.episode_ids).isdisjoint(validation.episode_ids)
normalization_error=float(np.abs(((raw_actions/256-1)+1)*256-raw_actions).max())
assert normalization_error<1e-4
start,end=validation.bounds[11]
decoded=[]
with av.open(str(ROOT/'data/raw/lerobot_pusht/videos/observation.image/chunk-000/file-000.mp4')) as container:
    for index,frame in enumerate(container.decode(video=0)):
        if index in [0,1,start,start+9]:
            assert np.array_equal(frame.to_ndarray(format='rgb24'),validation.images[index])
            decoded.append(index)
        if index>=start+9:break
env=ContactReplayEnv()
env.reset(options={'reset_to_state':initial_state(11)})
replay=[]
for t in range(10):
    rgb=env._render()
    assert rgb.shape==(96,96,3) and rgb.dtype==np.uint8
    replay.append(dict(frame=t,pusher_xy_error=float(np.linalg.norm(env.get_obs()[:2]-raw_agents[start+t])),
                       visible_mask_iou=overlap(block_mask(validation.images[start+t]),block_mask(rgb))))
    env.step(raw_actions[start+t].astype(np.float64))
env.close()
class SpyPolicy:
    def __init__(self):self.calls=[]
    def predict(self,images,agents):
        self.calls.append((images.copy(),agents.copy()))
        return np.tile(agents[-1]+np.array([1.,0.]),(8,1))
spy=SpyPolicy()
_,arrays,_=run_episode(11,'visual_dp',spy,max_steps=17,save_frames=False)
assert len(spy.calls)==3
for call,step in zip(spy.calls,[0,8,16]):
    ids=[max(0,step-1),step]
    assert np.array_equal(call[0],arrays['observations_rgb'][ids])
    assert np.array_equal(call[1],arrays['states'][ids,:2])
policy=load_policy(ROOT/'runs/stages234_20261022/visual_dp/best.pt',device='cpu',seed=123)
item=validation[int(np.flatnonzero(validation.anchors==start+10)[0])]
policy.generator.manual_seed(123)
online=policy.predict(item['images'].transpose(0,2,3,1),item['agents'])
policy.generator.manual_seed(123)
batch=policy.predict_batch(item['images'][None],item['agents'][None])[0]
packing_error=float(np.abs(online-batch).max())
assert packing_error<1e-4
assert online.shape==(8,2) and np.isfinite(online).all() and online.min()>=0 and online.max()<=512
after={str(p.relative_to(ROOT)):sha(p) for p in PROTECTED}
assert before==after
report=dict(passed=True,scope='CPU-only read-only audit of visual policy interfaces, cache and timing; no test demonstrations loaded and no parameters changed',
    protected_files_sha256=before,protected_files_unchanged=before==after,
    raw_actions_minmax=[float(raw_actions.min()),float(raw_actions.max())],action_normalization_roundtrip_max_error=normalization_error,
    window_boundary_checks=window_checks,exact_rgb_video_cache_frames_checked=decoded,
    validation_episode11_replay_checks=replay,online_history_spy=dict(calls=3,at_steps=[0,8,16],frames='[0,0], [7,8], [15,16]',matched=True),
    batch_vs_online_packing_max_action_error=packing_error,output_shape=list(online.shape),output_minmax=[float(online.min()),float(online.max())],
    validation_anchor=start+10,online_actions=online.tolist(),reference_actions=item['actions'][1:9].tolist(),
    read_only_assessment='No scaling, RGB/BGR, x/y, observation-action index or action-chunk alignment defect found. This does not establish adequate visual policy accuracy.',
    limitations=['Compressed training RGB differs slightly from clean online render.',
                 '60 epochs with small CNN/U-Net and DDIM20 is an adapted limited-budget baseline, not official full reproduction.',
                 'Checkpoint chosen by validation denoising loss; this is not guaranteed to rank closed-loop success.',
                 'One 8-action chunk is executed open-loop before using fresh image observations.'])
p=OUTPUT/'result.json'
assert not p.exists()
p.write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report,indent=2))
