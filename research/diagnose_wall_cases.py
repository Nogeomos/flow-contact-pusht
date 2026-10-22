"""Post-experiment diagnostics of two wall cases; does not alter frozen results."""
from pathlib import Path
import hashlib, json, sys, time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'scripts'))
from reconstruct_contacts import ContactReplayEnv, OUTLINE
from research.controller import ContactController, transform
from research.contact_learning import load_policy

OUT=ROOT/'runs/wall_diagnostics_20261022'
CHECKPOINT=ROOT/'runs/stages234_20261022/contact_main/best.pt'
CASES=[('left_wall','左墙附近',[180.,230.,68.,170.,0.]),
       ('top_left_corner','左上墙角',[180.,170.,68.,128.,np.pi])]

def initialize(state):
    env=ContactReplayEnv()
    env.reset(options={'reset_to_state':[180.,230.,256.,256.,0.]})
    # Define exact body-origin poses at initialization; never teleport during control.
    env.block.angle=state[4]
    env.block.position=state[2:4]
    env.block.velocity=(0.,0.)
    env.block.angular_velocity=0.
    env.agent.position=state[:2]
    env.agent.velocity=(0.,0.)
    env.space.reindex_shapes_for_body(env.block)
    env.space.reindex_shapes_for_body(env.agent)
    assert np.allclose(env.get_obs(),state)
    vertices=transform(OUTLINE,env.get_obs()[2:])
    assert vertices.min()>=7-1e-7 and vertices.max()<=504+1e-7
    return env

def probe_pusher():
    results=[]
    for x in [0.,21.,491.,512.]:
        env=initialize([100. if x<256 else 400.,100.,256.,256.,0.])
        for _ in range(30):env.step(np.array([x,100.]))
        xy=np.array(env.agent.position)
        results.append(dict(target_x=x,actual_xy=xy.tolist(),
                            pusher_radius=15.,left_wall_clearance=float(xy[0]-22),
                            right_wall_clearance=float(489-xy[0])))
        env.close()
    return results

def main():
    OUT.mkdir(parents=True,exist_ok=False)
    policy=load_policy(CHECKPOINT,device='cpu')
    reports=[]; trajectories=[]
    for name,label,initial in CASES:
        env=initialize(initial); controller=ContactController(env,score_fn=policy)
        states=[env.get_obs().copy()]; coverage=[float(env._get_coverage())]
        actions=[]; contacts=[]; t=time.perf_counter()
        for step in range(300):
            if coverage[-1]>=.95:break
            action=controller.next_action(300-step)
            state,_,_,_,info=env.step(action)
            states.append(state.copy()); actions.append(action.copy())
            contacts.append(len(env.contact_events)); coverage.append(float(info['coverage']))
        states=np.asarray(states); coverage=np.asarray(coverage)
        plans=controller.plans
        report=dict(name=name,label=label,initial_state=initial,final_state=states[-1].tolist(),
                    steps=len(actions),success=bool(coverage.max()>=.95),
                    max_coverage=float(coverage.max()),final_coverage=float(coverage[-1]),
                    contact_intervals=int(np.count_nonzero(contacts)),
                    no_feasible_plans=sum(p['kind']=='no_feasible_contact' for p in plans),
                    plans=plans,seconds=time.perf_counter()-t)
        np.savez_compressed(OUT/f'{name}.npz',states=states,actions=np.asarray(actions),
                            coverage=coverage,contact_event_counts=np.asarray(contacts),goal_pose=env.goal_pose)
        reports.append(report); trajectories.append((label,states,coverage,env.goal_pose.copy()))
        (OUT/f'{name}.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        print(json.dumps({k:v for k,v in report.items() if k!='plans'},ensure_ascii=True),flush=True)
        controller.close();env.close()
    plt.rcParams.update({'font.sans-serif':['Microsoft YaHei'],'axes.unicode_minus':False})
    fig,axes=plt.subplots(2,3,figsize=(11,7.6))
    for row,(label,states,cov,goal) in enumerate(trajectories):
        for col,index in enumerate([0,min(100,len(states)-1),len(states)-1]):
            ax=axes[row,col]
            ax.plot([5,506,506,5,5],[5,5,506,506,5],color='#aaa',linewidth=3)
            ax.add_patch(Polygon(transform(OUTLINE,goal),color='#98d698',alpha=.6))
            ax.plot(states[:index+1,0],states[:index+1,1],color='#377bba',alpha=.4,linewidth=.8)
            ax.add_patch(Polygon(transform(OUTLINE,states[index,2:]),color='#8d9fae'))
            ax.add_patch(Circle(states[index,:2],15,color='#377bba'))
            ax.set(xlim=(0,512),ylim=(512,0),aspect='equal',
                   title=f'{label} / 第{index}步 / 覆盖率{cov[index]:.3f}')
    fig.suptitle('冻结模型与控制器的两例追加诊断：没有修改原测试集或调参',fontsize=13)
    fig.tight_layout();fig.savefig(OUT/'wall_cases.png',dpi=150);plt.close(fig)
    summary=dict(purpose='Post-hoc two-case diagnostic, not a wall/corner benchmark or additional samples in the original test score.',
                 checkpoint_sha256=hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest(),
                 controller_sha256=hashlib.sha256((ROOT/'research/controller.py').read_bytes()).hexdigest(),
                 results=[{k:v for k,v in r.items() if k!='plans'} for r in reports],
                 pusher_wall_probes=probe_pusher(),
                 known_limitations=['No explicit wall features in learned scores; absolute poses present.',
                                    'Desired flow ignores wall feasibility.',
                                    'Controller center clipping 21..491 is looser than full circle clearance 22..489.',
                                    'Kinematic pusher is collision geometry but is not pushed back by static walls.',
                                    'Candidate search has no explicit recovery strategy or requirement to improve current error.'])
    (OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(str(OUT),flush=True)

if __name__=='__main__':main()
