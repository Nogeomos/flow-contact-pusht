"""Read-only post-test diagnostics; does not tune or execute any policy."""
from pathlib import Path
import argparse
import json
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    summary=json.loads((args.input/'summary.json').read_text())
    failures=[row for row in summary['results'] if not row['success']]
    diagnoses=[]
    panels=[]
    for result in failures:
        episode=result['episode_index']
        arrays=np.load(args.input/f'episode_{episode:06d}.npz',allow_pickle=False)
        full=json.loads((args.input/f'episode_{episode:06d}.json').read_text())
        coverage=arrays['coverage']
        states=arrays['states']
        best=int(coverage.argmax())
        late_start=max(0,len(coverage)-51)
        half=np.flatnonzero(coverage>=.5)
        diagnosis=dict(episode_index=episode,real_steps=result['steps'],max_coverage=float(coverage[best]),
            best_step=best,final_coverage=float(coverage[-1]),
            final_point_rmse=result['final_point_rmse'],final_angle_error_deg=result['final_angle_error_deg'],
            first_half_coverage_step=int(half[0]) if len(half) else None,
            last_50_step_max_coverage_gain=float(coverage[late_start:].max()-coverage[late_start]),
            last_50_step_pusher_travel=float(np.linalg.norm(np.diff(states[late_start:,:2],axis=0),axis=1).sum()),
            contact_intervals=result['contact_intervals'],candidate_rollouts=result['candidate_rollouts'],
            no_feasible_plan_count=sum(plan['kind']=='no_feasible_contact' for plan in full['plans']))
        if coverage[best]>=.9:
            observation='接近目标但未越过0.95成功阈值；在300步预算内仍有精细对齐误差。'
        elif result['final_angle_error_deg']>10:
            observation='达到动作预算时仍有超过10度的朝向误差；该数值描述终态，不能单独证明失败原因。'
        else:
            observation='达到动作预算时尚未覆盖目标；需结合位置误差和路径观察，不能仅凭接触次数解释失败。'
        diagnosis['observation_zh']=observation
        diagnosis['interpretation_limit_zh']='no_feasible也可表示剩余动作不足以完成接近，不能直接等同于障碍物阻塞。这些分析没有用于调整冻结控制器。'
        diagnoses.append(diagnosis)
        indices=[0,min(75,len(coverage)-1),min(150,len(coverage)-1),min(225,len(coverage)-1),best,len(coverage)-1]
        panels.append((episode,indices,arrays['observations_rgb'][indices],coverage[indices]))
    args.output.mkdir(parents=True)
    report=dict(method=summary['method'],split=summary['split'],successes=summary['successes'],
        count=summary['count'],failures=diagnoses,
        evaluation_is_frozen=True,policy_parameters_changed_from_test=False,
        note_zh='逐例观察只使用本次仿真轨迹，不读取原始测试示范未来。失败全部保留；不据此调参或再次挑选测试集。')
    (args.output/'failure_analysis.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    if panels:
        font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',17)
        bold=ImageFont.truetype('C:/Windows/Fonts/arialbd.ttf',22)
        canvas=Image.new('RGB',(1128,78+len(panels)*222),'#eef3f8')
        draw=ImageDraw.Draw(canvas)
        draw.text((20,13),f"{summary['method']}: all {len(panels)} held-out test failures | 300-action budget",fill='#19344c',font=bold)
        for row,(episode,indices,images,coverages) in enumerate(panels):
            y=66+row*222
            draw.text((20,y),f'Episode {episode}',fill='#19344c',font=font)
            for col,(step,rgb,coverage) in enumerate(zip(indices,images,coverages)):
                x=12+col*186
                canvas.paste(Image.fromarray(rgb).resize((174,174)),(x,y+25))
                title=('best' if col==4 else 'final' if col==5 else f'step {step}')
                draw.text((x,y+200),f'{title}: {coverage:.3f}',fill='#19344c',font=font)
        canvas.save(args.output/'all_test_failures.png')
    # ASCII console JSON survives legacy Windows code pages; saved UTF-8 keeps Chinese.
    print(json.dumps(report,indent=2,ensure_ascii=True))


if __name__=='__main__':main()
