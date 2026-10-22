"""Turn actual saved predictions and rollouts into scientific figures and videos."""
from pathlib import Path
import argparse, json, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from reconstruct_contacts import boundary_points, OUTLINE

plt.rcParams.update({'font.sans-serif':['Microsoft YaHei','DejaVu Sans'],'axes.unicode_minus':False,
                     'figure.dpi':150,'savefig.bbox':'tight','axes.spines.top':False,'axes.spines.right':False})
NAMES={'contact_main':'完整 flow + 软标签 (64点)','contact_seed1':'完整模型 / 种子1','contact_seed2':'完整模型 / 种子2',
       'contact_32':'32点','contact_128':'128点','contact_hard':'硬标签','contact_endpoint':'仅终点位移',
       'contact_gru':'GRU 时序编码','contact_no_flow':'移除 flow'}
METHODS={'rule_test':'规则接触 + 控制器','contact_test':'学习接触 + 控制器',
         'flow_action_test':'Flow → Action','visual_dp_test':'轻量视觉 Diffusion Policy'}


def transform(p,pose):
    c,s=np.cos(pose[2]),np.sin(pose[2]);return p@np.array([[c,s],[-s,c]])+pose[:2]


def figures(base,out):
    models={name:json.loads((base/name/'test_metrics.json').read_text()) for name in NAMES if (base/name/'test_metrics.json').exists()}
    if models:
        fig,ax=plt.subplots(1,2,figsize=(12,5.3))
        names=list(models);y=np.arange(len(names))
        ax[0].barh(y,[models[n]['metrics']['top1_within10']*100 for n in names],color='#287c99')
        ax[0].set_yticks(y,[NAMES[n] for n in names]);ax[0].invert_yaxis();ax[0].set_xlim(0,100)
        ax[0].set_xlabel('预测点距接触区域 ≤10 单位的窗口比例 (%)')
        ax[0].set_title('离线接触定位：给定示范未来 flow')
        ax[1].barh(y,[models[n]['metrics']['mean_contact_distance'] for n in names],color='#e4a24b')
        ax[1].set_yticks(y,[]);ax[1].invert_yaxis();ax[1].set_xlabel('预测点到真实接触区域的平均距离 (仿真单位)')
        ax[1].set_title('只在存在接触的窗口统计；越低越好')
        fig.tight_layout();fig.savefig(out/'offline_ablation.png');plt.close(fig)
        fig,ax=plt.subplots(figsize=(9,4.4))
        for name in ['contact_main','contact_no_flow','contact_endpoint','contact_gru']:
            if not (base/name/'history.json').exists():continue
            hist=json.loads((base/name/'history.json').read_text())
            ax.plot([h['epoch'] for h in hist],[h['validation']['mean_contact_distance'] for h in hist],label=NAMES[name])
        ax.set(xlabel='训练轮数',ylabel='验证集接触定位平均距离',title='训练进展：仅用验证集选择 checkpoint')
        ax.legend();fig.tight_layout();fig.savefig(out/'training_curves.png');plt.close(fig)
    summaries={key:json.loads((base/key/'summary.json').read_text()) for key in METHODS if (base/key/'summary.json').exists()}
    if summaries:
        fig,ax=plt.subplots(1,2,figsize=(11.5,5))
        labels=[METHODS[n].replace(' + ','\n+ ').replace('轻量视觉 Diffusion Policy','轻量视觉\nDiffusion Policy') for n in summaries]
        values=np.array([v['success_rate'] for v in summaries.values()])*100
        ci=np.array([v['success_rate_wilson_95'] for v in summaries.values()])*100
        colors=['#8aa0ad','#177e89','#d69d48','#8373b8'][:len(summaries)]
        ax[0].bar(np.arange(len(labels)),values,color=colors,yerr=np.stack([values-ci[:,0],ci[:,1]-values]),capsize=4)
        for i,v in enumerate(summaries.values()):ax[0].text(i,ci[i,1]+2,f"{v['successes']}/{v['count']}",ha='center')
        ax[0].set(ylim=(0,110),ylabel='成功率 (%)',title='完整 episode 闭环测试 / 95% Wilson 区间')
        ax[0].set_xticks(np.arange(len(labels)),labels,fontsize=8)
        ax[1].bar(np.arange(len(labels)),[v['mean_final_coverage'] for v in summaries.values()],color=colors)
        ax[1].axhline(.95,linestyle='--',color='#555555',label='成功门槛 0.95')
        ax[1].set(ylim=(0,1.05),ylabel='目标覆盖率',title='最终覆盖率 / 相同 21 个初始状态')
        ax[1].set_xticks(np.arange(len(labels)),labels,fontsize=8);ax[1].legend()
        fig.text(.5,.01,'每条最多300步；接触方法使用仿真位姿和候选物理预测，视觉DP信息和计算条件不同。',ha='center',fontsize=9)
        fig.tight_layout(rect=[0,.04,1,1]);fig.savefig(out/'closed_loop_comparison.png');plt.close(fig)
    if (base/'visual_dp/history.json').exists():
        hist=json.loads((base/'visual_dp/history.json').read_text())
        fig,ax=plt.subplots(figsize=(8,4))
        ax.plot([h['epoch'] for h in hist],[h['train_noise_mse'] for h in hist],label='训练噪声预测误差')
        ax.plot([h['epoch'] for h in hist],[h['validation_noise_mse'] for h in hist],label='验证噪声预测误差 (EMA)')
        ax.set(xlabel='轮数',ylabel='MSE',title='轻量视觉 Diffusion Policy 实际训练');ax.legend();fig.tight_layout()
        fig.savefig(out/'diffusion_training.png');plt.close(fig)
    if (base/'contact_no_flow_test/summary.json').exists():
        ab=[json.loads((base/name/'summary.json').read_text()) for name in ['contact_test','contact_no_flow_test']]
        values=np.array([s['success_rate'] for s in ab])*100
        ci=np.array([s['success_rate_wilson_95'] for s in ab])*100
        fig,ax=plt.subplots(figsize=(7,4.5))
        ax.bar([0,1],values,color=['#177e89','#a2b3bd'],
               yerr=np.stack([values-ci[:,0],ci[:,1]-values]),capsize=5)
        for i,s in enumerate(ab):ax.text(i,ci[i,1]+2,f"{s['successes']}/21",ha='center')
        ax.set_xticks([0,1],['Flow 条件接触模型','移除 flow 输入'])
        ax.set(ylim=(0,105),ylabel='闭环成功率 (%)',title='相同控制器的补充消融 / 单训练种子')
        fig.text(.5,.02,'主试验后追加诊断；相同21场景，不能算作新的独立测试样本。',ha='center',fontsize=8)
        fig.tight_layout(rect=[0,.05,1,1]);fig.savefig(out/'flow_closed_loop_ablation.png');plt.close(fig)
    return models,summaries


def prediction_example(base,out):
    path=base/'contact_main/test_predictions.npz'
    if not path.exists():return
    with np.load(path) as d:
        ids,dist,pred,truth,states=[d[k] for k in ['ids','distance','prediction','target','states']]
    choice=pred.argmax(1);error=dist[np.arange(len(dist)),choice];positive=dist.min(1)<511
    candidates=np.flatnonzero(positive)
    order=candidates[np.argsort(error[candidates])]
    selected=[order[len(order)//4],order[len(order)//2],order[int(len(order)*.95)]]
    fig,axes=plt.subplots(2,3,figsize=(11,7))
    local=boundary_points(64)
    for col,index in enumerate(selected):
        state=states[index];points=transform(local,state[2:]);poly=transform(OUTLINE,state[2:]);poly=np.vstack([poly,poly[0]])
        for row,score in enumerate([truth[index],pred[index]]):
            ax=axes[row,col];ax.fill(poly[:,0],poly[:,1],color='#dde6ec');ax.plot(poly[:,0],poly[:,1],color='#7f929b')
            ax.scatter(points[:,0],points[:,1],c=score,cmap='magma',vmin=0,vmax=1,s=24,zorder=3)
            ax.add_patch(plt.Circle(state[:2],15,color='#3c94ab',alpha=.7))
            if row:ax.scatter(*points[choice[index]],s=150,marker='*',c='#25ba58',edgecolors='black',zorder=5)
            ax.set(xlim=(0,512),ylim=(512,0),aspect='equal')
            ax.set_title(f"示范 {ids[index,0]} / 帧 {ids[index,1]}"+(f" / 距离 {error[index]:.1f}" if row else ''))
            if col==0:ax.set_ylabel('示范接触软标签' if row==0 else '模型接触分数')
    fig.suptitle('测试集离线预测：三个按误差分位选取的样本；绿色星号是模型最高分点',fontsize=12)
    fig.tight_layout(rect=[0,.06,1,1])
    color_axis=fig.add_axes([.35,.025,.3,.015])
    fig.colorbar(matplotlib.cm.ScalarMappable(norm=matplotlib.colors.Normalize(0,1),cmap='magma'),
                 cax=color_axis,orientation='horizontal',label='接触软标签 / 模型分数（未经概率校准）')
    fig.savefig(out/'contact_predictions.png');plt.close(fig)


def comparison_video(base,out,episode=12):
    paths=[(base/key/f'episode_{episode:06d}.npz') for key in METHODS]
    if not all(p.exists() for p in paths):return
    data=[]
    for path in paths:
        with np.load(path) as d:data.append((d['observations_rgb'],d['coverage']))
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',18)
    small=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',14)
    frames=[]
    maxlen=max(len(im) for im,c in data)
    for t in range(0,maxlen,3):
        panel=Image.new('RGB',(960,1050),'#f2f5fa');draw=ImageDraw.Draw(panel)
        draw.text((24,12),f'同一测试初始状态 {episode} | 物理时间 {t/10:.1f} 秒',font=font,fill='#153047')
        for k,((images,cov),label) in enumerate(zip(data,METHODS.values())):
            i=min(t,len(images)-1);x=(k%2)*480;y=55+(k//2)*480
            draw.text((x+20,y),label,font=font,fill='#153047')
            panel.paste(Image.fromarray(images[i]).resize((432,432)),(x+24,y+28))
            draw.text((x+24,y+458),f'覆盖率 {cov[i]:.3f} | 步数 {i}'+(' | 成功停止' if cov[i]>=.95 else ''),font=small,fill='#153047')
        draw.text((24,1020),'接触方法使用物理预测控制器；视觉DP为本项目轻量训练版本。',font=small,fill='#526778')
        frames.append(panel)
    frames[0].save(out/f'comparison_episode_{episode}.gif',save_all=True,append_images=frames[1:],duration=300,loop=0)
    frames[-1].save(out/f'comparison_episode_{episode}_final.png')


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',default='runs/stages234_20261022');p.add_argument('--output',default='runs/stages234_20261022/figures');a=p.parse_args()
    base=Path(a.run);out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    models,summaries=figures(base,out);prediction_example(base,out);comparison_video(base,out)
    (out/'figure_sources.json').write_text(json.dumps({'offline':list(models),'closed_loop':list(summaries),
        'display_episode':12,'selection':'first held-out episode, not selected for success'},indent=2),encoding='utf-8')
    print(str(out),flush=True)

if __name__=='__main__':main()
