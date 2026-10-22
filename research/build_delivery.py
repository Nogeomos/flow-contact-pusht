"""Build a Chinese report and executable README strictly from measured artifacts.

Run with the bundled document Python (ReportLab available there). No experiments
are run and no result is estimated or filled in when an artifact is missing.
"""
from pathlib import Path
from html import escape
import argparse, json, math
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak, Preformatted

ROOT=Path(__file__).resolve().parents[1]
BLUE=colors.HexColor('#16384d');TEAL=colors.HexColor('#177e89');GRAY=colors.HexColor('#596c79')
pdfmetrics.registerFont(TTFont('YaHei','C:/Windows/Fonts/msyh.ttc',subfontIndex=0))
pdfmetrics.registerFont(TTFont('YaHeiBold','C:/Windows/Fonts/msyhbd.ttc',subfontIndex=0))
STYLE=ParagraphStyle('Body',fontName='YaHei',fontSize=10.5,leading=17,textColor=BLUE,wordWrap='CJK',spaceAfter=8)
SMALL=ParagraphStyle('Small',parent=STYLE,fontSize=8.5,leading=13,spaceAfter=6,textColor=GRAY)
TITLE=ParagraphStyle('Title',parent=STYLE,fontName='YaHeiBold',fontSize=23,leading=33,spaceAfter=15)
SUB=ParagraphStyle('Sub',parent=STYLE,fontName='YaHeiBold',fontSize=14,leading=21,spaceAfter=9)
CELL=ParagraphStyle('Cell',parent=STYLE,fontSize=9,leading=14,spaceAfter=0)
WHITE=ParagraphStyle('White',parent=CELL,textColor=colors.white,fontName='YaHeiBold')


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def p(text,style=STYLE):return Paragraph(escape(str(text).replace('–','-').replace('—','-')),style)
def percent(x):return f'{x*100:.2f}%'
def tab(rows,widths):
    converted=[[p(cell,WHITE if i==0 else CELL) for cell in row] for i,row in enumerate(rows)]
    t=Table(converted,colWidths=widths,repeatRows=1,hAlign='LEFT')
    t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),BLUE),('VALIGN',(0,0),(-1,-1),'TOP'),
        ('LEFTPADDING',(0,0),(-1,-1),8),('RIGHTPADDING',(0,0),(-1,-1),8),
        ('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6),
        ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.HexColor('#eef4f7'),colors.white]),
        ('LINEBELOW',(0,-1),(-1,-1),.5,colors.HexColor('#a9bcc5'))]))
    return t


def pic(path,width=520,maxheight=310):
    from PIL import Image as PILImage
    with PILImage.open(path) as im:w,h=im.size
    ratio=min(width/w,maxheight/h)
    return Image(str(path),width=w*ratio,height=h*ratio,hAlign='CENTER')


def footer(canvas,doc):
    canvas.saveState();canvas.setStrokeColor(colors.HexColor('#d0dee5'))
    canvas.line(36,35,A4[0]-36,35);canvas.setFont('YaHei',8);canvas.setFillColor(GRAY)
    canvas.drawString(36,22,'PushT 接触学习 | 第 2-4 阶段实测报告 | 2026-10-22')
    canvas.drawRightString(A4[0]-36,22,str(doc.page));canvas.restoreState()


def build_pdf(base,output):
    summaries={n:read(base/n/'summary.json') for n in ['rule_test','contact_test','flow_action_test','visual_dp_test']}
    offline={n:read(base/n/'test_metrics.json') for n in ['contact_main','contact_32','contact_128','contact_hard','contact_endpoint','contact_gru','contact_no_flow','flow_action']}
    evidence=read(base/'statistical_analysis.json');acceptance=read(base/'final_acceptance.json')
    assert acceptance['status']=='passed'
    dp=read(base/'visual_dp/training_result.json');dm=read(base/'visual_dp/training_manifest.json')
    dpo=read(base/'visual_dp/test_metrics.json');rt=read(base/'training_runtime.json')
    recovery=read(base/'log_collision_recovery.json')
    main=offline['contact_main']['metrics'];contact=summaries['contact_test'];rule=summaries['rule_test']
    rows=[['方法','成功数 / 21','平均最终覆盖率','输入与控制条件']]
    names=['规则接触 + 控制器','学习接触 + 控制器','Flow → action','轻量视觉 Diffusion Policy']
    for key,name in zip(summaries,names):
        s=summaries[key];rows.append([name,f"{s['successes']} / 21  ({percent(s['success_rate'])})",f"{s['mean_final_coverage']:.4f}",
              '准确位姿 + 候选物理预测' if key in ['rule_test','contact_test'] else ('准确位姿 + 规划 flow' if key=='flow_action_test' else '两帧图像 + 推杆位置')])
    story=[]
    def title(t,k):story.extend([p(k,SMALL),p(t,TITLE)])
    def text(t):story.append(p(t))
    def nextpage():story.append(PageBreak())
    title('从物体运动到接触\n第二至四阶段实验报告','二维 PushT | 真实训练、闭环执行与对照实验')
    text('已经完成：训练数据构建、接触模型与消融、独立控制器、闭环测试、视觉扩散策略和直接动作对照。下列数字全部来自保存的实际运行结果。')
    story.extend([tab(rows,[143,105,88,187]),Spacer(1,10),pic(base/'figures/closed_loop_comparison.png',maxheight=245)])
    text(f"主接触模型在离线测试中，接触点定位命中率为 {percent(main['top1_within10'])}；闭环成功 {contact['successes']}/21。三个训练种子的成功率为 71.43%、57.14%、71.43%，均值 66.67%。")
    text('目前证据支持“流程可运行、模型能预测示范接触”。主模型比规则多成功两条，不能据此宣称稳定或显著优于规则。视觉基线是轻量适配版，未复现论文完整训练预算；本轮也没有证明跨机器人或真实世界泛化。')

    nextpage();title('数据、坐标与研究问题','01 / 可复现的实验基础')
    story.append(tab([['项目','已核验事实'],['原始数据','206 条示范，25,650 帧，10 Hz；原始字段无物体位姿与接触真值。'],
        ['整条示范划分','训练 164 / 验证 21 / 测试 21；固定种子 20261022。'],
        ['接触窗口','8 个动作区间 + 9 个状态；19,130 / 2,589 / 2,281 个训练/验证/测试窗口。'],
        ['排除规则','排除没有视频参考端点的最后区间，以及内部拼接缝接触区间；不按成功或误差筛选。'],
        ['视觉窗口','20,443 / 2,758 / 2,449；允许同一示范边界 padding，padding 不计入训练损失。']], [116,407]))
    story.append(Spacer(1,12));story.append(p('一个编号，始终代表同一个材料点',SUB))
    text('在 T 形物体局部轮廓上等弧长采样。主实验用 64 点，另测 32 与 128 点。世界坐标满足 p_world = R(theta) p_local + t。t 是物体刚体原点，不是质心；局部物理质心为 (0,45)。')
    text('x 向右、y 向下；角度用弧度，正角在屏幕上表现为顺时针。长度是 512×512 场景中的仿真单位，不能直接称为米或毫米。动作是推杆中心的绝对目标坐标，推杆实际位置由 PD 控制与碰撞共同决定。')
    story.append(p('flow 的含义',SUB))
    text('每个点的 flow 是未来位置减去当前点的位置。所有未来位移都转到当前帧的局部坐标系：这样保留了运动，也让不同位姿下的输入更一致。flow 不是推杆动作。')
    text('离线训练和离线测试使用示范未来运动；真正闭环执行只使用当前状态和固定目标构造短期 flow。闭环脚本不读取测试示范未来状态、动作、标签或视频。')
    text('接触标签来自固定仿真器的碰撞回放，是重建监督，不是独立实物接触真值。它说明示范接触过哪里，不能直接解释为最优接触或推动成功概率。')

    nextpage();title('接触模型与离线结果','02 / 阶段二：训练和消融已经运行')
    text('主模型对每个轮廓点输入 13 维当前几何/推杆特征，以及 8×2 维短期位移。共享两层 96 维网络编码每个点，再汇总全局特征，输出逐点分数和一个有接触分数。全部接触/动作实验训练 40 轮，仅按验证集指标选择 checkpoint。')
    text('标签取未来 8 个动作周期真实接触区域的并集：软标签为距离高斯核（sigma=5）的最大值，无接触窗口全为 0。训练采用加权二元交叉熵；分数未做概率校准。硬标签消融使用距离≤7.5。GRU 消融显式编码 8 个时间步；仅终点消融只保留最后位移。')
    labels={'contact_main':'主模型 64 / soft','contact_32':'32 点','contact_128':'128 点','contact_hard':'硬标签','contact_endpoint':'仅终点位移','contact_gru':'GRU 时序','contact_no_flow':'移除 flow'}
    table=[['模型','定位≤10','平均距离','无接触假阳性']]
    for key,name in labels.items():
        m=offline[key]['metrics'];table.append([name,percent(m['top1_within10']),f"{m['mean_contact_distance']:.3f}",percent(m['no_contact_false_positive_rate'])])
    story.extend([tab(table,[170,105,110,138]),Spacer(1,12),pic(base/'figures/training_curves.png',maxheight=220)])
    text('定位指标只统计真实有接触的 1,497 个测试窗口；另有 784 个无接触窗口。重叠窗口不是独立实验，不能用 2,281 个窗口替代 21 条示范做成功率统计。')

    nextpage();title('预测示例与消融解释','03 / 对结果的边界保持清楚')
    story.append(pic(base/'figures/contact_predictions.png',maxheight=360))
    text('上排为示范接触软标签，下排为模型分数；绿色星号是最高分点。三个样本按误差分位选择，包含定位较差的情况，未只挑最好的图片。')
    story.append(p('不能把离散化收益全算作模型收益',SUB))
    text('32 / 64 / 128 点的平均定位误差为 3.477 / 3.060 / 2.862 单位。但即使总能选择距离真实接触最近的采样点，离散化下限也分别为 1.127 / 0.371 / 0.121。更密的采样本身就降低误差下限。')
    text('移除 flow 后仍有 90.71% 的定位命中率，说明当前推杆与几何信息已很强；加入 flow 后为 93.92%，同时无接触假阳性由 15.05% 降到 0.77%。本实验没有显示软标签一定优于硬标签，也没有显示展开时序一定优于 GRU。')
    text('主模型接触存在判断 F1 为 '+f"{main['presence_f1']:.4f}"+'。presence head 未用于在线动作选择或停止；在线控制器使用逐点排序，停止条件来自目标覆盖率和动作预算。有接触分数只用于辅助训练与离线诊断，不解释为成功概率。')

    nextpage();title('从接触分数到实际推动','04 / 阶段三：独立控制器与反馈执行')
    for heading,body in [
        ('1. 由目标构造 flow','读取当前物体位姿与固定目标；0.8 秒内的质心平移最多 24 单位，转角最多 0.24 弧度。'),
        ('2. 选择候选接触','按照规则分数或模型分数，保留 8 个空间分散的可接近轮廓点；两种接触方法使用相同后续控制器。'),
        ('3. 规划接近路径','将 T 形轮廓按推杆半径和安全余量膨胀，用可见图避开物体；接近目标速度为 120 单位/秒。'),
        ('4. 比较短推方案','每个候选测试 12 与 40 单位两种法向推动，8 个控制周期推动、2 周期保持。候选在独立物理世界中预测。'),
        ('5. 执行并重新规划','真实环境只接收推杆目标坐标；执行一段接近—短推—保持后重新读取状态。物体平移和转动由物理仿真产生。')]:
        story.append(p(heading,SUB));text(body)
    story.append(tab([['共享设置','实际取值'],['真实执行预算','每条最多 300 个动作周期（30 秒），达到覆盖率 0.95 停止。'],
        ['物理与接触','mass=1，moment=3000，space.damping=0，碰撞 shape friction=0，dt=0.01，每控制周期10子步。'],
        ['私有预测限制','复制位姿与速度，但不复制碰撞 warm-start 缓存；接触单步检查最大位置差 0.02865 单位。'],
        ['信息与计算条件','接触法使用准确位姿与候选物理预测；直接动作/视觉策略不做候选仿真。动作预算相同，计算预算不同。']], [115,408]))
    text('覆盖率定义为物体与目标的相交面积除以目标面积，不是IoU。没有把物体直接设置到目标，也没有用示范未来替代闭环反馈。测试之前已冻结控制器；测试后的诊断不用于修改控制器参数。')

    nextpage();title('视觉 Diffusion Policy 对照','05 / 阶段四：真实训练的轻量基线')
    text('扩散策略从带噪声的动作序列逐步恢复可执行动作。本实验复用官方 ConditionalUnet1D，视觉编码器从头训练。输入为最近两帧 96×96 RGB 图像和推杆实际坐标；源码固定版本见末页。')
    story.append(tab([['设置','本轮实际配置'],['网络','卷积视觉编码器 + 官方条件 U-Net，宽度64/128/256；'+f"{dm['parameters']:,}"+' 个参数。'],
        ['时序','2帧观测；预测16步动作；每次执行8步后重新观察。'],['扩散过程','训练100个噪声时间步；推理DDIM 20步；EMA验证损失选最佳权重。'],
        ['训练','完整训练示范，batch64，'+str(dp['epochs'])+'轮，'+str(dp['optimizer_steps'])+'次优化；用全部验证窗口选权重。'],
        ['实测资源',f"训练 {dp['wall_seconds']/60:.1f} 分钟；PyTorch最大张量显存 {dp['cuda_peak_allocated_bytes']/2**20:.1f} MiB（不含全部驱动/桌面开销）。"],
        ['结果',f"最佳验证轮数 {dp['best_epoch']}；测试离线首动作目标误差 {dpo['mean_first_action_l2']:.2f} 单位；闭环 {summaries['visual_dp_test']['successes']}/21。"]],[110,413]))
    story.extend([Spacer(1,12),pic(base/'figures/diffusion_training.png',maxheight=200)])
    text('训练损失继续降低，但验证误差在第10轮后整体上升，出现明显过拟合。因此测试使用验证集选出的第10轮权重，没有使用最后一轮，也没有根据测试结果重新挑选权重。')
    text('这是原生 Windows 上的轻量适配实验：CNN、网络宽度、训练轮数、数据划分和推理采样都与论文完整设置有差别。因此结果不能代表官方 Diffusion Policy 的性能上限，也不能据此宣称研究方法优于原论文。')

    nextpage();title('同一个测试初始状态','06 / 完整动画另存，报告显示最后一帧')
    story.append(pic(base/'figures/comparison_episode_12_final.png',width=510,maxheight=535))
    text('固定选择测试集第一条初始状态（episode 12），没有按哪个方法成功来挑场景。动画中达到成功的策略保持最后状态；其余策略继续执行至 300 步。绿色是目标区域，灰色为物体，蓝色为推杆。')
    text('跨方法比较需连同输入和控制器条件一起阅读。尤其是接触方法的内部物理预测，既贡献执行能力，也增加计算资源。')

    nextpage();title('失败案例与可信结论','07 / 不隐藏负结果，不把相关性写成因果')
    text('主接触模型有 6 条失败，最终朝向误差均较小（小于1.7°），但位置/重合度仍未达到0.95；规则有8条失败，其中部分仍有明显朝向误差。接近成功也按失败记录，例如规则 episode96 的最高覆盖率0.94709。')
    text('Flow→action 在本轮21条闭环测试中均未成功，其中9条没有发生接触且物体未移动。离线首动作误差较小不等于闭环能完成接近、推动和重新规划。这个结果限于本次模型和训练方式，不能推广到全部 flow→action 方法。')
    text('示范未来与目标规划flow的分布结构不同：训练窗口终点位移RMS中位数6.94，主模型在线为22.53。在线幅度仍在训练第1至99百分位内；这些诊断不能直接证明失败原因。')
    story.append(p('统计上能说什么',SUB))
    text('主模型15/21与规则13/21的配对结果中，只有学习方法成功3例，只有规则成功1例；精确McNemar p=0.625。三个训练种子的成功率有8.25个百分点样本标准差，说明稳定性仍需加强。')
    noflow=read(base/'contact_no_flow_test/summary.json')
    text(f"追加诊断：无flow接触模型使用相同控制器，仅成功{noflow['successes']}/21；主模型独有成功9例，无flow独有0例。它支持flow对本次闭环的作用，但这是主试验后追加的单种子消融，须与预先确定的主要结果区分。")
    pair=evidence['paired_primary_contact_minus_rule']
    text('按独立episode配对重采样的结果另存统计文件。窗口重叠、同场景多种子、单个checkpoint条件，都限制置信区间的解释；不把重复测试计成新的独立场景。')
    story.append(p('已经完成与尚未证明',SUB))
    text('已经完成：可运行训练与推理代码、10组接触/动作训练、视觉DP训练、固定测试集对照、3个接触训练种子、可视化、独立验收。尚未证明：跨本体迁移、真实机器人、视觉位姿估计鲁棒性、论文级最优性能，以及接触监督是否对应最优操作。')
    text('后续研究建议：优先检验闭环规划flow的训练覆盖，以及接触模型对控制器性能的独立贡献；再做不同推杆几何/动力学和视觉噪声的明确迁移实验。不能将这些建议列为已完成结果。')

    nextpage();title('重跑、交付与版本记录','08 / 从证据文件追溯每个结果')
    text('项目目录：.。完整PowerShell命令位于 STAGES234_README.md；本页给出从已保存权重重跑主方法的最短流程。')
    cmd="""Set-Location '.'
$py = '.\\.venv-train\\Scripts\\python.exe'
& $py -m pip check
& $py research\\evaluate_closed_loop.py `
  --method learned_contact --split test --episodes 21 `
  --factory research.contact_learning:load_policy `
  --checkpoint runs/stages234_20261022/contact_main/best.pt `
  --device cpu --output-dir runs/my_contact_retest --videos 3"""
    story.append(Preformatted(cmd,ParagraphStyle('Code',fontName='Courier',fontSize=8,leading=12,textColor=BLUE)))
    story.append(Spacer(1,10));story.append(tab([['版本 / 证据','记录'],['训练环境','Python3.12.14 / PyTorch2.7.1+cu126 / NumPy1.26.4 / diffusers0.35.1；RTX4060 Laptop，8GB。'],
        ['数据固定版本','7628202a2180972f291ba1bc6723834921e72c19'],['gym-pusht源码','e49cf07fd32ad6e69443af2921dd8343b0f9ad7d'],['Diffusion Policy源码','5ba07ac6661db573af695b419a7947ecb704690f'],
        ['依赖锁','requirements-training.lock.txt；第一阶段 .venv 保持不变。'],['验收',f"final_acceptance.json：{acceptance['status']}；旧阶段受保护文件 {acceptance['protected_stage1_files_unchanged']} 个哈希保持。"],
        ['实验目录','runs/stages234_20261022：配置、best.pt、history、逐例NPZ/JSON、动画、命令日志和统计。']],[115,408]))
    text('开发中发现离线与闭环日志同名，覆盖了三组旧闭环控制台日志；轨迹、权重和指标文件未变。已修复为独立命名前缀并拒绝覆盖，在新进程复跑四组测试、逐数组对比。新日志作为复验证据，样本数未增加。')
    story.append(p('官方来源',SUB))
    for label,url in [('PushT数据集','https://huggingface.co/datasets/lerobot/pusht'),('PushT环境','https://github.com/huggingface/gym-pusht'),('Diffusion Policy实现','https://github.com/real-stanford/diffusion_policy'),('项目介绍','https://diffusion-policy.cs.columbia.edu/'),('PyTorch固定版本安装','https://pytorch.org/get-started/previous-versions/')]:
        story.append(Paragraph(f'<link href="{url}" color="#177e89">{escape(label)}：{url}</link>',SMALL))
    output.parent.mkdir(parents=True,exist_ok=True)
    if output.exists():raise FileExistsError(output)
    doc=SimpleDocTemplate(str(output),pagesize=A4,rightMargin=36,leftMargin=36,topMargin=37,bottomMargin=46,
        title='PushT 接触学习：第二至四阶段实测报告',author='项目实验记录')
    doc.build(story,onFirstPage=footer,onLaterPages=footer)


def build_readme(base,path):
    sums={n:read(base/n/'summary.json') for n in ['rule_test','contact_test','flow_action_test','visual_dp_test']}
    dp=read(base/'visual_dp/training_result.json')
    lines=['# 第二至第四阶段：训练、闭环与对照实验','',
        '本轮是在第一阶段基础上实际运行的实验版。训练、验证和测试均有记录；下表来自固定21个测试初始状态，每条最多300步。二维仿真结果没有证明跨本体或真实机器人泛化。','',
        '| 方法 | 成功数 | 成功率 | 平均最终覆盖率 |','|---|---:|---:|---:|']
    for (name,s),label in zip(sums.items(),['规则接触+控制器','学习接触+控制器（主种子0）','Flow→action','轻量视觉Diffusion Policy']):
        lines.append(f"| {label} | {s['successes']}/21 | {percent(s['success_rate'])} | {s['mean_final_coverage']:.4f} |")
    lines+=['','接触模型三个训练种子分别15/21、12/21、15/21，均值66.67%、样本标准差8.25个百分点。同21个场景重复评测，不是63个独立场景。主模型比规则多成功2条，不能宣称成功率显著提升。',
        '', '主试验后补充的无flow闭环消融为6/21，相同控制器下主模型独有成功9例、无flow独有0例。单种子补充诊断支持flow作用，但不替换预先主模型或主要比较。证据在`contact_no_flow_test`。',
        '','接触方法使用准确物体位姿与独立候选物理预测；视觉方法仅使用图像和推杆位置。真实动作预算相同，信息和计算预算不同。DP采用官方U-Net源码，但编码器、网络规模、训练预算和采样设置经过适配，不是原论文完整复现。','',
        '## 先看结果','',
        '- [中文实验报告](output/pdf/PushT_第二至四阶段实验报告_2026-10-22.pdf)',
        '- [四方法同一场景动画](runs/stages234_20261022/figures/comparison_episode_12.gif)',
        '- [闭环对照图](runs/stages234_20261022/figures/closed_loop_comparison.png)',
        '- [离线消融图](runs/stages234_20261022/figures/offline_ablation.png)',
        '- [接触预测实例](runs/stages234_20261022/figures/contact_predictions.png)',
        '- [完整统计](runs/stages234_20261022/statistical_analysis.json)',
        '- [新进程验收](runs/stages234_20261022/final_acceptance.json)',
        '- [第二阶段样本](runs/stages234_20261022/stage2_samples/samples.npz)与[样本来源](runs/stages234_20261022/stage2_samples/manifest.json)',
        '- [第二至四阶段字段说明](docs/STAGES234_DATA_SCHEMA.md)',
        '- [协议](docs/STAGES234_PROTOCOL.md)、[字段](docs/STAGE1_DATA_SCHEMA.md)、[视觉DP细节](research/VISUAL_DP.md)','',
        '## 打开PowerShell并使用训练环境','',
        '不需要修改执行策略或激活脚本，直接使用环境中的Python：','',
        '```powershell',"Set-Location '.'","$py = '.\\.venv-train\\Scripts\\python.exe'",'& $py --version','& $py -m pip check','```','',
        '训练环境为Python3.12.14、torch2.7.1+cu126。已实际通过GPU前向/反向和CPU加载检查。第一阶段仍使用独立`.venv`，没有修改系统CUDA、驱动或WSL设置。','',
        '## 从保存的模型重跑闭环','',
        '每次使用新输出目录，程序拒绝覆盖既有结果。以下为同样的21条测试；重复运行用于核验，不增加科学统计样本量。','',
        '```powershell',"$run = 'runs/retest_' + (Get-Date -Format 'yyyyMMdd_HHmmss')",'& $py research/evaluate_closed_loop.py --method rule --split test --episodes 21 `','  --output-dir "$run/rule_test" --videos 3',
        '& $py research/evaluate_closed_loop.py --method learned_contact --split test --episodes 21 `','  --factory research.contact_learning:load_policy `','  --checkpoint runs/stages234_20261022/contact_main/best.pt `','  --device cpu --output-dir "$run/contact_test" --videos 3',
        '& $py research/evaluate_closed_loop.py --method flow_action --split test --episodes 21 `','  --factory research.contact_learning:load_policy `','  --checkpoint runs/stages234_20261022/flow_action/best.pt `','  --device cpu --output-dir "$run/flow_action_test" --videos 3',
        '& $py research/evaluate_closed_loop.py --method visual_dp --split test --episodes 21 `','  --factory research.visual_dp:load_policy `','  --checkpoint runs/stages234_20261022/visual_dp/best.pt `','  --device cuda --output-dir "$run/visual_dp_test" --videos 3','```','',
        '无GPU时视觉推理可改`--device cpu`，速度会变慢。接触/规则控制器每个接近-短推-保持段后重新规划；DP每8个动作重新观察。','',
        '## 从头重新训练','',
        '数据集已经下载并重新核验。先按完整episode固定划分，再生成窗口。新的训练输出使用新目录；不要改变测试集挑选结果。','',
        '```powershell',"$run = 'runs/train_' + (Get-Date -Format 'yyyyMMdd_HHmmss')",
        '& $py research/run_experiments.py --phase contact --epochs 40 --output $run',
        '& $py research/run_experiments.py --phase ablation --epochs 40 --output $run',
        '& $py research/run_experiments.py --phase evaluate --output $run',
        '& $py research/visual_dp.py prepare',
        '& $py research/visual_dp.py train --epochs 60 --batch-size 64 --device cuda `',
        '  --max-val-samples 0 --output "$run/visual_dp"',
        '& $py research/visual_dp.py evaluate --checkpoint "$run/visual_dp/best.pt" `',
        '  --split test --max-samples 0 --device cuda --output "$run/visual_dp/test_metrics.json"','```','',
        f"本轮DP训练60轮，共{dp['optimizer_steps']}次更新，实测{dp['wall_seconds']/60:.1f}分钟。接触训练使用逐点共享MLP；消融包括32/64/128点、软硬标签、仅终点、GRU和移除flow。所有best权重仅由验证集选取。",'',
        '迁移到另一台Windows机器时先安装兼容Python3.12，再创建`.venv-train`；从PyTorch官方CUDA12.6源安装torch2.7.1，再安装依赖锁。没有NVIDIA GPU时选择官方CPU wheel，不能照搬CUDA检测结论。','',
        '```powershell','py -3.12 -m venv .venv-train','& .venv-train/Scripts/python.exe -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu126','& .venv-train/Scripts/python.exe -m pip install -r requirements-training.lock.txt','& .venv-train/Scripts/python.exe -m pip check','```','',
        '精确本轮解析版本见`requirements-training.lock.txt`；Torch CUDA wheel需额外指定官方索引，不能仅依赖默认PyPI。CPU机器应先安装官方CPU wheel，再装`requirements-training.txt`，不要安装含cu126的本机CUDA锁文件。完整原始数据及固定源码的准备命令见`STAGE1_README.md`。','',
        '## 重新验收已保存结果','',
        '```powershell',"$out = 'runs/stages234_20261022/acceptance_' + (Get-Date -Format 'yyyyMMdd_HHmmss') + '.json'",'& $py research/verify_experiments.py --output $out','```','',
        '该验收在新进程检查逐例轨迹、独立重算覆盖率与在线flow、训练/验证/测试分组、模型权重哈希、验证集checkpoint选择、旧阶段文件哈希与日志修复证据。训练/测试实际命令和stdout/stderr保存在`runs/stages234_20261022/logs`；DP日志另见`visual_dp_train.log`及对应command记录。','',
        '一次开发缺陷使离线和闭环日志重名，三组旧闭环控制台日志被覆盖；轨迹、模型和指标没有覆盖。已修复唯一前缀+拒绝覆盖，并在`fresh_verification`新进程复跑四组、逐数组对比。重跑是补充证据，不伪装原始日志，也不计作独立测试。','',
        '安装过程中两个辅助包filelock/fsspec留下重复版本元数据。训练时实际使用版本的79个源码文件均匹配对应wheel记录；训练结束后保留备份并重新安装同版本，检查重复元数据为零。证据为`runtime_metadata_before_repair.json`与`runtime_metadata_after_repair.json`。未在训练中更换算法依赖，也未改动第一阶段环境。','',
        '## 数据与推理约定','',
        '- `states[t]`为动作前状态；`actions[t]`是绝对推杆中心目标；接触累计该动作的10个物理子步。',
        '- 模型监督为未来8周期接触区域并集；局部高斯软标签sigma5，无接触全零。加权BCE输出称分数，不是校准成功概率。',
        '- 离线flow可以来自示范未来；真实闭环只允许当前状态和目标规划flow。评测脚本不读取测试未来轨迹。',
        '- coverage是物体与目标相交面积/目标面积；success按>=0.95。图像IoU、定位命中率和任务成功率是三种不同指标。',
        '- 主闭环固定64点soft-sequence seed0。32/128只是离线消融；不能直接把其checkpoint传给固定64点控制器。','',
        '## 尚未证明的内容','',
        '跨本体和真实世界迁移、图像到物体位姿估计、最优接触监督、完整论文Diffusion Policy复现均未完成。离线定位改善不等于闭环稳定改善；三训练种子结果有波动。下一步应围绕规划flow分布、控制器独立贡献和推杆几何变化设计明确实验，而不是把负结果改写成成功。','']
    if path.exists():raise FileExistsError(path)
    path.write_text('\n'.join(lines),encoding='utf-8')


def main():
    a=argparse.ArgumentParser();a.add_argument('--run',default='runs/stages234_20261022');a.add_argument('--output',default='output/pdf/PushT_第二至四阶段实验报告_2026-10-22.pdf');args=a.parse_args()
    base=ROOT/args.run;out=ROOT/args.output
    build_pdf(base,out);build_readme(base,ROOT/'STAGES234_README.md');print(out,flush=True)

if __name__=='__main__':main()
