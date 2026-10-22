"""Create a portable, embedded-font Chinese progress report from audited data."""
from pathlib import Path
import json
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT, TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate, Frame, PageTemplate, Paragraph, Spacer, Table, TableStyle,
    Image, PageBreak, KeepTogether,
)
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/pdf/PushT_阶段工作报告_2026-10-22.pdf"
OUT.parent.mkdir(parents=True, exist_ok=True)
audit = json.loads((ROOT / "reports/data_audit.json").read_text(encoding="utf-8"))
replay = json.loads((ROOT / "reports/replay_validation.json").read_text(encoding="utf-8"))
manifest = json.loads((ROOT / "resource_manifest.json").read_text(encoding="utf-8"))
assert audit["status"] == "passed" and audit["episodes"] == 206

pdfmetrics.registerFont(TTFont("YaHei", "C:/Windows/Fonts/msyh.ttc", subfontIndex=0))
pdfmetrics.registerFont(TTFont("YaHeiBold", "C:/Windows/Fonts/msyhbd.ttc", subfontIndex=0))
pdfmetrics.registerFont(TTFont("Code", "C:/Windows/Fonts/consola.ttf"))
pdfmetrics.registerFontFamily("YaHei", normal="YaHei", bold="YaHeiBold", italic="YaHei", boldItalic="YaHeiBold")

INK = colors.HexColor("#172E45")
TEAL = colors.HexColor("#147E83")
MUTED = colors.HexColor("#526579")
PALE = colors.HexColor("#EEF5F6")
LINE = colors.HexColor("#D7E1E9")
W, H = A4
LEFT = 45
WIDTH = W - 2 * LEFT

styles = {
    "title": ParagraphStyle("title", fontName="YaHeiBold", fontSize=25, leading=35, textColor=INK, wordWrap="CJK", spaceAfter=10),
    "sub": ParagraphStyle("sub", fontName="YaHei", fontSize=11.5, leading=19, textColor=MUTED, wordWrap="CJK", spaceAfter=13),
    "h1": ParagraphStyle("h1", fontName="YaHeiBold", fontSize=18, leading=26, textColor=INK, wordWrap="CJK", spaceAfter=13),
    "h2": ParagraphStyle("h2", fontName="YaHeiBold", fontSize=11.5, leading=18, textColor=TEAL, wordWrap="CJK", spaceBefore=12, spaceAfter=5),
    "body": ParagraphStyle("body", fontName="YaHei", fontSize=10.2, leading=17.2, textColor=INK, wordWrap="CJK", spaceAfter=7),
    "small": ParagraphStyle("small", fontName="YaHei", fontSize=8.5, leading=13.3, textColor=MUTED, wordWrap="CJK", spaceAfter=5),
    "cell": ParagraphStyle("cell", fontName="YaHei", fontSize=9.1, leading=14.5, textColor=INK, wordWrap="CJK"),
    "th": ParagraphStyle("th", fontName="YaHeiBold", fontSize=9.2, leading=14.5, textColor=colors.white, wordWrap="CJK"),
    "code": ParagraphStyle("code", fontName="Code", fontSize=8.5, leading=13.3, textColor=INK),
    "caption": ParagraphStyle("caption", fontName="YaHei", fontSize=8.5, leading=13.5, textColor=MUTED, wordWrap="CJK", spaceAfter=9),
}

def p(text, style="body"):
    return Paragraph(text, styles[style])

def table(rows, widths, header=True, padding=8):
    wrapped = [[p(str(v), "th" if header and i == 0 else "cell") for v in row] for i, row in enumerate(rows)]
    t = Table(wrapped, colWidths=widths, hAlign="LEFT", repeatRows=1 if header else 0)
    spec = [("VALIGN", (0,0), (-1,-1), "TOP"), ("LEFTPADDING", (0,0), (-1,-1), 9),
            ("RIGHTPADDING", (0,0), (-1,-1), 9), ("TOPPADDING", (0,0), (-1,-1), padding),
            ("BOTTOMPADDING", (0,0), (-1,-1), padding), ("LINEBELOW", (0,0), (-1,-1), .4, LINE)]
    if header:
        spec += [("BACKGROUND", (0,0), (-1,0), TEAL), ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#F6F8FA")])]
    else:
        spec += [("BACKGROUND", (0,0), (-1,-1), PALE)]
    t.setStyle(TableStyle(spec))
    return t

def box(text):
    t = Table([[p(text)]], colWidths=[WIDTH])
    t.spaceAfter = 6
    t.setStyle(TableStyle([("BACKGROUND", (0,0), (-1,-1), PALE),
                          ("BOX", (0,0), (-1,-1), .5, LINE),
                          ("LEFTPADDING", (0,0), (-1,-1), 12), ("RIGHTPADDING", (0,0), (-1,-1), 12),
                          ("TOPPADDING", (0,0), (-1,-1), 10), ("BOTTOMPADDING", (0,0), (-1,-1), 7)]))
    return t

def image(path, width=WIDTH):
    obj = Image(str(ROOT / path))
    obj.drawHeight *= width / obj.drawWidth
    obj.drawWidth = width
    obj.hAlign = "CENTER"
    return obj

def page(canvas, doc):
    canvas.saveState()
    canvas.setFillColor(TEAL)
    canvas.rect(0, H - 8, W, 8, fill=1, stroke=0)
    canvas.setFont("YaHei", 8)
    canvas.setFillColor(MUTED)
    canvas.drawString(LEFT, H - 31, "FLOW / CONTACT     课程项目阶段记录")
    canvas.drawRightString(W - LEFT, H - 31, "2026.10.22")
    canvas.setStrokeColor(LINE)
    canvas.line(LEFT, 41, W - LEFT, 41)
    canvas.setFont("YaHei", 8)
    canvas.drawString(LEFT, 26, "范围：资源获取、仿真重建与数据审计")
    canvas.drawRightString(W - LEFT, 26, f"{doc.page} / 5")
    canvas.restoreState()

story = []
add = story.append
add(Spacer(1, 13))
add(p("从物体运动到接触", "title"))
add(p("PushT 数据准备与验证工作报告", "h1"))
add(p("两人课程项目 · 截至 2026 年 10 月 22 日\n<br/>基于课题《从物体运动到接触：面向跨本体操作的接触可供性学习》", "sub"))
add(box("<b>本阶段结论：</b>公开数据、仿真环境和基线代码已经取得；全部 206 条示范已完成回放、接触标签重建与审计，可以进入模型和控制器实验阶段。"))
add(Spacer(1, 14))
add(table([
    ["206 条", "25,650 帧", "24,000 个"],
    ["完整示范", "原始观测与动作", "可用的 8 步窗口"],
], [WIDTH/3]*3, header=False))
add(p("已经做完的工作", "h2"))
add(p("1. 获取 LeRobot PushT 全量示范及视频，下载 gym-pusht 和 Diffusion Policy 官方源码，固定数据与代码版本，并保存文件校验值。"))
add(p("2. 按原始初始化方式回放动作，恢复物体状态、64 个固定表面点的轨迹和碰撞接触；生成逐动作区间的接触软标签。"))
add(p("3. 对齐公开视频与仿真回放，用独立审计脚本检查字段、几何、接触和数据划分；生成可直接使用的训练窗口索引。"))
add(p("这项工作的用途", "h2"))
add(p("后续模型将学习：给定物体表面的期望运动，应该优先接触哪些位置；再由独立控制器将接触选择转换成推杆动作。最终要检验这种中间表示是否更容易适配不同执行器。"))
add(box("<b>当前边界：</b>尚未训练接触模型，没有学习策略成功率，也尚未证明跨本体迁移。课题作者内部处理文件的公开入口未获确认；本项目接触标签为独立仿真重建结果。"))
add(p("阅读导航：第 2 页重建方法；第 3 页验证图与指标；第 4 页数据划分和使用约定；第 5 页后续分工、公开来源与换机复现。", "small"))

add(PageBreak())
add(p("01  数据是怎样重建的", "h1"))
add(p("公开数据提供推杆状态、动作和视频。训练所需的物体位姿、稳定身份的表面点及接触信息，通过固定版本的物理仿真回放补齐。"))
add(table([
    ["步骤", "本次实现"],
    ["恢复初始化", "按示范编号恢复原始随机状态；保留旧环境的位置／角度设置顺序与物理参数，减少版本差异。"],
    ["逐动作回放", "控制频率 10 Hz，物理频率 100 Hz；每个动作执行 10 个物理子步，同时记录状态并核对视频时间戳。"],
    ["记录接触", "只保留推杆与 T 形物体的接触；在碰撞回调中立即保存世界／局部坐标、法向和冲量信息。"],
    ["生成训练标签", "沿物体外轮廓等弧长取 64 点；以当前动作区间内的外表面接触为中心，生成 sigma=5 的高斯软标签。"],
], [85, WIDTH-85]))
add(p("每条示范包含什么", "h2"))
add(p("设 L 为动作数、N=64。每条示范单独保存为 episode_XXXXXX.npz，可用 NumPy 读取，无需启用 pickle。", "small"))
add(table([
    ["字段", "形状", "用途"],
    ["states", "(L+1, 5)", "推杆位置、物体原点位置及角度"],
    ["points_world", "(L+1, N, 2)", "身份固定的物体表面点轨迹"],
    ["actions", "(L, 2)", "公开示范的推杆目标位置"],
    ["contact_events", "变长记录", "100 Hz 子步中的接触点、法向和冲量"],
    ["contact_soft_labels", "(L, N)", "每个动作区间访问过的接触区域"],
    ["有效性标记", "两组 (L,)", "标记标签异常与缺少视频终点的区间"],
], [151, 90, WIDTH-241]))
add(p("不能弄错的三个约定", "h2"))
add(p("<b>时间：</b>states[t] 在 action[t] 执行前；标签 t 汇总该动作的十个物理子步。8 步窗口需要 9 帧几何。"))
add(p("<b>坐标：</b>物体位置是 body origin，不是质心；局部质心为 (0, 45)。归一化时，整个窗口统一变换到起始帧坐标系。"))
add(p("<b>冲量：</b>同一接触组的总冲量会重复保存在多个接触点上，求和前需按 episode、action_index、manifold_index 去重。零冲量记录仍是几何接触，不代表有效推进。"))

add(PageBreak())
add(p("02  验证结果与图像证据", "h1"))
add(table([
    ["验证项", "本次结果", "可以支持的判断"],
    ["可见物体掩码平均 IoU", f"{replay['visible_mask_iou_mean']:.4f}", "回放与公开视频在图像上较为一致"],
    ["最低 IoU / 低于 0.8", f"{replay['visible_mask_iou_min']:.4f} / 2 帧", "保留低一致性样本，未按 IoU 筛选"],
    ["推杆位置最大误差", "2.14 × 10<super>-5</super>", "动作时序与推杆回放高度一致"],
], [166, 104, WIDTH-270]))
add(Spacer(1, 9))
add(image("reports/replay_comparison.png", 365))
add(p("图 1　一般帧示例：左为公开示范，右为物理回放。示范 0、第 81 帧，当前帧物体掩码 IoU=0.952。绿色区域表示目标，蓝色圆点表示推杆。", "caption"))
add(image("reports/worst_00_episode_095_frame_105.png", 365))
add(p("图 2　最低 IoU 帧：示范 95、第 105 帧，IoU=0.794。橙点表示前一动作区间访问过的接触位置，映射到当前物体表面。", "caption"))
add(p("独立审计脚本已通过：10 个源文件的 SHA256 校验、206 条示范的动作与字段核对、刚体点变换、接触法向／冲量符号、软标签重算和划分互斥检查。刚体点最大误差为 2.99 × 10<super>-5</super>，软标签最大误差为 2.98 × 10<super>-8</super>。位置误差使用仿真坐标单位。", "small"))
add(p("解释限制：原视频为 96×96 有损视频。IoU 不是模型准确率、任务成功率或精确位姿误差；审计说明数据内部一致，不等于独立接触真值或第三方认证。", "small"))

add(PageBreak())
add(p("03  训练数据如何使用", "h1"))
add(p("已按完整示范划分训练、验证和测试集，随机种子为 20261022。重叠窗口始终留在同一集合内，避免相邻片段泄漏；没有按示范成功率或图像 IoU 筛选。"))
add(table([
    ["集合", "示范数", "有效 8 步窗口"],
    ["训练", str(audit["episode_split_counts"]["train"]), f"{audit['window_split_counts']['train']:,}"],
    ["验证", str(audit["episode_split_counts"]["validation"]), f"{audit['window_split_counts']['validation']:,}"],
    ["测试", str(audit["episode_split_counts"]["test"]), f"{audit['window_split_counts']['test']:,}"],
    ["合计", "206", "24,000"],
], [WIDTH/3]*3))
add(p("哪些窗口被排除了", "h2"))
add(p("候选窗口共 24,208 个，排除 208 个。其中 206 个包含示范最后动作后的状态，该状态没有原视频终点可核验；另 2 个包含初始重叠引起的物体内部拼接缝接触。16 条接缝记录仍保留在原始重建结果中并标记。"))
add(p("119,158 条接触点记录中，有 374 条零冲量记录，保留为几何接触。记录来自重复的物理子步，不能理解成同等数量的独立接触位置；24,000 个重叠窗口也不是独立统计样本。"))
add(p("一个训练样本的约定", "h2"))
add(box("以起始动作 s 为锚点，H=8：<br/>当前几何：points_world[s]<br/>未来 8 帧：points_world[s+1:s+9]<br/>接触监督：contact_soft_labels[s:s+8]<br/>输入轨迹可整理为 (64, 8, 2)，接触热图为 (64, 8)。"))
add(p("闭环评测时，期望轨迹必须由当前物体状态与目标生成，不能读取测试示范的未来轨迹。仿真重建得到的示范接触可用于诊断控制器，但不是部署时可获得的信息。"))
add(p("本地成果索引", "h2"))
add(table([
    ["项目相对路径", "内容"],
    ["data/raw/lerobot_pusht/", "原始动作、推杆状态、视频与元数据"],
    ["data/derived/replayed_contacts/", "206 个重建示范文件"],
    ["data/derived/splits.json、windows.json", "固定示范划分和有效窗口索引"],
    ["reports/replay_validation.json", "回放一致性、异常统计与版本信息"],
    ["reports/data_audit.json", "审计结果、逐示范检查与校验值"],
], [260, WIDTH-260], padding=7))

add(PageBreak())
add(p("04  后续安排与跨设备接续", "h1"))
add(p("先把接触变成能执行的动作", "h2"))
add(p("第一步在对齐的示范状态上用重建接触调试接近、短推和撤离；随后接入目标驱动的闭环。这样可以区分预测错误与控制器执行失败。基础闭环稳定后，再冻结接触网络、更换执行器，测成功率、额外数据和适配成本。"))
add(table([
    ["分工", "负责内容与交付"],
    ["成员 A", "Flow → Contact 和 Flow → Action；先做简单网络，统一数据划分，再做时序、软标签和数据量消融。"],
    ["成员 B", "接近／短推／撤离／重规划控制器及统一评测；规则接触与学习接触使用同一控制器。"],
    ["共同完成", "视觉 Diffusion Policy 对照、失败归因和报告；跨本体实验放在基础闭环之后，3D/ALOHA 后续扩展。"],
], [76, WIDTH-76]))
add(p("公开资源与固定版本", "h2"))
resources = [
    ("LeRobot PushT 数据", "https://huggingface.co/datasets/lerobot/pusht", manifest["dataset_revision"]),
    ("gym-pusht 仿真环境", "https://github.com/huggingface/gym-pusht", manifest["repositories"]["gym-pusht"][1]),
    ("Diffusion Policy 基线", "https://github.com/real-stanford/diffusion_policy", manifest["repositories"]["diffusion_policy"][1]),
]
for name, url, revision in resources:
    add(p(f'<b>{name}</b>　<a href="{url}" color="#147E83">{url}</a><br/>版本：<font name="Code" size="8">{revision}</font>', "small"))
add(p("resource_manifest.json 记录下载地址、固定版本、字节数和 SHA256。官方源码和许可证保留在 external/；接触重建与审计脚本由本项目补充。", "small"))
add(p("换设备后如何继续", "h2"))
add(p("本 PDF 已嵌入中文字体与对比图，可独立阅读；它不包含完整数据和代码。若要运行实验，请另行复制 flow_contact_project 项目，至少携带 scripts/、requirements-data.lock.txt 和 resource_manifest.json；无需复制原设备的 .venv。", "small"))
commands = [
    "python -m venv .venv",
    ".\\.venv\\Scripts\\python.exe -m pip install -r requirements-data.lock.txt",
    ".\\.venv\\Scripts\\python.exe scripts\\acquire_resources.py",
    ".\\.venv\\Scripts\\python.exe scripts\\reconstruct_contacts.py",
    ".\\.venv\\Scripts\\python.exe scripts\\audit_dataset.py",
]
code = "<br/>".join(escape(line) for line in commands)
t = Table([[p(code, "code")]], colWidths=[WIDTH])
t.spaceAfter = 6
t.setStyle(TableStyle([("BACKGROUND", (0,0), (-1,-1), PALE), ("LEFTPADDING", (0,0), (-1,-1), 9),
                      ("RIGHTPADDING", (0,0), (-1,-1), 9), ("TOPPADDING", (0,0), (-1,-1), 9), ("BOTTOMPADDING", (0,0), (-1,-1), 9)]))
add(t)
add(p("以上命令在 Windows PowerShell 的项目目录中执行，建议使用 Python 3.12。下载资源需要联网；训练依赖仍需单独配置，尤其是 Diffusion Policy 的训练环境。", "small"))

doc = BaseDocTemplate(str(OUT), pagesize=A4, title="从物体运动到接触：PushT 数据准备与验证工作报告", author="项目工作记录", subject="206 条示范的获取、仿真接触重建、审计与后续计划", pageCompression=1)
doc.addPageTemplates(PageTemplate(id="report", frames=[Frame(LEFT, 53, WIDTH, H-107, leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)], onPage=page))
doc.build(story)
reader = PdfReader(OUT)
assert len(reader.pages) == 5, f"Unexpected page count: {len(reader.pages)}"
texts = [page.extract_text() for page in reader.pages]
for needle in ["119,158", "24,000", "0.9608", "19,130", "尚未训练", "Python 3.12"]:
    assert needle in "\n".join(texts), needle
print(json.dumps({"path": str(OUT), "pages": len(reader.pages), "bytes": OUT.stat().st_size, "page_text_lengths": list(map(len, texts))}, ensure_ascii=True))
