# Flow → Contact：PushT 课程项目

本目录已取得公开示范、仿真环境和 Diffusion Policy 官方代码，并实现了物体点轨迹与接触标签的重建流程。当前阶段是数据准备与核验；尚未训练接触模型，也没有学习策略的成功率结果。

## 当前完成结果（2026-10-22）

- 全部 206 条示范、25,650 帧完成回放，重建 119,158 条物理子步接触点记录。
- 独立审计通过：源文件校验、动作时间对齐、刚体点变换、接触法向与冲量符号、软标签重算和划分无泄漏检查均通过。
- 回放与公开视频的物体可见掩码平均 IoU 为 0.9608；这是图像一致性指标，不是模型准确率或任务成功率。
- 训练 / 验证 / 测试按整条示范划分为 164 / 21 / 21；对应 horizon=8 的有效窗口为 19,130 / 2,589 / 2,281，共 24,000 个。重叠窗口不等于独立样本。
- 从 24,208 个候选窗口中排除 208 个：206 个包含未核验的末端状态，2 个包含内部拼接缝接触。没有按成功率或 IoU 筛选数据。

打开 [回放演示](reports/replay_preview.gif) 可对比公开示范和重建结果；完整检查见 [审计报告](reports/data_audit.json)。

## 已取得的资源

| 内容 | 本地位置 | 来源 |
|---|---|---|
| 206 条示范、25,650 帧动作/推杆状态、完整视频和元数据 | `data/raw/lerobot_pusht/` | [LeRobot PushT](https://huggingface.co/datasets/lerobot/pusht) |
| PushT 仿真环境完整源码 | `external/gym-pusht/` | [huggingface/gym-pusht](https://github.com/huggingface/gym-pusht) |
| Diffusion Policy 完整源码与 PushT 配置 | `external/diffusion_policy/` | [real-stanford/diffusion_policy](https://github.com/real-stanford/diffusion_policy) |
| 重建后的物体状态、点轨迹、碰撞记录与接触软标签 | `data/derived/replayed_contacts/` | 本项目从公开动作在固定仿真环境中重建 |

所有来源、下载大小、SHA256 和版本记录在 `resource_manifest.json`。下载脚本固定使用相同版本：

- 数据：`7628202a2180972f291ba1bc6723834921e72c19`
- gym-pusht：`e49cf07fd32ad6e69443af2921dd8343b0f9ad7d`
- diffusion_policy：`5ba07ac6661db573af695b419a7947ecb704690f`

PDF 提及的作者处理结果没有公开下载入口被确认。本目录的标签是独立重建的结果，不声称与作者内部文件逐项相同。Diffusion Policy 旧站点的原始 PushT zip 地址在本次访问时返回 404；当前流程使用可获取的 LeRobot 数据。

## 运行流程

当前电脑已创建项目独立环境 `.venv`，可以在本目录使用 PowerShell：

```powershell
.\.venv\Scripts\python.exe scripts\reconstruct_contacts.py
.\.venv\Scripts\python.exe scripts\audit_dataset.py
```

下载全部公开资源的可复现命令为：

```powershell
.\.venv\Scripts\python.exe scripts\acquire_resources.py
```

换一台电脑时，推荐 Python 3.12，创建独立环境并安装 `requirements-data.lock.txt`，随后依次运行获取、重建、审计脚本。这里只安装数据准备依赖；Diffusion Policy 的训练环境仍需单独配置。

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-data.lock.txt
.\.venv\Scripts\python.exe scripts\acquire_resources.py
.\.venv\Scripts\python.exe scripts\reconstruct_contacts.py
.\.venv\Scripts\python.exe scripts\audit_dataset.py
```

快速检查可用 `reconstruct_contacts.py --max-episodes 3`。探测输出保存在独立的 `probe_3_episodes` 目录，不覆盖全量结果。全量流程逐帧解码视频并核对时间戳，直接使用基础环境，避免环境包装器的默认步数上限截断示范。

## 数据含义与时间对齐

每条 `episode_XXXXXX.npz` 使用 NumPy 数组，读取时无需开启 pickle。设该示范有 L 条动作，N=64 个沿外轮廓等弧长采样点。

| 字段 | 形状 | 含义 |
|---|---|---|
| `states` | `(L+1, 5)` | 推杆 x/y、物体 body origin x/y、角度；第 t 行在 action t 执行前 |
| `velocities` | `(L+1, 5)` | 推杆线速度、物体线速度、角速度；使用 Pymunk 的速度定义 |
| `local_boundary_points` | `(N, 2)` | 固定身份的物体局部外轮廓点，不采样内部拼接缝 |
| `points_world` | `(L+1, N, 2)` | 相同物体点在每个时刻的世界坐标 |
| `actions` | `(L, 2)` | 公开数据的推杆目标位置，控制频率 10 Hz |
| `contact_events` | 变长结构化数组 | 每个 100 Hz 物理子步的推杆与物体碰撞记录 |
| `contact_soft_labels` | `(L, N)` | action t 的十个子步中访问过的外表面接触区域，局部距离高斯软标签，sigma=5 |
| `contact_label_valid` | `(L,)` | 此动作区间的接触监督是否可用；存在内部接缝接触的区间为 False |
| `transition_has_video_endpoint` | `(L,)` | 此区间结束时是否有对应原视频帧；每条示范最后一项为 False |
| `reference_visible_mask_iou` | `(L,)` | 基于已知渲染颜色的物体掩码重合度 |
| `reference_strict_color_mask_iou` | `(L,)` | 保留的早期单颜色阈值指标，供比较分割方法的影响 |

位姿中的物体位置是 body origin，不要直接当成质心。物体局部 CoG 为 `(0,45)`。`states[L]` 与 `points_world[L]` 是执行最后动作后的仿真延续，缺少原视频终点验证；标准训练窗口会排除这个终点。

碰撞记录只保留推杆与 T 形物体的接触，排除墙面接触。`normal_into_block` 是从推杆指向物体的世界坐标法向。`point_local` 在碰撞回调中立即转换，避免后续物体运动导致坐标错配。

`arbiter_total_impulse` 是整组接触作用于 shape A 的冲量，`impulse_on_block` 已换算符号。它们在同一组的各接触点上重复保存；求和前必须按 **episode + action_index + manifold_index** 去重，不能把每个点的值直接相加。`on_exterior` 和 `distance_to_exterior` 标识接触是否在外轮廓上。零冲量接触仍被视作几何接触，不能解释为有效推进。

## 训练划分与窗口

审计已生成 `data/derived/splits.json` 与 `windows.json`：按完整 episode 固定划分 164 条训练、21 条验证、21 条测试，随机种子为 `20261022`。不按成功率或图像重合度筛选示范。

每个 horizon=8 的窗口覆盖 8 个动作/接触区间，需读取 9 帧几何：当前帧作为锚点，后面 8 帧作为期望轨迹。以下是索引约定，具体窗口记录格式见生成文件：

```python
import numpy as np
episode = np.load("data/derived/replayed_contacts/episode_000000.npz")
s, H = 10, 8
geometry = episode["points_world"][s]                  # (N, 2)
future = episode["points_world"][s+1:s+H+1]            # (H, N, 2)
X = future.transpose(1, 0, 2)                         # (N, H, 2)
A = episode["contact_soft_labels"][s:s+H].T           # (N, H)
```

使用物体坐标归一化时，将整个窗口统一变换到第 s 帧的坐标系，不能逐帧分别对齐。窗口清单排除任何接触标签无效或没有视频终点的区间，且不会跨 episode。

## 如何解释验证结果

`reports/replay_validation.json` 记录逐 episode 的视觉一致性、推杆误差、异常接触数、版本和脚本校验值。`reports/data_audit.json` 记录独立的几何、数据字段、资源校验和训练划分检查。`reports/replay_preview.gif` 展示第一条示范与重建回放，橙点为前一动作区间访问过的接触位置映射到当前物体表面。

推杆状态吻合支持动作时序与 PD 回放一致；视频掩码重合度支持物体运动在可见图像上接近。96×96 有损视频无法独立证明物体位姿精确一致，更不能证明接触标签唯一最优。接触、flow 和速度都来自同一次固定仿真回放，属于一致的仿真监督。

原数据末尾的 `next.done`、`next.reward` 有重复值，部分 `next.success` 为 False；本项目没有把这些示范宣称为全部达到 95% 成功阈值。报告分别保存最后一个有视频对应状态的覆盖率和最后动作后未核验的覆盖率。今后策略评测应在统一测试环境中重新计算成功判定。

## 两人小组的后续分工

下一阶段先建立一个可以归因的闭环：在对齐的示范状态上，用重建接触标签调试控制器，确认它能接近并执行短推；随后用规则接触和预测接触进入目标驱动闭环。真实闭环中的期望轨迹应由当前物体位姿与目标位姿生成，不能读取测试示范的未来轨迹。示范接触只用于诊断控制器，不能当成部署时可获得的信息。

1. 一人负责接触模型与 Flow→Action：共享数据划分与编码器，先建立简单网络，再做时空注意力、软标签和训练数据量的消融。
2. 一人负责控制器与评测：先完成接近、短推、撤离换点和重新规划；用规则接触和学习接触接同一个控制器，建立统一测试初始状态。
3. 共同完成视觉 Diffusion Policy 对照、失败归因和报告。两种 flow 方法的整个系统应使用相同的几何、期望运动和执行状态；视觉基线的额外信息差异应单独说明。

基础闭环稳定后，冻结接触网络，更换执行器与控制器，测成功率、额外数据和适配成本。3D/ALOHA 放在这一验证之后。
