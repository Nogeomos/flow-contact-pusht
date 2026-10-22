# 第一阶段数据字段与坐标说明

本文对应本轮实际运行目录 `runs/stage1_20261022_02`。本轮新回放了编号 0、1、2 的 **3 条完整示范，共 420 帧**，记录了 2,380 条接触点事件；没有使用历史处理结果作为本轮实验结果。原始公开数据确实含 206 条示范、25,650 行，但“原始数据已读取”不等于“本轮已全部回放”。证据见[本轮数据检查](../runs/stage1_20261022_02/inspection/data_inspection.json)和[新回放审计](../runs/stage1_20261022_02/replay/fresh_audit.json)。

这里的“接触标签”来自固定版本仿真器的碰撞回调，不是数据集提供的独立人工标注，也不是模型预测。第一阶段尚未训练模型。

## 1. 原始公开数据实际提供什么

数值表在[原始 Parquet 文件](../data/raw/lerobot_pusht/data/chunk-000/file-000.parquet)，一行对应一个动作时刻。以下字段均已实际读取；括号内为一行的形状。

| 字段 | 内容与解释 |
|---|---|
| `observation.state`（2） | 推杆实际中心的世界坐标 `[x, y]`，不是 T 形物体的位姿。本轮用完整回放核验了前 3 条示范的对齐。 |
| `action`（2） | 推杆希望到达的**绝对目标坐标** `[x, y]`。不是位移、速度或物体目标位姿。 |
| `episode_index`（标量） | 示范编号；每条完整示范是一段独立轨迹。 |
| `frame_index`（标量） | 该示范内部的帧编号，从 0 开始。 |
| `timestamp`（标量） | 该示范内的时间，单位秒，约等于 `frame_index / 10`。 |
| `index`（标量） | 整份数据中的行索引。 |
| `task_index`（标量） | 任务编号，关联任务元数据。 |
| `next.reward`（标量） | 数据集保存的奖励值。不能用它代替接触标注；终端行语义需单独核验。 |
| `next.done`、`next.success`（布尔标量） | 数据集保存的终止、成功标志，两者含义不同。末端重复/转换约定使其不能单独证明末动作之后的状态已被观测。 |

图像单独存放于 [MP4 视频](../data/raw/lerobot_pusht/videos/observation.image/chunk-000/file-000.mp4)，不是上述 Parquet 中的逐行像素数组。其逻辑字段名为 `observation.image`，每帧为 `96×96×3`，10 FPS，AV1 有损压缩。[示范元数据](../data/raw/lerobot_pusht/meta/episodes/chunk-000/file-000.parquet)记录每条示范的行范围和视频起止时间。[info.json](../data/raw/lerobot_pusht/meta/info.json)记录本版本的数据布局。

原始数值表**没有直接给出**物体位置/角度、完整初始物理状态、64 个轮廓点或接触事件。本轮通过原采集程序的随机初始化规则、示范编号和固定仿真版本重建它们，再与原视频、推杆实际坐标对照。这个方法在本轮 3 条示范上得到支持，不能据此宣称每个接触都是独立真值。

动作目标与推杆实际位置不同是正常现象：底层 PD 控制器根据位置误差和当前速度更新推杆运动；目标点可以领先于实际推杆。物体如何平移和转动由碰撞与物理仿真决定。

## 2. 先理解坐标、参考点和固定编号

世界画布范围约为 `512×512` 仿真长度单位。屏幕上 `+x` 向右、`+y` 向下。位置没有标定为米；96×96 视频像素与 512 单位世界坐标不是同一尺度。时间单位为秒，角度单位为弧度；在这个屏幕坐标约定下，正角度看起来是顺时针旋转。

局部坐标跟着物体一起运动。T 形外轮廓按下面的顶点顺序定义：

```text
(-60, 0), (60, 0), (60, 30), (15, 30),
(15, 120), (-15, 120), (-15, 30), (-60, 30)
```

沿这条外轮廓等弧长采样 64 点，编号为 0 到 63；当前轮廓周长 480，因此相邻采样间隔为 7.5 仿真单位。64 是本项目选定的采样参数，不是原始数据已有的字段。第 i 个点始终对应同一个物体材料位置，物体旋转时不会重新排序。

设局部点是列向量 q，物体姿态为参考点坐标 t 和角度 θ，则：

```text
R(θ) = [[cos θ, -sin θ],
        [sin θ,  cos θ]]
p_world = R(θ) q + t
```

在 NumPy 中把 N 个点存成 `N×2` 的行向量矩阵 Q，公式写作 `Q @ R.T + t`。

这里 t 是仿真器的 `body.position`，即局部原点的世界位置。局部原点 `(0, 0)` 位于横条上边的中心，**不是质心**。仿真器实际采用的局部质心是 `(0, 45)`，其世界位置应另算 `R(θ) @ [0, 45] + t`。它也不同于这个几何并集按面积计算的形心 `(0, 40.7142857...)`；本轮保留上游设定，没有为了匹配视频修改它。

重置函数中设置位置、角度的顺序以及一次初始物理步会影响结果。因此 `reset_input` 是传入重置函数的参数，不应直接当成实际第一帧；真正的第一帧请读 `states[0]`。

## 3. 一条新回放 NPZ 的字段

以[示范 0 的新回放](../runs/stage1_20261022_02/replay/data/derived/probe_3_episodes/episode_000000.npz)为例，L=161 个动作，N=64 个点。NPZ 是多个命名 NumPy 数组打包后的文件；可用 `np.load(..., allow_pickle=False)` 读取。

| 字段 | 形状、类型 | 含义 |
|---|---|---|
| `episode_index` | 标量，int32 | 来源示范编号。 |
| `reset_input` | 5，float64 | 重置输入：推杆 x/y、物体参考点 x/y、物体角度。实际初始化结果看 `states[0]`。 |
| `states` | `(L+1, 5)`，float64 | 每行依次为推杆实际 x/y、物体参考原点 x/y、物体角度。角度归入 `[0, 2π)`。 |
| `velocities` | `(L+1, 5)`，float64 | 推杆 `body.velocity` 的 x/y、物体 `body.velocity` 的 x/y、物体角速度。线速度用仿真单位/秒，角速度用弧度/秒；不要把物体 body 速度不加区分地当作偏置参考原点的差分速度。 |
| `local_boundary_points` | `(N, 2)`，float64 | 固定材料点的局部坐标。 |
| `points_world` | `(L+1, N, 2)`，float32 | 每帧固定点的世界坐标，由刚体变换得到。float32 存储会引入约 `1e-5` 量级的数值舍入。 |
| `actions` | `(L, 2)`，float64 | 从原数据读取的推杆绝对目标坐标。 |
| `contact_events` | `(M,)`，结构化数组 | 所有物理子步的推杆—物体接触点事件，M 可与 L 不同；详细字段见下一节。 |
| `contact_soft_labels` | `(L, N)`，float32 | 每个动作区间内的接触邻域分数；无接触区间整行严格为 0。 |
| `contact_label_valid` | `(L,)`，bool | 若某区间出现不属于 T 外轮廓的内部拼接缝接触，则该区间为假。标签有效不等于动作成功。 |
| `transition_has_video_endpoint` | `(L,)`，bool | 该动作后的状态是否有原视频帧可核验。每条示范最后一个值为假，后续窗口应排除它。 |
| `reference_visible_mask_iou` | `(L,)`，float64 | 原视频与回放图像中可见物体掩码的交并比，范围 0 到 1。这里用渲染颜色分类近似得到掩码，不是精确位姿真值或模型准确率。 |
| `reference_strict_color_mask_iou` | `(L,)`，float64 | 保留的另一种固定颜色阈值掩码对照指标，可能受填充色和视频压缩影响。 |
| `agent_position_error` | `(L,)`，float64 | 回放动作前的推杆中心与原表对应坐标的欧氏距离，单位为仿真长度单位。 |

虽然有 L+1 个仿真状态，原数据只有 L 帧观测。多出的最后一个状态由最后动作产生，不能自动视为原示范已观测状态。本轮检查样本不会使用以此状态为终点的窗口。

## 4. 接触事件和时间对齐

控制频率为 10 Hz，每个动作保持 0.1 秒；物理步长为 0.01 秒，因此一个动作包含 10 个物理子步。一个子步可以出现多个接触点，也可能完全无接触。

```text
states[t] / points_world[t]：动作执行前的观测
              ↓ 提交 actions[t]
       物理子步 0, 1, ..., 9
       收集 contact_events[action_index == t]
       汇总成 contact_soft_labels[t]
              ↓
states[t+1] / points_world[t+1]：动作执行后的状态
```

`contact_events` 是结构化数组，可用 `events["point_world"]` 访问其中一个字段。

| 事件字段 | 含义 |
|---|---|
| `action_index` | 属于哪个动作区间 t。 |
| `substep` | 区间内物理子步编号 0 到 9。 |
| `manifold_index` | 本动作内碰撞回调接触组的编号；同一组的多个接触点共享它。 |
| `point_world`（2） | 该子步碰撞回调中，位于物体一侧的接触点世界坐标。 |
| `point_local`（2） | 用该事件时刻的物体姿态将接触点转换到局部坐标后的材料位置。 |
| `normal_into_block`（2） | 世界坐标中的单位法向，方向由推杆指向物体内部。 |
| `arbiter_total_impulse`（2） | 回调提供的整个接触组冲量，对 shape A 取向。 |
| `impulse_on_block`（2） | 将上述冲量按物体所在 A/B 一侧修正符号，表示施加给物体的冲量。 |
| `block_is_shape_a` | 物体是否为该回调中的 shape A。 |
| `signed_separation` | 两侧接触点的有符号分离距离；负值表示几何穿透。 |
| `distance_to_exterior` | `point_local` 到 T 外轮廓线段集合的最短距离。 |
| `on_exterior` | 上述距离小于 `1e-4`，用于排除两个矩形形状内部的拼接缝接触。 |

冲量不是力；若要估计平均力，还需要考虑作用时间。`arbiter_total_impulse` 与 `impulse_on_block` 都是**整个接触组的冲量**，会在同组每个接触点记录上重复。求和前必须按 `(episode_index, action_index, manifold_index)` 去重，不能直接对所有事件求和。

不同子步的 `point_world` 对应不同物体姿态；把所有点画到一张动作前的图上是时间汇总示意。若用 `point_local` 重新投影到当前姿态，只表示“物体的哪些材料位置曾接触”，不能声称它们在当前瞬间同时接触。图中的接触法向也属于事件发生时刻。

## 5. 软标签：把接触点变成 64 个数

令 q_i 为第 i 个固定局部轮廓点，C_t 为动作 t 的所有有效外轮廓接触局部坐标集合。当前实现使用 σ=5 个仿真长度单位：

```text
C_t 非空：y[t, i] = exp(- min_{c ∈ C_t} ||q_i - c||² / (2 × 5²))
C_t 为空：y[t, i] = 0
```

也就是：靠近该动作内接触过的位置，分数较高；距离越远，分数越小。它不是对 64 点做总和为 1 的概率归一化。采样点不一定恰好落在接触点上，所以最大值也不一定等于 1。

高斯函数在数学上有长尾，数值上还可能下溢到 0。因此“有多少非零分数”不是一个可靠的接触区域大小指标。本轮无接触区间的全零标签来自显式分支，而不是强制选择距离推杆最近的点。

这里计算的是局部二维欧氏距离，不是沿轮廓走过的距离。T 形凹角附近，欧氏距离可能把两个沿轮廓相隔更远的位置混在同一邻域中。第二阶段可以比较沿轮廓弧长的平滑方式，但本轮没有开展这项实验。

这些事件是**仿真器报告的几何接触**。零冲量接触也可能存在；几何接触不保证产生有效推动，更不代表“此点最优”或“推动成功概率”。所有标签的可信程度还依赖回放初始状态和物理版本的一致性。

## 6. 两个可检查的 8 步窗口

[samples.npz](../runs/stage1_20261022_02/visualization/samples.npz)中 K=2、H=8、N=64。两个样本都来自本轮新回放的示范 0；只是检查样本，**没有在此创建训练/验证/测试划分**。

- 样本 0：起点 t=50。几何使用帧 50 到 58，共 9 帧；动作与标签使用 50 到 57，共 8 步。起点动作有 5 条接触事件，未来 0.8 秒的轮廓点平均位移约 35.0204 单位。查看[有接触图](../runs/stage1_20261022_02/visualization/contact_sample.png)。
- 样本 1：起点 t=0。几何使用帧 0 到 8；动作与标签使用 0 到 7。整个窗口都没有接触，8×64 个标签全部为 0。查看[无接触图](../runs/stage1_20261022_02/visualization/no_contact_sample.png)。

两条窗口的 `contact_label_valid` 和 `transition_has_video_endpoint` 都为真。相应[验证记录](../runs/stage1_20261022_02/visualization/visualization_validation.json)也保存了输入与脚本哈希、选择索引和投影误差。[动画](../runs/stage1_20261022_02/visualization/contact_flow_preview.gif)展示接近到接触的变化。

| 样本文件字段 | 形状 | 内容 |
|---|---|---|
| `episode_index` | 标量 | 两条窗口的来源示范编号。 |
| `sample_start_index` | K | `[50, 0]`。 |
| `local_boundary_points` | N×2 | 固定局部点。 |
| `anchor_states` | K×5 | 每条窗口第一帧状态。 |
| `states` | K×(H+1)×5 | 9 帧状态。 |
| `points_world` | K×(H+1)×N×2 | 9 帧固定点坐标。 |
| `flow_from_anchor` | K×H×N×2 | 每个未来帧的点坐标减去窗口第一帧点坐标。不是相邻帧位移，也不是速度。 |
| `actions` | K×H×2 | 8 步推杆绝对目标坐标。 |
| `contact_soft_labels` | K×H×N | 8 步接触分数。 |
| `contact_label_valid` | K×H | 标签区间有效标志。 |
| `transition_has_video_endpoint` | K×H | 动作终点有视频可核验的标志。 |
| `contact_event_count` | K×H | 各动作区间的接触点事件数。 |

示范未来运动可用于训练和离线分析；真实闭环测试中的期望 flow 必须由当前状态、目标和规划器生成，不能读取测试示范未来轨迹。后续训练需先按完整示范划分集合，再在各集合内构造窗口，以免同一轨迹的重叠窗口泄漏。

## 7. 人工短推演示的额外字段

[simple_push.npz](../runs/stage1_20261022_02/simple_push_a/simple_push.npz)是另外生成的可复现仿真演示，不是公开示范的重建。它有 70 个动作、71 帧状态，总时长 7 秒。

其 `states`、`velocities`、`actions`、`local_boundary_points`、`points_world`、`contact_events`、`contact_soft_labels`、`contact_label_valid` 与前文意义相同，但该文件的 `points_world` 保存为 float64。额外或不同的字段如下：

| 字段 | 形状 | 内容 |
|---|---|---|
| `reset_input` | 5 | 人工初始条件 `[100, 215, 260, 200, 0]`。 |
| `time_s` | 71 | 每帧时间，从 0 到 7 秒，间隔 0.1 秒。 |
| `block_pose` | 71×3 | 物体参考原点 x/y 与**未取模**角度；跨越 2π 时比 `states[:, 4]` 更便于看连续旋转。 |
| `block_center_of_gravity_world` | 71×2 | 物体质心世界轨迹，已与参考原点轨迹分开保存。 |
| `pusher_position_actual` | 71×2 | 推杆实际中心轨迹，与 `states[:, :2]` 一致。 |
| `phases` | 70 | 每步的控制阶段字符串。 |
| `chosen_contact_local` | 2 | 人工指定的左侧接触位置 `[-60, 15]`；真实接触位置仍由碰撞产生，可能随着物体转动改变。 |
| `has_contact` | 70 | 本区间是否出现推杆—物体碰撞事件；为假时 64 个标签全零。 |

`phases` 依次为 `approach`（接近 2 秒）、`wait_before_push`（等待 0.5 秒）、`push`（向右推 3 秒）、`wait_after_push`（等待 0.5 秒）和 `retreat`（撤离 1 秒）。这些阶段的目标路径由程序指定；物体质心位移和角度变化来自仿真。指定目标路径的速度不等于实际推杆瞬时速度。

这个演示没有原始视频可供对比，因此也没有 `transition_has_video_endpoint` 或回放 IoU 字段；它验证的是仿真、坐标和控制接口，没有验证模型学会接触预测，也没有给出目标任务成功率。完整参数和数值结果见[simple_push.json](../runs/stage1_20261022_02/simple_push_a/simple_push.json)。

## 8. 少量 Python 读取示例

先按项目说明启动独立 Python 环境并进入项目根目录。以下代码可放进 Python 交互环境或脚本中运行。

```python
from pathlib import Path
import numpy as np

run = Path("runs/stage1_20261022_02")
path = run / "replay/data/derived/probe_3_episodes/episode_000000.npz"
with np.load(path, allow_pickle=False) as z:
    print(z.files)
    t = 50
    print("actual pusher:", z["states"][t, :2])
    print("action target:", z["actions"][t])
    events = z["contact_events"]
    current_events = events[events["action_index"] == t]
    print("event count:", len(current_events))  # 5
    print("local contacts:", current_events["point_local"])
    # 先检查有效性，再使用这一行标签。
    assert z["contact_label_valid"][t]
    assert z["transition_has_video_endpoint"][t]
    print("64 scores:", z["contact_soft_labels"][t])
```

```python
with np.load(run / "visualization/samples.npz", allow_pickle=False) as z:
    print(z["points_world"].shape)       # (2, 9, 64, 2)
    print(z["contact_soft_labels"].shape)  # (2, 8, 64)
    assert np.all(z["contact_soft_labels"][1] == 0)
    computed_flow = z["points_world"][:, 1:] - z["points_world"][:, :1]
    assert np.allclose(computed_flow, z["flow_from_anchor"])

with np.load(run / "simple_push_a/simple_push.npz", allow_pickle=False) as z:
    cog_displacement = z["block_center_of_gravity_world"][-1] - z["block_center_of_gravity_world"][0]
    angle_change = z["block_pose"][-1, 2] - z["block_pose"][0, 2]
    print("physical CoG displacement:", cog_displacement)
    print("physical rotation in radians:", angle_change)
    assert np.all(z["contact_soft_labels"][~z["has_contact"]] == 0)
```

上述数组说明对应本轮已生成文件。改变点数、标签方法或数据版本后，先查看实际数组形状和版本记录，不能继续假设形状永远是 64 点或 8 步。
