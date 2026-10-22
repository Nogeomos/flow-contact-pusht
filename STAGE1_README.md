# 第一阶段：原生 Windows 的 PushT 数据与推动实验

本入口对应本轮重新执行的第一阶段。先检查了目录和上级目录，未发现适用的 `AGENTS.md`。已有源码、原始数据、独立环境和历史报告均保留；复用前进行了校验。**本轮结果来自新的运行进程和 `runs/stage1_20261022_02/`，没有将历史处理结果当作本轮实验结果。**

本轮没有训练模型，也没有安装 PyTorch、CUDA Toolkit 或修改系统配置。第一阶段在原生 Windows 完成，不需要 WSL2。

## 先看结果

| 已经运行验证的内容 | 结果 |
|---|---|
| 环境与依赖 | Python 3.12.14 项目独立环境；锁文件版本核对及 `pip check` 通过 |
| 原始资源 | 206 条、25,650 帧原始记录；10 个源文件校验通过，两个源码目录共 384 个归档成员匹配 |
| 本轮完整示范回放 | 仅前 3 条，共 420 帧，2,380 条子步接触点记录 |
| 回放与原视频 | 平均可见物体掩码 IoU 0.957751；推杆位置最大误差 2.05884e-5 仿真单位 |
| 无接触监督 | 208 个动作区间无接触，接触软标签全为 0 |
| 编号点与检查样本 | 64 个固定材料点；一条有接触、一条整窗无接触的 8 步样本 |
| 简单推动 | 70 步、7 秒；物体质心移动 47.90062 仿真单位，转动 18.45439° |
| 可重复性 | 两个新进程中，推动数据的全部 16 个数组逐项完全一致 |
| 正式验收 | 9 条子进程命令、11 项检查通过；242 个既有文件哈希未变 |

IoU 衡量两幅图中物体可见区域的交集/并集，不是模型准确率或任务成功率。位置误差不等于物体位姿误差。接触标签来自固定仿真器，尚无独立接触真值。

可以直接打开：

- [接触、编号轮廓点与离线 flow 图片](runs/stage1_20261022_02/visualization/contact_sample.png)
- [无接触图片](runs/stage1_20261022_02/visualization/no_contact_sample.png)
- [接触与 flow 动画](runs/stage1_20261022_02/visualization/contact_flow_preview.gif)
- [原视频与回放对比动画](runs/stage1_20261022_02/replay/reports/probe_3_episodes/replay_preview.gif)
- [简单推动动画](runs/stage1_20261022_02/simple_push_a/simple_push.gif)
- [阶段记录、限制及第二阶段建议](docs/STAGE1_RECORD_2026-10-22.md)
- [字段和坐标说明](docs/STAGE1_DATA_SCHEMA.md)
- [正式验收记录与实际命令](runs/stage1_20261022_02/acceptance.json)

## 打开 PowerShell 并重新运行

打开 Windows Terminal 的 PowerShell 标签页，输入：

```powershell
Set-Location '.'
$py = '.\.venv\Scripts\python.exe'
& $py --version
& $py -m pip check
& $py scripts\run_stage1.py --offline
```

最后一条命令会创建新的 `runs/stage1_日期_时间/`，依次校验资源、检查环境和数据、回放 3 条示范、绘图、运行两次短推并比较轨迹。程序会打印目录位置。各步骤在新的 Python 进程中执行；标准输出、错误输出、退出码和实际命令保存在该目录的 `logs/` 和 `acceptance.json`。

这里直接指定独立环境的 Python，相当于明确选择该环境；不需要执行 `Activate.ps1`，也不需要改变 PowerShell 的执行策略。系统 PATH 中的 `python` 是 3.11.9，不能把它与本项目实际使用的 3.12.14 混淆。

如需同时重新在线核对官方来源：

```powershell
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$sources = "runs\sources_$stamp"
& $py scripts\check_official_sources.py --output-dir $sources
& $py scripts\run_stage1.py --offline --official-check "$sources\official_sources.json"
```

在线核验需要访问 Hugging Face、GitHub 原始文件域名和项目官网。离线验收不需要网络，但原始资源必须已经存在。下载缺失文件可运行下一节的资源准备命令并省略 `--offline`；已有文件不匹配时程序报错保留原文件，不会覆盖。

## 分开运行每一个环节

以下命令使用新的时间戳目录，不会覆盖正式验收结果：

```powershell
$run = 'runs\manual_' + (Get-Date -Format 'yyyyMMdd_HHmmss')
& $py scripts\prepare_resources.py --output-dir "$run\resources" --offline
& $py scripts\check_environment.py --output-dir "$run\inspection"
& $py scripts\inspect_data.py --output-dir "$run\inspection"
& $py scripts\replay_stage1.py --episodes 3 --points 64 --output-dir "$run\replay"
& $py scripts\visualize_contacts.py `
  --input "$run\replay\data\derived\probe_3_episodes\episode_000000.npz" `
  --output-dir "$run\visualization"
& $py scripts\simple_push.py --output-dir "$run\simple_push"
```

`replay_stage1.py` 限制为 1 至 10 条，默认 3 条；它会再次校验原始资源和源码，然后重新回放，不读取旧的 NPZ。`--points` 可调整采样数量，改变它属于新的设计实验。各命令对已存在的目标文件或输出目录拒绝覆盖；重跑时更换 `$run`。

各脚本职责：

| 脚本 | 功能 |
|---|---|
| `check_environment.py` | 当前 OS、CPU、RAM、GPU、驱动、磁盘、Python、Git 和依赖导入检查 |
| `check_official_sources.py` | 在线核对固定版本元数据/README 和项目官网，记录 HTTP 状态与 SHA256 |
| `prepare_resources.py` | 校验清单；只下载/解压缺失文件；拒绝覆盖不同内容的文件 |
| `inspect_data.py` | 读取原始字段、时间戳和真实物理参数；核查 DP 环境/配置需求 |
| `replay_stage1.py` | 小规模完整示范回放和独立几何/接触审计 |
| `visualize_contacts.py` | 世界/局部坐标可视化、64 点编号、flow 箭头、接触软标签与检查样本 |
| `simple_push.py` | 不训练模型的确定性接近、短推、等待和撤离 |
| `run_stage1.py` | 新进程执行主要流程、记录日志、检查版本锁与重复性 |

其中回放和几何审计复用了现有 `reconstruct_contacts.py`、`audit_dataset.py` 的函数；这两个文件未修改。原有 `acquire_resources.py` 会重写解压目标，第一阶段请使用新的 `prepare_resources.py`。

## 环境与安装记录

本轮实测 Windows 11 25H2，i9-13980HX，32 个逻辑处理器，物理内存约 15.63 GiB；RTX 4060 Laptop，8188 MiB 显存，驱动 596.49。检查时 C 盘空闲约 85 GiB。动态空闲内存/磁盘值以每次环境报告为准。

本轮复用并验证现存 `.venv`，没有重新安装或更改依赖。实际版本：

| 依赖 | 安装分发版本 |
|---|---|
| Python | 3.12.14 |
| NumPy | 1.26.4 |
| Gymnasium | 0.29.1 |
| Pymunk | 6.6.0 |
| Pygame | 2.5.2 |
| Shapely | 2.0.6 |
| OpenCV | 4.10.0.84 |
| PyArrow | 17.0.0 |
| PyAV | 16.0.1 |
| Pillow | 11.3.0 |

完整依赖见 `requirements-data.lock.txt`。Pymunk 的安装元数据为 6.6.0，但模块运行时 `version` 属性显示 6.6.2；环境报告同时保留这两项，不将它们混为同一种版本依据。gym-pusht 从固定提交的 `external/gym-pusht/` 导入，其项目元数据为 0.1.8，未与随机最新版本混用。

只有在新设备或 `.venv` 不存在时才需要创建环境。先安装/定位 Python 3.12，并把 `$python312` 改成实际绝对路径；本机已核验可用的基础解释器如下：

```powershell
$python312 = '<USER_HOME>\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
& $python312 --version
if (Test-Path -LiteralPath '.\.venv') { throw '已有 .venv，请复用并检查，不要覆盖。' }
& $python312 -m venv .venv
$py = '.\.venv\Scripts\python.exe'
& $py -m pip install -r requirements-data.lock.txt
& $py -m pip check
& $py scripts\prepare_resources.py --output-dir ('runs\setup_' + (Get-Date -Format 'yyyyMMdd_HHmmss'))
& $py scripts\run_stage1.py --offline
```

上面是新环境安装步骤；**本轮已运行的是既有独立环境的验证与实验流程，并未在另一设备重装验证**。跨设备复制项目时不要复制 `.venv`；至少携带 scripts、锁文件和 `resource_manifest.json`，资源可以按固定清单重新获取。PDF 生成脚本属于历史工具，其依赖不在第一阶段数据环境中。

## 数据含义：先分清三个量

- **实际推杆位置**：当前时刻蓝色推杆中心在哪里，对应原始 `observation.state` 的两个数。
- **动作指令**：下一控制周期希望推杆追踪的绝对目标坐标 `[x,y]`，不是位移或速度。底层 PD 控制器依据位置误差和速度更新推杆；实际位置不会瞬间等于目标。
- **物体 flow**：同一个物体材料点随时间的位移/轨迹。它描述物体如何运动，不直接指定推杆目标。

原始 parquet 有 `observation.state`、`action`、`episode_index`、`frame_index`、`timestamp`、`next.reward`、`next.done`、`next.success`、`index`、`task_index`。图像存在独立的 96×96、10 fps AV1 视频中。**原始文件没有物体位置/朝向字段，也没有接触标签。**

实际下载的元数据为 LeRobot v3.0，网页数据卡仍含 v2.0 格式示例；代码按真实 v3.0 文件布局读取，没有直接套用网页旧示例。

64 个点由本项目在 T 形外轮廓等弧长采样。固定局部点经过 `p_world = R(theta) @ p_local + body.position` 转到世界坐标，编号不随帧变化。图像中 x 向右、y 向下，角度单位为弧度，正角在屏幕上呈顺时针；长度是 512×512 场景内的仿真单位，不是米。

接触记录来自推杆与物体的真实仿真碰撞回调，汇总一个 0.1 秒动作内的十个物理子步；不使用“无论是否接触都找最近点”的伪标签。没有接触时标签全部为零。有接触时，按局部接触位置到轮廓点的距离产生高斯软标签。它说明示范接触过哪里，不能解释为最优接触或成功概率。

## Diffusion Policy 当前核验边界

固定官方源码的完整仿真环境面向 Linux/NVIDIA，原配置为 Python 3.9、PyTorch 1.12.1、CUDA Toolkit 11.6；与本项目的数据环境分开。官方 PushT 示例采用 batch 64、8 个数据加载进程、horizon 16、2 帧观测、执行 8 步动作，并读取 Zarr 数据。

当前没有 Torch 或 `nvcc`，也没有实测默认模型显存、训练速度或 Windows 兼容性。`nvidia-smi` 能识别显卡只说明驱动可用；其 CUDA Version 不等于已安装 CUDA Toolkit。官方这些文件没有给出明确的最低 RAM/显存要求，不能据此承诺 8 GB 能运行原始默认配置。后续需要单独环境、数据适配和小批量显存试跑。本轮不启用 WSL2，不进行完整 DP 训练。

来源：[PushT 数据](https://huggingface.co/datasets/lerobot/pusht)、[gym-pusht](https://github.com/huggingface/gym-pusht)、[Diffusion Policy 实现](https://github.com/real-stanford/diffusion_policy)、[项目介绍](https://diffusion-policy.cs.columbia.edu/)。固定版本和 SHA256 在 `resource_manifest.json`；本轮在线核验在 `reports/stage1_sources_20261022/official_sources.json`。
