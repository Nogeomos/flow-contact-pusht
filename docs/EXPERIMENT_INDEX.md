# 实验导航与复现入口

本页区分已经运行的结果、精选发布材料与需要重新生成的本地文件。实验截止日期为 2026-10-22；总览见[项目 README](../README.md)。

在另一台电脑实际安装和恢复时，请优先遵循[Windows 复现主入口](REPRODUCING.md)。本页主要提供实验导航、结果解释和补充命令；[移植检查记录](../artifacts/portability/verification.json)限定了已实测范围，复用了现有依赖，不是第二台电脑或全新 CPU wheel 安装的证明。

## 阅读顺序与发布范围

先读[实验协议](STAGES234_PROTOCOL.md)，再看根 README 的四方法结果表、[失败分析](STAGES234_LIMITATIONS_AND_FAILURES.md)及[视觉 DP 适配说明](../research/VISUAL_DP.md)。字段分别见[阶段 1](STAGE1_DATA_SCHEMA.md)和[阶段 2–4](STAGES234_DATA_SCHEMA.md)。

| 想查看的内容 | 已发布入口 |
|---|---|
| 离线阅读报告 | [前期工作 PDF](../artifacts/stage1/report.pdf)、[第二至四阶段 PDF](../artifacts/stages234/report.pdf) |
| 四方法结果与同场景过程 | [对照图](../artifacts/stages234/figures/closed_loop_comparison.png)、[动画](../artifacts/stages234/figures/comparison_episode_12.gif)、[精确统计](../artifacts/stages234/statistical_analysis.json) |
| 接触预测与消融 | [预测实例](../artifacts/stages234/figures/contact_predictions.png)、[消融图](../artifacts/stages234/figures/offline_ablation.png) |
| 原工作区正式验收结果 | [验收 JSON](../artifacts/stages234/final_acceptance.json) |
| 墙边与墙角补充诊断 | [图](../artifacts/wall_diagnostics/wall_cases.png)、[数值](../artifacts/wall_diagnostics/summary.json)、[10 Hz 净空审计](../artifacts/wall_diagnostics/original_rollout_clearance_audit.json) |

发布仓库中的 `artifacts/` 保存精选图表、汇总和小样本，不上传整个本地 `runs/`。发布附件分为：

| 内容 | 用途 |
|---|---|
| `artifacts/stage1/`、`artifacts/stages234/`、`artifacts/wall_diagnostics/` | 快速查看已运行结果和诊断 |
| `artifacts/models/` | 模型清单；权重文件在发布附件中 |
| `models.zip` | 11 个检查点，解压到仓库根目录后恢复原 `runs/` 路径 |
| `evidence.zip` | 完整正式证据的脱敏副本与 206 条回放，保留原相对路径 |

只查看结果无需安装环境。重跑闭环需要源码、划分和权重；从头训练还需要原始示范、回放和窗口。保留原路径不等于所有原机器命令都能直接复制：脱敏后的绝对路径与本机路径不同，应使用本页的相对路径命令。

具体文件与校验值以[发布总清单](../artifacts/PUBLICATION_MANIFEST.json)、[附件清单](../artifacts/RELEASE_ASSETS.json)和[模型清单](../artifacts/models/manifest.json)为准。[发布页 v1.0.0-experiments](https://github.com/Nogeomos/flow-contact-pusht/releases/tag/v1.0.0-experiments)需要私有仓库访问权限。上游许可及本项目许可状态见[第三方说明](../THIRD_PARTY_NOTICES.md)。

将两个 ZIP 下载到仓库外的同一个目录，例如相邻的 `flow-contact-release-downloads` 文件夹，再在仓库根目录执行：

```powershell
py -3.12 scripts/verify_publication.py --release-dir ../flow-contact-release-downloads --require-clean-git
```

该[发布校验脚本](../scripts/verify_publication.py)根据清单核对发布字节与附件，并要求 Git 工作区干净；它不重新训练、不执行示范回放，也不证明模型成功率。要检查算法行为，继续按后文运行物理演示或闭环评测。

## 阶段 1：数据、坐标、标签与物理接口

公开示范总数为 206 条、25,650 帧。原始动作是推杆的绝对目标中心坐标，观测中的推杆位置是实际坐标，两者不同。原始数据没有直接提供 T 形物体位姿与接触真值；本项目从固定版本仿真器回放得到这些字段，再与原视频比较。

第一阶段的新进程验收重新回放前三条，共 420 帧、2,380 条物理子步接触点记录。可见掩码平均 IoU 为 0.957751，推杆位置最大误差为 2.05884e-5 世界单位。70 步简单推动使物体质心移动 47.90062 单位、转动 18.45439°；两次独立进程的全部 16 个保存数组完全一致。它是人工接触与控制接口演示，不是模型训练成果。

后续训练采用全量 206 条已重建并重新审计的数据，共 119,158 条子步接触点记录；全量可见掩码平均 IoU 约 0.9608。两个 IoU 来自不同范围，不应混用。视频有损压缩和遮挡使该指标不能证明精确物体位姿，也不能证明接触标签具有唯一最优性。

## 阶段 2：接触学习与离线消融

数据按完整 episode 划分为 164 / 21 / 21 条，种子 20261022。排除无视频终点或含内部拼接缝接触的窗口后，8 步窗口共 24,000 个：训练 19,130、验证 2,589、测试 2,281。其中测试有接触窗口 1,497 个，无接触窗口 784 个。

主模型固定为 64 点、软标签、完整 8 步 flow、训练种子 0。监督目标是未来八个动作区间接触区域的并集，无接触时全零；模型输出的是分数而非校准成功概率。presence 辅助输出参与训练和离线分析，当前控制器只消费点分数。

| 配置（种子 0） | 有接触窗口 top-1 距离 ≤10 比例 | 平均接触距离 |
|---|---:|---:|
| 主模型：64 点、软标签、8 步序列 | 93.92% | 3.05985 |
| 32 点 | 91.65% | 3.47655 |
| 128 点 | 93.05% | 2.86244 |
| 硬标签 | 94.12% | 2.96942 |
| 仅终点 flow | 93.25% | 3.01197 |
| GRU 时序编码 | 95.12% | 2.95974 |
| 移除 flow | 90.71% | 4.47728 |

表中距离使用仿真长度单位。更多采样点会降低离散候选集合本身的最小可能距离，因此不能只比较绝对距离就认定网络更好。32/128 点只用于离线消融，当前闭环接口固定 64 点。主模型没有根据测试表改选。

## 阶段 3：独立控制器与在线 flow

在线 flow 由当前物体状态和目标构造，不读取测试示范未来。模型输入当前物体几何、推杆位置与规划 flow；随后控制器最多考察 8 个空间分离的接触点和 2 种推动长度，以可视图安排接近路径，并在私有仿真环境中预测候选结果。真实环境只执行推杆动作，不强制改物体状态。

接近、短推和保持的整个动作队列完成后才重规划。轨迹另存的逐控制帧 flow 仅供诊断；实际模型输入在逐计划记录中。候选物理预测没有复制碰撞求解器的 warm-start 缓存，接触检查中单步位置最大差约 0.02865，角度最大差约 0.0001506 弧度；它是近似预测，最终指标来自真实执行。

规则与学习接触共享该控制器，动作上限都是 300。Flow → action 和视觉 DP 没有这部分候选仿真搜索，因此计算预算不同。视觉 DP 还没有接触方法使用的真值物体位姿。

## 阶段 4：闭环对照与失败

| 方法 | 成功数 / 21 | 平均最大覆盖率 | 平均最终覆盖率 |
|---|---:|---:|---:|
| 规则接触 | 13 | 0.88327 | 0.87116 |
| 学习接触主模型 | 15 | 0.94303 | 0.93754 |
| Flow → action | 0 | 0.17435 | 0.11719 |
| 轻量视觉 DP | 0 | 0.30828 | 0.08686 |

覆盖率达到 0.95 即成功；它是相交面积/目标面积，不是 IoU。原环境严格大于 0.95 的判据另存。规则成功率 Wilson 95% 区间为 40.88%–79.25%，主模型为 50.04%–86.19%。配对成功比较 p=0.625，不能认定成功率显著提升。接触三个训练种子成功率均值 66.67%、样本标准差 8.25 个百分点，重复使用同一组 21 场景。

主模型失败在 71、89、102、115、145、162。它们最终角度误差均小于 1.7°，但形状覆盖率仍未达阈值。Flow → action 的九条失败全程无接触，物体没有运动。视觉 DP 在 60 轮、19,200 次更新后，以验证去噪误差选取第 10 轮权重；离线八步动作误差低于保持当前位置的基线，却没有转化成闭环成功。接口检查未发现 RGB/BGR、坐标缩放、x/y 或观测动作错位。

事后补充的无 flow 闭环诊断成功 6/21；主模型相对它独有成功 9 条、反向 0 条。这是单种子探索诊断，不替换预先四方法主结果。它与墙边两例诊断均不增加原测试场景数量。

## 环境与复现步骤

所有命令从仓库根目录的 PowerShell 运行。使用新输出目录，保留旧结果。以下是代码的复现入口；跨机器重新安装、重新训练不承诺逐浮点比特一致。

### 1. 数据与仿真环境

安装 Python 3.12 后：

```powershell
py -3.12 -m venv .venv
$pyData = '.\.venv\Scripts\python.exe'
& $pyData -m pip install -r requirements-data.lock.txt
& $pyData -m pip check

$run = 'runs/reproduce_' + (Get-Date -Format 'yyyyMMdd_HHmmss')
& $pyData scripts/prepare_resources.py --output-dir "$run/resources"
& $pyData scripts/check_environment.py --output-dir "$run/environment"
& $pyData scripts/inspect_data.py --output-dir "$run/inspection"
& $pyData scripts/run_stage1.py --offline --output-dir "$run/stage1"
```

`prepare_resources.py` 从 [resource_manifest.json](../resource_manifest.json) 下载缺失文件并按固定 SHA256 校验，恢复 `data/raw/`、`downloads/` 与 `external/`。首次需要网络。环境检查使用本机真实结果，不假设有 GPU。第一阶段环境没有 PyTorch；可独立执行全部数据/仿真演示。

全量训练数据需要从附件恢复或另行重建。仓库保留固定的 `splits.json` 与 `windows.json`；新的包装脚本在独立目录重建并检查成员一致，保留这两份清单的历史来源记录。**已有全量成果或从 evidence.zip 恢复时跳过 `--install`，不用重建覆盖历史文件。**

```powershell
& $pyData scripts/rebuild_derived.py --output-dir "$run/rebuilt_data" --install
```

这会生成独立重建目录与 `fresh_derived_audit.json`，并在目标目录尚不存在时安装 `data/derived/replayed_contacts/`。已经有回放数组时，可去掉 `--install` 进行独立重建比较。底层重建与审计函数保持不变；不要直接重跑旧 `audit_dataset.py` 来覆盖仓库中的冻结清单。三条示范的第一阶段探测子集不能代替 206 条训练数据。

### 2. 独立训练环境：GPU 或 CPU

本轮真实训练使用 Python 3.12.14、RTX 4060 Laptop 8 GB、PyTorch 2.7.1+cu126。可用 NVIDIA GPU 且驱动兼容时：

```powershell
py -3.12 -m venv .venv-train
$py = '.\.venv-train\Scripts\python.exe'
& $py -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu126
& $py -m pip install -r requirements-training.lock.txt
& $py -m pip check
```

无 NVIDIA GPU 时另选 CPU wheel，不安装含 `+cu126` 的 CUDA 锁文件：

```powershell
py -3.12 -m venv .venv-train
$py = '.\.venv-train\Scripts\python.exe'
& $py -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
& $py -m pip install -r requirements-training.txt
& $py -m pip check
& $py -c "import torch; print(torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

两段安装命令二选一。CPU 安装路径是移植方案，本轮已运行 CPU 模型加载与闭环推理；完整 60 轮视觉训练是在 GPU 上完成的。这里的 `requirements-training.txt` 固定主要依赖，CPU 分支的传递依赖不等于原 GPU 环境的完整锁。项目不修改系统驱动、CUDA Toolkit 或 WSL 配置。

发布还提供 [setup_windows.ps1](../scripts/setup_windows.ps1) 自动安装入口，支持 `DataOnly`、`Cpu`、`Cuda126` 和只读 `-CheckOnly`。它默认选择项目内环境，保留无自身标记的既有环境；CPU/CUDA 模式分别安装对应 Torch wheel，再安装其他锁定依赖。`DataOnly` 默认 `.venv`，与历史第一阶段验收的环境路径一致。使用该入口后，将 `$py` 指向其实际报告的 Python 路径。

### 3. 从头训练并做离线评估

确保上一节的 206 条重建数据和划分已生成。下面会实际训练，不能只读结果时顺手运行：

```powershell
$run = 'runs/train_' + (Get-Date -Format 'yyyyMMdd_HHmmss')
& $py research/run_experiments.py --phase contact --epochs 40 --output $run
& $py research/run_experiments.py --phase ablation --epochs 40 --output $run
& $py research/run_experiments.py --phase evaluate --output $run
& $py research/visual_dp.py prepare
& $py research/visual_dp.py train --epochs 60 --batch-size 64 --device cuda --max-val-samples 0 --output "$run/visual_dp"
& $py research/visual_dp.py evaluate --checkpoint "$run/visual_dp/best.pt" --split test --max-samples 0 --device cuda --output "$run/visual_dp/test_metrics.json"
```

CPU 机器将两条视觉命令的 `--device cuda` 改为 `--device cpu`。接触训练自动选择可用设备。`contact` 阶段训练主接触和直接动作模型；`ablation` 阶段训练额外两个种子及全部消融。checkpoint 只按验证集选择；不要用测试表挑选新的主模型。

### 4. 闭环重跑

若使用发布权重，推荐按附件清单校验后通过 [restore_artifact.py](../scripts/restore_artifact.py) 恢复，已有不同文件会保留并报错。示例在仓库根目录执行，ZIP 位于前述相邻下载文件夹：

```powershell
$assets = Get-Content artifacts/RELEASE_ASSETS.json -Raw | ConvertFrom-Json
$restore = 'runs/restore_' + (Get-Date -Format 'yyyyMMdd_HHmmss')
foreach ($name in @('models.zip', 'evidence.zip')) {
  $asset = $assets.assets | Where-Object { $_.filename -eq $name }
  if (@($asset).Count -ne 1) { throw "Missing or duplicate asset: $name" }
  & $py scripts/restore_artifact.py --zip "../flow-contact-release-downloads/$name" --sha256 $asset.sha256 --output "$restore/$name.json"
  if ($LASTEXITCODE -ne 0) { throw "Restoration failed: $name" }
}
```

恢复后模型沿用原实验相对路径，设置：

```powershell
$models = 'runs/stages234_20261022'
```

若从上一节重新训练，则改为 `$models = $run`。目录中应有 `contact_main/best.pt`、`flow_action/best.pt`、`visual_dp/best.pt`。闭环还需要 `data/derived/splits.json` 与固定版本上游源码；可以从证据附件恢复划分，或按前述步骤生成。在线模型不读取示范未来。

```powershell
$eval = 'runs/closed_loop_' + (Get-Date -Format 'yyyyMMdd_HHmmss')
& $py research/evaluate_closed_loop.py --method rule --split test --episodes 21 --max-steps 300 --device cpu --output-dir "$eval/rule_test" --videos 3
& $py research/evaluate_closed_loop.py --method learned_contact --split test --episodes 21 --factory research.contact_learning:load_policy --checkpoint "$models/contact_main/best.pt" --max-steps 300 --device cpu --output-dir "$eval/contact_test" --videos 3
& $py research/evaluate_closed_loop.py --method flow_action --split test --episodes 21 --factory research.contact_learning:load_policy --checkpoint "$models/flow_action/best.pt" --max-steps 300 --device cpu --output-dir "$eval/flow_action_test" --videos 3
& $py research/evaluate_closed_loop.py --method visual_dp --split test --episodes 21 --factory research.visual_dp:load_policy --checkpoint "$models/visual_dp/best.pt" --max-steps 300 --device cpu --seed 0 --output-dir "$eval/visual_dp_test" --videos 3
```

视觉推理可使用 `--device cuda` 加速。其随机数生成器在整组固定 episode 顺序中连续使用；单独重跑中间一条可能不等于整组运行时的扩散噪声状态。相同 21 条重复运行是复验，不增加独立统计样本。

每个输出目录包含 `summary.json`、逐 episode 的 NPZ 数组与 JSON 诊断，前 3 条还生成 GIF/PNG。新的训练与闭环命令没有自动生成原实验所有专项审计证据，不能把“脚本退出成功”直接称为完整实验验收。

## 代码导航与数据约定

| 入口 | 作用 |
|---|---|
| [prepare_resources.py](../scripts/prepare_resources.py) | 固定资源下载、校验和上游源码恢复 |
| [run_stage1.py](../scripts/run_stage1.py) | 多个新进程完成第一阶段验收与日志 |
| [rebuild_derived.py](../scripts/rebuild_derived.py) | 在新目录重建全量数组并保留冻结划分/窗口清单 |
| [restore_artifact.py](../scripts/restore_artifact.py) | 校验 ZIP 后恢复文件，拒绝覆盖不同内容 |
| [reconstruct_contacts.py](../scripts/reconstruct_contacts.py) | 全量回放、编号点、物理子步接触重建 |
| [audit_dataset.py](../scripts/audit_dataset.py) | 数据审计及完整 episode 划分/窗口清单生成 |
| [validate_stage2_data.py](../research/validate_stage2_data.py) | 对全量历史数据只读复核 |
| [contact_learning.py](../research/contact_learning.py) | 接触与 Flow → action 网络、特征、训练、离线评估 |
| [visual_dp.py](../research/visual_dp.py) | 视觉缓存、轻量扩散策略训练与推理 |
| [controller.py](../research/controller.py) | 目标 flow、接触规则、接近路径、候选仿真预测 |
| [evaluate_closed_loop.py](../research/evaluate_closed_loop.py) | 四种策略的统一真实动作预算评测 |
| [validate_controller.py](../research/validate_controller.py) | 预测误差、候选分数作用与未来数据隔离检查 |
| [verify_experiments.py](../research/verify_experiments.py) | 对完整原实验工作区验收轨迹、代码/权重哈希和证据链 |

坐标 x 向右、y 向下，角度为弧度，长度为仿真世界单位。物体位姿位置是 body origin，不是质心；局部质心为 `(0,45)`。固定点遵循 `p_world = R(theta) p_local + t`，点编号不随物体旋转重排。

`states[t]` 是执行 `actions[t]` 前的状态，接触标签累计该动作的十个 100 Hz 物理子步。末动作后的仿真延续没有对应原视频帧，训练窗口排除该端点。零冲量几何接触不能当成有效推动；同组冲量在各点重复记录，求和必须先去重。详细字段见[数据说明](STAGE1_DATA_SCHEMA.md)。

## 墙体诊断与工程记录

控制器针对 T 形物体的路径膨胀半径为 18；预接触点和动作使用 `21..491` 粗边界。墙轴坐标为 5 与 506、墙半径 2，因此内面是 7 与 504；推杆半径 15 要求圆心至少在 `22..489`，还没有计入安全裕量。不能把现有路径函数称为完整墙体安全保证。

推杆有实体圆形碰撞几何，与动态 T 接触后可以推动它；但 kinematic 刚体不受碰撞反冲，四个目标探针已经实际验证它能追踪到墙内或穿墙位置。接触模型只包含绝对位置等几何特征，没有显式墙；规划 flow 不检查墙可行性。靠墙时只能尝试其他可达接触候选，尚无可靠的多阶段墙角脱困规划。

| 原测试方法 | 保存的状态数 | 侵墙状态数 | 最小净空 |
|---|---:|---:|---:|
| 规则接触 | 5,414 | 0 | 0.80562 |
| 主学习接触 | 5,342 | 0 | 2.83744 |
| Flow → action | 6,321 | 37，均在 episode 102 | -9.94583 |

净空为 `min(x-22, y-22, 489-x, 489-y)`，小于 -1e-6 判为侵入。**审计仅检查 10 Hz 保存状态，不证明各物理子步安全**；没有在此表审计视觉 DP 或额外种子。两例新增墙边/墙角诊断仅用于解释固定系统行为，不用于重新调参或汇入原成功率：左墙 261 步覆盖率 0.9570 成功，左上角 300 步覆盖率 0.9147 失败。

原实验还保留两项已披露的工程问题。一次离线/闭环日志重名覆盖了三组旧控制台记录；轨迹和模型未覆盖。修复后四组 21 条轨迹在新进程重跑，756 个数组字段中除覆盖率最高约 5.55e-16 舍入差外全部一致，200 个原文件哈希不变；新日志是补充证据，不冒称原日志。另一次安装留下 filelock/fsspec 重复元数据，训练后保留备份并恢复相同实际活跃版本；79 个源码文件保持一致，模型未重训。

原验收脚本需要完整原 `runs/`、源数据、模型和受保护文件；精选 `artifacts/` 无法代替这些依赖。发布证据中的个人路径已脱敏，因此部分历史文件字节及嵌入的原文件哈希关系不再相同；不要将直接重跑原审计器宣传为精简发布包的一键验收。发布清单用于核验发布文件；原验收 JSON 记录的是原工作区当时已经完成的检查。

## 尚未完成的研究

跨本体迁移、改变推杆半径、真实机械臂、图像到物体位姿估计、物理参数扰动、可行 flow 规划和严格墙体约束都仍是后续工作。原测试集已用于最终分析，后续改进应预先登记新的验证/测试用途；不能针对这 21 条失败调参后仍将它们称为未见测试集。

[历史数据准备 README](archive/README_data_preparation_20261022.md) 原样保留，便于理解数据流程；其中阶段状态只反映归档时点。当前项目状态以本页和根 README 为准。

