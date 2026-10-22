# 第二至第四阶段：训练、闭环与对照实验

本轮是在第一阶段基础上实际运行的实验版。训练、验证和测试均有记录；下表来自固定21个测试初始状态，每条最多300步。二维仿真结果没有证明跨本体或真实机器人泛化。

| 方法 | 成功数 | 成功率 | 平均最终覆盖率 |
|---|---:|---:|---:|
| 规则接触+控制器 | 13/21 | 61.90% | 0.8712 |
| 学习接触+控制器（主种子0） | 15/21 | 71.43% | 0.9375 |
| Flow→action | 0/21 | 0.00% | 0.1172 |
| 轻量视觉Diffusion Policy | 0/21 | 0.00% | 0.0869 |

接触模型三个训练种子分别15/21、12/21、15/21，均值66.67%、样本标准差8.25个百分点。同21个场景重复评测，不是63个独立场景。主模型比规则多成功2条，不能宣称成功率显著提升。

主试验后补充的无flow闭环消融为6/21，相同控制器下主模型独有成功9例、无flow独有0例。单种子补充诊断支持flow作用，但不替换预先主模型或主要比较。证据在`contact_no_flow_test`。

接触方法使用准确物体位姿与独立候选物理预测；视觉方法仅使用图像和推杆位置。真实动作预算相同，信息和计算预算不同。DP采用官方U-Net源码，但编码器、网络规模、训练预算和采样设置经过适配，不是原论文完整复现。

## 先看结果

- [中文实验报告](output/pdf/PushT_第二至四阶段实验报告_2026-10-22.pdf)
- [四方法同一场景动画](runs/stages234_20261022/figures/comparison_episode_12.gif)
- [闭环对照图](runs/stages234_20261022/figures/closed_loop_comparison.png)
- [离线消融图](runs/stages234_20261022/figures/offline_ablation.png)
- [接触预测实例](runs/stages234_20261022/figures/contact_predictions.png)
- [完整统计](runs/stages234_20261022/statistical_analysis.json)
- [新进程验收](runs/stages234_20261022/final_acceptance.json)
- [第二阶段样本](runs/stages234_20261022/stage2_samples/samples.npz)与[样本来源](runs/stages234_20261022/stage2_samples/manifest.json)
- [第二至四阶段字段说明](docs/STAGES234_DATA_SCHEMA.md)
- [协议](docs/STAGES234_PROTOCOL.md)、[字段](docs/STAGE1_DATA_SCHEMA.md)、[视觉DP细节](research/VISUAL_DP.md)

## 打开PowerShell并使用训练环境

不需要修改执行策略或激活脚本，直接使用环境中的Python：

```powershell
Set-Location '.'
$py = '.\.venv-train\Scripts\python.exe'
& $py --version
& $py -m pip check
```

训练环境为Python3.12.14、torch2.7.1+cu126。已实际通过GPU前向/反向和CPU加载检查。第一阶段仍使用独立`.venv`，没有修改系统CUDA、驱动或WSL设置。

## 从保存的模型重跑闭环

每次使用新输出目录，程序拒绝覆盖既有结果。以下为同样的21条测试；重复运行用于核验，不增加科学统计样本量。

```powershell
$run = 'runs/retest_' + (Get-Date -Format 'yyyyMMdd_HHmmss')
& $py research/evaluate_closed_loop.py --method rule --split test --episodes 21 `
  --output-dir "$run/rule_test" --videos 3
& $py research/evaluate_closed_loop.py --method learned_contact --split test --episodes 21 `
  --factory research.contact_learning:load_policy `
  --checkpoint runs/stages234_20261022/contact_main/best.pt `
  --device cpu --output-dir "$run/contact_test" --videos 3
& $py research/evaluate_closed_loop.py --method flow_action --split test --episodes 21 `
  --factory research.contact_learning:load_policy `
  --checkpoint runs/stages234_20261022/flow_action/best.pt `
  --device cpu --output-dir "$run/flow_action_test" --videos 3
& $py research/evaluate_closed_loop.py --method visual_dp --split test --episodes 21 `
  --factory research.visual_dp:load_policy `
  --checkpoint runs/stages234_20261022/visual_dp/best.pt `
  --device cuda --output-dir "$run/visual_dp_test" --videos 3
```

无GPU时视觉推理可改`--device cpu`，速度会变慢。接触/规则控制器每个接近-短推-保持段后重新规划；DP每8个动作重新观察。

## 从头重新训练

数据集已经下载并重新核验。先按完整episode固定划分，再生成窗口。新的训练输出使用新目录；不要改变测试集挑选结果。

```powershell
$run = 'runs/train_' + (Get-Date -Format 'yyyyMMdd_HHmmss')
& $py research/run_experiments.py --phase contact --epochs 40 --output $run
& $py research/run_experiments.py --phase ablation --epochs 40 --output $run
& $py research/run_experiments.py --phase evaluate --output $run
& $py research/visual_dp.py prepare
& $py research/visual_dp.py train --epochs 60 --batch-size 64 --device cuda `
  --max-val-samples 0 --output "$run/visual_dp"
& $py research/visual_dp.py evaluate --checkpoint "$run/visual_dp/best.pt" `
  --split test --max-samples 0 --device cuda --output "$run/visual_dp/test_metrics.json"
```

本轮DP训练60轮，共19200次更新，实测28.0分钟。接触训练使用逐点共享MLP；消融包括32/64/128点、软硬标签、仅终点、GRU和移除flow。所有best权重仅由验证集选取。

迁移到另一台Windows机器时先安装兼容Python3.12，再创建`.venv-train`；从PyTorch官方CUDA12.6源安装torch2.7.1，再安装依赖锁。没有NVIDIA GPU时选择官方CPU wheel，不能照搬CUDA检测结论。

```powershell
py -3.12 -m venv .venv-train
& .venv-train/Scripts/python.exe -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu126
& .venv-train/Scripts/python.exe -m pip install -r requirements-training.lock.txt
& .venv-train/Scripts/python.exe -m pip check
```

精确本轮解析版本见`requirements-training.lock.txt`；Torch CUDA wheel需额外指定官方索引，不能仅依赖默认PyPI。CPU机器应先安装官方CPU wheel，再装`requirements-training.txt`，不要安装含cu126的本机CUDA锁文件。完整原始数据及固定源码的准备命令见`STAGE1_README.md`。

## 重新验收已保存结果

```powershell
$out = 'runs/stages234_20261022/acceptance_' + (Get-Date -Format 'yyyyMMdd_HHmmss') + '.json'
& $py research/verify_experiments.py --output $out
```

该验收在新进程检查逐例轨迹、独立重算覆盖率与在线flow、训练/验证/测试分组、模型权重哈希、验证集checkpoint选择、旧阶段文件哈希与日志修复证据。训练/测试实际命令和stdout/stderr保存在`runs/stages234_20261022/logs`；DP日志另见`visual_dp_train.log`及对应command记录。

一次开发缺陷使离线和闭环日志重名，三组旧闭环控制台日志被覆盖；轨迹、模型和指标没有覆盖。已修复唯一前缀+拒绝覆盖，并在`fresh_verification`新进程复跑四组、逐数组对比。重跑是补充证据，不伪装原始日志，也不计作独立测试。

安装过程中两个辅助包filelock/fsspec留下重复版本元数据。训练时实际使用版本的79个源码文件均匹配对应wheel记录；训练结束后保留备份并重新安装同版本，检查重复元数据为零。证据为`runtime_metadata_before_repair.json`与`runtime_metadata_after_repair.json`。未在训练中更换算法依赖，也未改动第一阶段环境。

## 数据与推理约定

- `states[t]`为动作前状态；`actions[t]`是绝对推杆中心目标；接触累计该动作的10个物理子步。
- 模型监督为未来8周期接触区域并集；局部高斯软标签sigma5，无接触全零。加权BCE输出称分数，不是校准成功概率。
- 离线flow可以来自示范未来；真实闭环只允许当前状态和目标规划flow。评测脚本不读取测试未来轨迹。
- coverage是物体与目标相交面积/目标面积；success按>=0.95。图像IoU、定位命中率和任务成功率是三种不同指标。
- 主闭环固定64点soft-sequence seed0。32/128只是离线消融；不能直接把其checkpoint传给固定64点控制器。

## 尚未证明的内容

跨本体和真实世界迁移、图像到物体位姿估计、最优接触监督、完整论文Diffusion Policy复现均未完成。离线定位改善不等于闭环稳定改善；三训练种子结果有波动。下一步应围绕规划flow分布、控制器独立贡献和推杆几何变化设计明确实验，而不是把负结果改写成成功。
