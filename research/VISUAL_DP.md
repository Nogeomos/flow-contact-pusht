# 视觉 Diffusion Policy 适配说明

这是从头训练的**轻量视觉 DP 基线**。它导入官方提交 `5ba07ac6661db573af695b419a7947ecb704690f` 的 `ConditionalUnet1D`，保持卷积去噪、时间步嵌入和 FiLM 条件机制；不修改上游文件，也不声称复现论文的完整结果。

## 模型看到什么

在线仅输入最近两帧 RGB 图像和这两帧中的推杆实际中心。图像是 96×96，包含环境固定绘制的绿色目标。模型没有物体真值位姿、未来图像、示范未来运动、接触标签或显式 flow。它输出绝对推杆目标坐标，不输出物体运动。

当前时间记为 t。输入 `[图像 t−1, 图像 t]` 和对应推杆位置；t=0 时重复首帧。扩散模型预测 16 个动作，索引对应 `[t−1, t, …, t+14]`，实际执行其中 `[1:9]` 的八个动作，也就是 t 到 t+7。随后用新的实际观测再次生成动作。终端补齐部分不参与训练损失，窗口不跨示范。

## 与完整官方设置的差别

| 项目 | 本实现 | 已下载官方 PushT 配置 |
|---|---|---|
| 图像编码器 | 四层卷积和全连接，随机初始化 | Robomimic 图像编码器 |
| U-Net 通道 | 64/128/256 | 512/1024/2048 |
| 扩散训练步数 | 100，cosine，加噪目标 epsilon | 100，cosine，epsilon |
| 推理 | DDIM 20 步，eta=0 | DDPM 100 步 |
| 观测/预测/执行长度 | 2/16/8 | 2/16/8 |
| 划分 | 固定完整示范 164/21/21 | 至多90条训练，独立配置划分 |
| 训练预算 | 由实际训练记录给出，默认60轮 | 3050轮 |
| 归一化 | 已知世界边界 xy/256−1 | 数据统计归一化 |

本实现采用 AdamW、梯度裁剪、EMA、GPU 混合精度。checkpoint 按固定验证噪声的 EMA 去噪均方误差选择；测试集性能不参与选择。`best.pt` 包含模型权重、配置、训练示范 ID、训练窗口索引、实现和数据划分的哈希。闭环实验的真实效果需要另行测量，去噪误差下降不等同于任务成功。

## 运行

从项目根目录启动 PowerShell：

```powershell
.venv-train\Scripts\python.exe research\visual_dp.py prepare
.venv-train\Scripts\python.exe research\visual_dp.py smoke --device cpu
.venv-train\Scripts\python.exe research\visual_dp.py train --device cuda --epochs 60 --batch-size 64 --output runs\stages234_20261022\visual_dp
.venv-train\Scripts\python.exe research\visual_dp.py evaluate --device cuda --split test --checkpoint runs\stages234_20261022\visual_dp\best.pt --output runs\stages234_20261022\visual_dp\test_offline.json
```

`--limit-steps` 限制每轮训练批次数，仅用于快速检查。正式训练不要加入该选项。已有训练目录拒绝覆盖，请新建目录。`prepare` 在磁盘保存 uint8 内存映射图像，逐帧检查时间戳并验证原始源文件哈希；再次运行会验证已有缓存而不会重复写入。

供闭环控制程序调用：

```python
from research.visual_dp import load_policy
policy = load_policy(".../best.pt", device="cuda", seed=0)
targets = policy.predict(images, agent_positions)
# images: uint8 RGB, (2,96,96,3)
# agent_positions: 实际推杆xy, (2,2)
# targets: (8,2)，0..512世界单位的绝对坐标
```

应使用 `env._render()` 的96像素观测，不能使用带动作十字标记的 `env.render()` 可视化代替。离线评测返回与示范动作之间的欧氏距离，并按完整示范进行 bootstrap；这些指标不代表唯一正确动作，也不代表闭环成功率。与真值物体位姿/几何输入的接触法相比，视觉 DP 的输入信息不同，结果应分别说明。

依赖由项目独立训练环境提供：PyTorch 2.7.1、einops 0.8.1、diffusers 0.35.1，以及第一阶段的数据读取依赖。是否成功训练、使用多少轮、用时和测试结果，以 `training_result.json` 和评测报告的实际输出为准。

## 本轮已运行结果（2026-10-22）

训练已真实执行60轮、19,200次更新，5,796,130个参数，训练循环用时1682.48秒。全验证集最优是第10轮，EMA去噪MSE为0.043241；第60轮验证MSE升至0.131161，因此最终使用第10轮 `best.pt`。训练曲线显示过拟合，不能根据训练损失继续下降判断泛化改善。

模型在闭环验证和测试前固定，测试后没有改权重、采样步骤或动作执行长度。实际结果：

| 项目 | 结果 |
|---|---:|
| 验证闭环成功 | 0 / 21 |
| 测试闭环成功 | 0 / 21，Wilson 95%区间 0–15.46% |
| 测试平均最大覆盖率 | 0.308277 |
| 测试平均最终覆盖率 | 0.086862 |
| 完整2449个测试窗口的首动作平均欧氏误差 | 28.81世界单位 |
| 八步动作平均欧氏误差 | 46.57世界单位 |
| 保持当前推杆位置的八步动作误差 | 54.87世界单位 |

离线动作误差有所降低，但没有转化为本轮闭环成功。本结果仅适用于这个轻量实现、数据划分和训练预算。模型只有一个训练随机种子，未验证跨机器人本体或真实机器人迁移。

在线采样使用seed=0的同一个随机数生成器，按固定示范ID顺序连续使用，各episode不重新设seed。离线采样单独使用seed=20261022。新进程重新运行第一个测试episode 12后，全部9个轨迹数组逐元素完全一致，NPZ的SHA256也相同；这只是复验，不增加独立测试样本数。

证据位于 `runs/stages234_20261022/`：

- `visual_dp/training_result.json`、`history.json`、`frozen_evaluation.json`：训练和测试前冻结记录。
- `visual_dp/test_metrics.json`：完整2449窗口的离线指标及逐示范、逐窗口明细。
- `visual_dp_test/summary.json`：21条闭环测试，另有逐条轨迹、前三条GIF和PNG。
- `visual_dp_test_analysis/all_test_failures.png`：全部失败的初始、过程、最佳和最终状态。
- `fresh_verification/visual_dp_comparison.json`：新进程数值复验。
- `visual_dp_commands.json`：真实命令、退出状态、日志和结果文件索引。

训练结束后曾修复独立环境中filelock/fsspec重复安装元数据；修复保持原活跃版本及79个源码文件一致，之后才启动闭环验证与测试，训练模型和推理代码均未改变。
