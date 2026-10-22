# 在另一台 Windows 电脑复现

本说明面向从 GitHub 克隆或下载的干净仓库。默认使用 Python 3.12 和 CPU；有兼容 NVIDIA GPU 时可选择 CUDA 12.6。训练与控制器的科学代码保持实验时的版本，新增入口只处理安装、资源恢复和独立回放。

## 1. 仓库、模型与证据包

仓库包含自写代码、依赖锁、固定 episode/window 划分、少量样本、汇总和报告。完整模型与逐步轨迹分别放在同一版本的 Release 附件中：

| 内容 | 恢复位置 | 用途 |
|---|---|---|
| `models.zip`，11 个 best 权重 | `runs/stages234_20261022/<模型名>/best.pt` | 加载已训练模型 |
| `evidence.zip`，完整实验记录与 206 条回放数组 | 包内原有相对路径 | 复核历史数值，或直接获得接触训练数据 |
| 固定版本的公开原始数据和外部源码 | `data/raw`、`downloads`、`external` | 由 `resource_manifest.json` 下载和逐文件校验 |
| 图像解码缓存 | `data/derived/visual_dp_cache` | 在本机生成，约 677 MiB，不放入 Git/Release |

在仓库的 Releases 页面下载相同版本的附件。私有仓库需要有访问权限并登录 GitHub；恢复脚本接收本地 ZIP，无需 `gh`。`artifacts/RELEASE_ASSETS.json` 保存 ZIP 的 SHA256；`artifacts/PUBLICATION_MANIFEST.json` 保存公开文件的校验和以及历史源文件到发布副本的映射。

发布副本移除了本机用户名和绝对路径等元数据，因此部分 JSON 和视觉 DP checkpoint 的文件 SHA 与历史源件不同。发布流程检查模型张量保持不变；不能把发布副本称为原始文件的逐字节副本。原实验代码、固定划分、数值结果及这一映射应一起解读。

## 2. 打开 PowerShell，安装独立环境

先安装兼容的 **64 位 Python 3.12**，然后打开 PowerShell，进入实际下载目录。以下目录只是示例，应替换为你自己的路径：

```powershell
Set-Location 'D:\projects\flow-contact-pusht'
py -3.12 --version
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/setup_windows.ps1 -Mode Cpu
$py = '.\.venv-cpu\Scripts\python.exe'
& $py -m pip check
```

`-ExecutionPolicy Bypass` 仅作用于这一 PowerShell 子进程，不修改系统执行策略。后续直接调用环境内 Python，无需激活脚本。没有 `py` 启动器时可以给出 Python 的实际位置：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/setup_windows.ps1 `
  -Mode Cpu -Python 'D:\Python312\python.exe'
```

训练环境使用另一目录：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/setup_windows.ps1 -Mode Cuda126
$py = '.\.venv-cuda126\Scripts\python.exe'
```

脚本安装 Torch 2.7.1 对应的官方 CPU 或 CUDA 12.6 wheel，其余版本来自 `requirements-training.lock.txt`，并检查版本、依赖冲突和实际前向/反向运算。CUDA 模式还会实际检测 GPU 运算；失败时保留日志和目录，不改驱动或安装 WSL。Wheel 来源与版本组合见 [PyTorch 官方历史版本页](https://pytorch.org/get-started/previous-versions/#v271)。这里没有使用 torchvision/torchaudio，所以不额外安装它们。

仅运行原始数据检查和简单推动时，可用 `-Mode DataOnly`，对应 `.venv`，省去 Torch。这个目录名也兼容旧的 `run_stage1.py` 环境验收；该旧入口硬编码检查 `.venv`，不接受 `.venv-cpu` 作为其环境证明。已存在且没有本脚本标记的环境不会被修改；可换一个 `-EnvironmentPath`。只检查已有环境而不安装：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/setup_windows.ps1 `
  -Mode Cpu -EnvironmentPath .venv-cpu -CheckOnly
```

不要直接将包含 `torch==2.7.1+cu126` 的完整锁文件用于 CPU 环境。安装脚本会分离 Torch wheel 选择，并将剩余精确依赖在一次事务中安装。脚本不会自动安装系统 Python、Git 或驱动。

## 3. 恢复公开数据、源码与模型

首先恢复固定数据和外部源码；已有文件校验相同则复用，内容不同则报错并保留。不要使用旧的 `acquire_resources.py` 覆盖入口。

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/restore_resources.ps1 -Python $py
```

这会恢复 Hugging Face 数据修订 `7628202a2180972f291ba1bc6723834921e72c19`、gym-pusht 提交 `e49cf07fd32ad6e69443af2921dd8343b0f9ad7d` 和 Diffusion Policy 提交 `5ba07ac6661db573af695b419a7947ecb704690f`。前者不是当前仓库随意更新后的接口，后者的完整训练栈也没有被安装；本项目按固定源码导入需要的环境与 U-Net。

将附件放在任意本地目录，再从发布清单提取校验和：

```powershell
$assets = (Get-Content artifacts/RELEASE_ASSETS.json -Raw | ConvertFrom-Json).assets
$modelHash = ($assets | Where-Object filename -eq 'models.zip').sha256
$evidenceHash = ($assets | Where-Object filename -eq 'evidence.zip').sha256
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/restore_resources.ps1 `
  -Python $py -Offline `
  -ModelsZip 'D:\Downloads\models.zip' -ModelsSha256 $modelHash `
  -EvidenceZip 'D:\Downloads\evidence.zip' -EvidenceSha256 $evidenceHash
```

只跑已训练模型时可以省去两个 `Evidence*` 参数。`-Offline` 表示公开资源必须已经存在；恢复附件本身始终只读取本地文件。公开资源准备失败但附件已经下载时，可以用 `-SkipPublicResources` 单独恢复附件。

脚本先校验 ZIP SHA 和全部成员路径、已有文件冲突，再写入缺少的文件；重复恢复相同文件会跳过。每次记录写入新的 `runs/restore_<标识>`。不同的旧结果或原始未脱敏 checkpoint 不会被覆盖；应在干净 checkout 恢复发布附件。

发布完整性检查与科学实验运行是两件事。附件下载目录内同时包含两个 ZIP 时，可执行：

```powershell
& $py scripts/verify_publication.py --release-dir 'D:\Downloads'
# 若是 git clone，且安装了 Git，还可检查所有跟踪文件与清单一致：
& $py scripts/verify_publication.py --release-dir 'D:\Downloads' --require-clean-git
```

## 4. 最小可运行检查和 CPU 闭环

每次选择新输出目录，避免覆盖已有实验。以下简单推动不加载模型；四种闭环检查分别加载实际的控制/策略实现：

```powershell
$run = 'runs/reproduction_' + (Get-Date -Format 'yyyyMMdd_HHmmss')
& $py scripts/simple_push.py --output-dir "$run/simple_push"
& $py research/evaluate_closed_loop.py --method rule --split test --episode-ids 12 `
  --max-steps 16 --videos 0 --device cpu --output-dir "$run/rule_smoke"
& $py research/evaluate_closed_loop.py --method learned_contact --split test --episode-ids 12 `
  --factory research.contact_learning:load_policy `
  --checkpoint runs/stages234_20261022/contact_main/best.pt `
  --max-steps 16 --videos 0 --device cpu --output-dir "$run/contact_smoke"
& $py research/evaluate_closed_loop.py --method flow_action --split test --episode-ids 12 `
  --factory research.contact_learning:load_policy `
  --checkpoint runs/stages234_20261022/flow_action/best.pt `
  --max-steps 16 --videos 0 --device cpu --output-dir "$run/action_smoke"
& $py research/evaluate_closed_loop.py --method visual_dp --split test --episode-ids 12 `
  --factory research.visual_dp:load_policy `
  --checkpoint runs/stages234_20261022/visual_dp/best.pt `
  --max-steps 16 --videos 0 --device cpu --output-dir "$run/dp_smoke"
```

这组 16 步检查验证程序连接与推理，不是完整成功率测试。完整测试将 `--episode-ids 12` 换成 `--episodes 21`，将步数改为 `--max-steps 300`，并为每种方法指定新的输出目录；`--videos 3` 可保存前三条可视化。CUDA 环境的视觉 DP 可改 `--device cuda`。输出包含逐步 NPZ、逐 episode JSON 与 `summary.json`。

闭环只需要固定 split、环境源码和模型权重，不需要读取测试示范的未来轨迹。视觉 DP 使用两帧图像与推杆位置；接触方法使用准确仿真位姿。重复同一 21 条场景不增加独立统计样本。DP 的随机数生成器在一次评测的各 episode 间连续使用；新进程单独执行后面的某条 episode，不保证复现整批评测到该条时的随机状态。

## 5. 从原始示范重新生成接触数据

若已恢复 `evidence.zip`，206 个 `data/derived/replayed_contacts/*.npz` 可直接供训练使用。若希望从头回放，不下载证据包也可以：

```powershell
$rebuild = 'runs/rebuild_' + (Get-Date -Format 'yyyyMMdd_HHmmss')
& $py scripts/rebuild_derived.py --output-dir $rebuild --install
```

这会在新目录完整回放 206 条、审计坐标/标签/原始动作、重新生成窗口，核对成员与 Git 中固定的 `splits.json`、`windows.json` 完全一致。只有全部通过后，才把数组复制到尚不存在的 `data/derived/replayed_contacts`。已有该目录时，省略 `--install` 可独立重建比较；不会更改历史数组或固定划分。

固定划分是 164/21/21 个 episode；接触窗口为 19130/2589/2281 个。新回放的运行时信息和报告 SHA 可能与历史报告不同，保存在新目录；脚本比较划分与窗口语义，不伪造历史 SHA。标签仍是固定仿真器重建的几何接触，不是独立的真实接触标注。

仅检查少量原始示范时，可以先运行：

```powershell
& $py scripts/replay_stage1.py --episodes 3 --output-dir "$run/three_replays"
```

## 6. 从头训练与独立评测

先完成资源恢复和 206 条回放恢复/重建。训练使用新的输出目录，不复用 Release 模型目录：

```powershell
$train = 'runs/retrain_' + (Get-Date -Format 'yyyyMMdd_HHmmss')
& $py research/run_experiments.py --phase contact --epochs 40 --output $train
& $py research/run_experiments.py --phase ablation --epochs 40 --output $train
& $py research/visual_dp.py prepare
& $py research/visual_dp.py train --epochs 60 --batch-size 64 --device cuda `
  --max-val-samples 0 --output "$train/visual_dp"
```

接触训练器自动选择可用 CUDA，否则使用 CPU；视觉 DP 的 CPU 训练需显式 `--device cpu`，速度会明显变慢。`prepare` 从固定 AV1 视频解码 25650 帧；视觉训练/验证/测试 anchor 数为 20443/2758/2449，与接触窗口数量不同。图像缓存需额外磁盘空间。

完成基于验证集的选择后，再运行测试，不用测试表现挑 checkpoint：

```powershell
& $py research/run_experiments.py --phase evaluate --output $train
& $py research/visual_dp.py evaluate --checkpoint "$train/visual_dp/best.pt" `
  --split test --max-samples 0 --device cuda --output "$train/visual_dp/test_metrics.json"
```

闭环命令与上一节一致，仅替换 checkpoint 为新的 `$train` 目录。32/128 点模型是离线采样消融，不能直接传入固定 64 点闭环控制器。不同机器、CPU/GPU和数值库的重复训练可能不逐位相同；应报告实际版本、验证选择、逐例结果和变化，不能只引用原报告成功率。

## 7. 已实际验证的范围与限制

本次发布准备创建了不含环境、外部源码、历史报告、派生数组和图像缓存的独立暂存目录；只复制代码、固定划分与经过清单校验的原始资源文件。实际检查见 `artifacts/portability/verification.json`。

已运行：固定资源核验及新解压、真实模型恢复、完整简单推动、四种方法在 CPU 上各 16 步闭环、完整 206 条重新回放和窗口审计、25650 帧图像缓存重建与视觉模型 CPU 前后向，以及重复恢复/错误校验和/路径穿越/冲突文件的保护检查。206 个新回放 NPZ 与历史数组文件 SHA 全部相同；固定 split/window 原字节保留。各正常流程在新进程运行；故意错误的恢复测试应失败且不写入目标文件。

另在第二个干净暂存目录恢复了发布候选 `models.zip` 与 `evidence.zip`，11 个发布 checkpoint 均实际加载到 CPU。接触主模型与脱敏后的视觉 DP 各运行 16 步，9 个轨迹数组与原始 checkpoint 对应的 CPU 运行全部精确一致。报告记录的是实际测试的候选包 SHA；最终下载包仍应对照该版本的发布清单独立校验。

这一轮复用了已经安装的 Python 3.12.14 依赖环境，**没有重新下载并安装一套 Torch，也没有在第二台电脑实测安装**。PowerShell 安装脚本已解析执行并实际检查现有锁版本、CPU/GPU 运算；网络可达性与全新 pip 安装仍需使用者所在设备验证。CPU 推理使用现有 CUDA wheel 的 CPU 路径，不能据此声称独立 CPU wheel 安装已验证。

历史 `research/verify_experiments.py` / `validate_stage2_data.py` 绑定原工作区完整证据和部分原 SHA；脱敏发布件不适合直接冒充这条旧审计链。发布完整性使用 `verify_publication.py`，新数据重建使用 `rebuild_derived.py`；它们的职责不同。旧 README 中的本机路径属于历史命令记录，应以本说明的相对路径命令为准。

本实现首先验证原生 Windows。资源清单保留 Windows 路径分隔符，部分绘图使用 `C:/Windows/Fonts/arial.ttf` 等字体路径；`--videos 0` 可跳过闭环视频字体依赖。未宣称 Linux/macOS、任意字体环境或跨本体真实机器人已验证。
