# 第三方来源与许可记录

本文件记录本项目实际使用的固定版本和本地核验到的许可声明。项目用于课程研究和私有 GitHub 协作；**没有为本项目自行编写的代码另行授予 MIT、Apache-2.0 或其他开源许可**。第三方许可只对应相应第三方内容，不自动成为整个仓库的许可。

## 固定资源

| 资源 | 版本 / 来源 | 实际核验的声明 |
|---|---|---|
| gym-pusht 仿真环境 | [huggingface/gym-pusht](https://github.com/huggingface/gym-pusht/tree/e49cf07fd32ad6e69443af2921dd8343b0f9ad7d)，提交 `e49cf07fd32ad6e69443af2921dd8343b0f9ad7d` | LICENSE 包含 Hugging Face 的 Apache License 2.0 全文，并保留衍生自 Diffusion Policy 的 MIT 声明 |
| Diffusion Policy 实现 | [real-stanford/diffusion_policy](https://github.com/real-stanford/diffusion_policy/tree/5ba07ac6661db573af695b419a7947ecb704690f)，提交 `5ba07ac6661db573af695b419a7947ecb704690f` | MIT License；版权声明为 2023 Columbia Artificial Intelligence and Robotics Lab |
| LeRobot PushT 数据集 | [固定版本数据卡](https://huggingface.co/datasets/lerobot/pusht/blob/7628202a2180972f291ba1bc6723834921e72c19/README.md)，revision `7628202a2180972f291ba1bc6723834921e72c19` | 数据卡元数据与正文均声明 `mit`；本记录依据该上游声明，不把它扩展为未公开材料的授权 |

已原样复制实际读取的许可与数据卡，便于离线核查：

- [gym-pusht LICENSE（含保留的 MIT 声明）](docs/third_party/gym-pusht-LICENSE.txt)
- [Diffusion Policy LICENSE](docs/third_party/diffusion-policy-LICENSE.txt)
- [LeRobot PushT 固定版本数据卡](docs/third_party/lerobot-pusht-dataset-card.md)

这些副本与下载后的原文件逐字节相同。校验值如下：

| 文件 | SHA256 |
|---|---|
| gym-pusht LICENSE | `87a2e7a1e8568a9bea2f2d5ab8ee1d716faebcc946ddbdaef8da9e655f8b0b9d` |
| Diffusion Policy LICENSE | `304e99a48271e10820f71a1b86ef395c06c100d7f68dd1d308790d1064fc4d41` |
| LeRobot PushT 数据卡 | `246e94b68f204bfcd05bb32e23631c3316b8de6152d7a85e693d4455540ef72f` |

下载地址、原始数据及源码归档的 SHA256 另见 [resource_manifest.json](resource_manifest.json)。本仓库的恢复脚本从固定版本上游取得资源，保留其原始许可文件。第三方内容应保留自身版权与许可声明；如今后修改或再分发第三方代码，应按对应许可处理，不以本项目文档替代上游许可全文。

## 使用关系与内容边界

`scripts/reconstruct_contacts.py` 使用 gym-pusht 的仿真环境；`research/visual_dp.py` 导入上述固定 Diffusion Policy 提交的 `ConditionalUnet1D`。本轮没有修改这两个下载的上游源码目录。视觉策略的编码器、网络规模、训练与采样流程由本项目适配，不声称完整复现原论文结果。

回放物体位姿、接触标签、模型权重、图表和实验报告来自本项目实际运行；它们的来源仍包括上述公开数据与仿真工具，不能称为原作者未公开的处理文件或独立接触真值。项目没有为这些新增内容另行作出开源许可声明。

原始数据与上游源码不重复纳入轻量 Git 仓库；根据固定资源清单获取。发布附件中的重建回放与实验材料保留来源信息。Python 依赖通过 [数据锁文件](requirements-data.lock.txt)和[训练锁文件](requirements-training.lock.txt)记录，由包管理器获取；本发布不捆绑虚拟环境，也没有逐项重新授权这些依赖。

用户提供的原始课程/研究 PDF **不包含在本次仓库或发布附件中**；本次整理不推断其再分发授权。项目自行生成的正式实验报告可以作为实验成果纳入发布；报告引用的第三方来源不因此获得新的权利。

## 学术引用

上游数据卡给出的引用如下，保留用于说明数据与方法背景：

```bibtex
@article{chi2024diffusionpolicy,
    author = {Cheng Chi and Zhenjia Xu and Siyuan Feng and Eric Cousineau and Yilun Du and Benjamin Burchfiel and Russ Tedrake and Shuran Song},
    title = {Diffusion Policy: Visuomotor Policy Learning via Action Diffusion},
    journal = {The International Journal of Robotics Research},
    year = {2024},
}
```

项目介绍见 [Diffusion Policy 官方页面](https://diffusion-policy.cs.columbia.edu/)。本课程项目的结果和局限见 [README](README.md)；第三方名称用于标明来源，不表示上游作者认可本项目结论。

