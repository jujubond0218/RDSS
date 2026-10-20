# RDSS 项目状态与下一轮优化

更新时间：2026-10-20。论文继续围绕RDSS推进。本文件是后续会话的状态入口；服务器、环境与操作边界见 [AGENTS.md](../AGENTS.md)。

## 当前阶段

首轮RDSS已实现，固定语义基线、Flowers噪声实验、PACS与NICO++连续域最小测试、原因诊断已完成。本轮将这些源码、描述构建配置、测试与文档统一归档到 `codex/rdss-first-round`，提交并推送后保留干净检查点。下一轮算法优化尚未开始。

论文方法主线：离线宽覆盖描述库；基于可信视觉证据进行相关性、判别性和冗余约束的语义选择；在线更新语义缓存，与模板CLIP和TDA视觉缓存融合，冻结编码器、无需梯度回传。论文围绕可靠判别性语义选择与连续测试流动态语义缓存更新展开。

## 已完成与实验结果

| 数据集/测试流 | TDA | TDA+全量 | 完整RDSS | 相关性+冗余诊断 |
| --- | ---: | ---: | ---: | ---: |
| Flowers-102，2463张/102类 | 71.4982% | 73.6500% | 73.2034% | 73.9748% |
| PACS最小域流，700张/7类 | 97.2857% | 97.5714% | 97.1429% | 97.1429% |
| NICO++公开子集，1120张/20类 | 92.5893% | 93.2143% | 92.5000% | 93.4821% |

全量与动态方法使用同一描述池和图像顺序，模板分支驱动共享的TDA视觉状态，语义预测后更新。完整首轮参数：k=8，每类近期证据上限32、衰减0.9、归一化熵<=0.2及模板/TDA一致门控、混淆top3、判别性权重1、冗余权重0.1、语义融合权重1。

Flowers每类100条宽覆盖描述已通过API生成并保存；PACS每类100条/共700条和NICO每类100条/共2000条使用当前对话编写的静态素材和脚本组合。Flowers每类追加25/50条属性错配后，全量分别73.8124%/73.6500%，完整RDSS分别73.6500%/73.2846%。

早先CuPL人工筛选第二版75.40%，对应官方全量74.18%，属于替换文本原型、各组分别更新视觉缓存的另一协议。完整记录见相应报告。

## 当前问题与诊断证据

1. **判别性代理分数有负面贡献。** 仅关闭判别性项、保留相关性与冗余后，Flowers比完整RDSS多对19张，NICO多对11张。当前Disc为Rel减混淆缓存最大相似度，依赖缓存的域、纯度与更新时效；默认实际目标为2Rel-Neg-0.1Red。
2. **新域证据进入锚点过慢。** NICO每阶段每类仅8张，而记忆上限32、衰减0.9；切换后前35张的新阶段证据平均权重仅4.61%～8.76%，最后35张仍只有43.78%～54.99%。PACS素描阶段也有选句切换滞后。
3. **少量证据下的原型替换过强。** 第一条可靠证据到来后，就从全量100句直接切到8句；需要保留类内覆盖并控制原型方向变化。Flowers追加50噪声使全量原型平均偏转4.472°，完整RDSS最终相对全量偏转12.160°。
4. **优先修复语义选择本身。** 完整RDSS语义原型单独分类也低于固定原型：Flowers72.6756%对73.1628%，PACS95.1429%对97.5714%，NICO91.5179%对93.0357%。融合权重不是唯一需要处理的环节。

详细依据、逐样本分析和组件结果见 [原因诊断](rdss-diagnosis.md)。上述机制解释与组件开关观测分开记录，完整原始结果保留。

## 后续优化顺序

先重设计判别性估计，检查混淆证据的时效、尺度与选句聚合后的分类间隔；随后区分快速当前样本查询与长期类记忆，解决域切换滞后；再控制证据不足时的语义原型替换幅度。保持已有可运行的端到端版本，在新功能分支逐项实现、配对比较。后续扩展数据规模和论文消融。

本轮工作限于检查点归档，不改正式RDSS算法，不启动新描述生成或安装环境。后续会话先确认Git状态与本文再继续。

## 文件与复现入口

实际仓库 `/root/autodl-tmp/project_R/RDSS`，本机同步镜像 `C:\Users\Administrator\Documents\ChatGPT\毕设\tmp\rdss-sync`；origin为 `https://github.com/jujubond0218/RDSS.git`。本地顶层目录保留对话资料和生成的暂存副本，已将源码与文档归档到实际仓库；其暂存目录使用本地Git排除规则管理，数据不删除、不另建提交历史。

| 内容 | 仓库入口/报告 |
| --- | --- |
| 固定语义基线 | `experiments/evaluate_fixed_semantic.py`；[结果](flowers102-fixed-semantic-results.md) |
| 完整首轮RDSS | `rdss.py`、`visual_cache.py`、`experiments/evaluate_rdss.py`；[说明](rdss-first-round.md) |
| 属性错配实验 | `experiments/evaluate_semantic_noise.py`；[说明](flowers102-semantic-noise.md) |
| PACS连续域 | `experiments/prepare_pacs_minimal.py`、`pacs_description_bank.py`、`evaluate_pacs_minimal.py`；[说明](pacs-minimal-shift.md) |
| NICO连续上下文 | `experiments/prepare_nico_minimal.py`、`nico_description_bank.py`、`evaluate_nico_minimal.py`；[说明](nico-minimal-shift.md) |
| 组件诊断 | `experiments/diagnostics/diagnose_rdss.py`；[原因报告](rdss-diagnosis.md) |
| 描述API生成 | `scripts/generate_description_bank.py`、`configs/description_api.example.json`；[说明](flowers102-broad100-generation.md) |

独立解释器 `/root/autodl-tmp/project_R/env/bin/python`，运行变量沿用AGENTS。PACS的Parquet准备使用已装PyArrow的base解释器 `/root/miniconda3/bin/python`；NICO准备无需该包。历史宽描述评估脚本依赖AGENTS记录的服务器实验资源，不需要为归档重新安装。

结果保留在 `/root/autodl-tmp/project_R/runs/` 与本地 `results/`，不提交原始特征、模型、数据或结果大文件。关键目录为：

- `flowers_rdss_first_20261020_201102`、`flowers_semantic_noise_20261020_202831`
- `pacs_minimal_shift_20261020_210546`、`nico_minimal_shift_20261020_220735`
- `rdss_diagnosis_20261020_223013`（包含domain_components/）

各目录保存参数、输入/源码哈希、预测、语义更新、原型等记录；根目录的AGENTS保留完整路径。API密钥仅留在被忽略的本地配置，示例配置密钥为空。

## 本轮归档验证

完整23项单元测试通过（含描述生成离线测试、固定语义与RDSS测试）。归档后的统一诊断入口完整重放Flowers/PACS/NICO，原始融合logits逐项一致，全部组件正确数与此前记录完全一致；重放目录 `/root/autodl-tmp/project_R/runs/rdss_diagnosis_20261020_232159`。验证日志为 `logs/rdss_checkpoint_tests_20261020.log` 和 `logs/rdss_checkpoint_replay_20261020.log`（均在project_R下）。提交前另做源码语法、密钥模式、忽略规则与git diff --check检查。
