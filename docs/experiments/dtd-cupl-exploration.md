# DTD CuPL 早期探索记录

实验日期：2026-10-20。本文件只归档早期DTD对照事实与来源；当前目标和执行顺序统一见 [研究计划](../semantic-adaptation-experiment-plan.md)，不将当时的DTD下一步方案当作现行任务。

早期讨论曾把“不适配目标域的语义”称作冗余；现统一区分：RDSS的冗余抑制针对重复或高度相似的语义证据，目标域不适配属于相关性和适配性问题。均匀重复、偏向性重复、属性错配是三类不同操作，不能互相代替证据。

## 已有实验事实

以下为同一 DTD test 划分、CLIP ViT-B/16、1692 张图片、测试流种子 1 的探索结果。各组共用同一视觉特征和顺序，视觉缓存从空开始独立更新。

| 条件 | Top-1 准确率 | 当前能说明什么 |
| --- | ---: | --- |
| 原始 TDA 类名模板 | 44.80% | 已跑通且可重现的本机基线 |
| TDA + 官方旧 CuPL 全部 60 条/类 | 54.08% | 这套静态描述明显有效 |
| 官方旧池去除规范化文本完全重复项 | 54.26% | 本次只多正确 3 张，不能等同于目标域适配证据 |
| 官方旧池每类随机 10 条，固定种子 0/1/2 | 53.37% / 51.65% / 51.89%，均值 52.30% | 描述子集的构成会影响结果；这是三组描述抽样，不是三次测试流重复 |
| TDA + 本次新生成描述 10 条/类 | 50.53% | 高于模板，但低于旧池各个等数量子集 |
| 新描述全部均匀复制一次 | 50.53% | 均匀复制不改变均值文本原型，实际原型差为 0 |
| 新池首条描述额外复制 10 次 | 44.62% | 人为改变某条描述的相对权重可以损害本次表现 |
| 新池上的简单历史视觉原型动态 Top-3 | 50.12% | 低于新版静态 0.41 个百分点，尚无动态优势 |

官方旧池共 2820 条，按类别内规范化文本去重后合计 2745 条；新池 470 条全部唯一。旧、新池按类平均的文本嵌入两两余弦相似度分别约为 0.8895、0.8529。新池重复较少但准确率较低，说明重复程度不是效果的唯一决定因素。

这些结果**尚未直接验证主要假设**。此前动态 Top-3 只在新版 10 条候选池上运行，没有完成官方旧池上“全部 60 条、随机 K 条、适配 K 条”的配对比较；均匀复制、偏向性复制和完全去重实验也没有直接操作“与目标域不匹配的属性”。不得把这些结果写成已经证明目标域语义筛选有效。

新描述由用户要求的低成本 `gpt-6-luna` 子智能体生成。它与旧池的生成模型、提示设计和文本内容均不同；即使控制描述数量，也不能单独归因于模型年代或模型能力。生成记录和评估前文本清理过程均保留，不能将它直接称为“高质量新版描述”。

## 文件与实验位置

服务器代码：`/root/autodl-tmp/project_R/RDSS`；独立解释器：`/root/autodl-tmp/project_R/env/bin/python`；DTD 数据：`/root/autodl-tmp/project_R/data/dtd`。复用这些资源，不需要重新安装环境或下载数据。

已完成的对照程序位于 `/root/autodl-tmp/project_R/experiments/cupl_dtd/compare_prompts.py`。官方旧描述为同目录 `prompts/official_full.json`；新版描述为 `prompts/modern_luna.json`。官方旧池固定来源为 [CuPL 仓库的 texture_prompts_full.json](https://github.com/sarahpratt/CuPL/blob/719619c829792ff8f37298266e941b7f077c7f5c/all_prompts/full_prompts/texture_prompts_full.json)。

完整探索结果在 `/root/autodl-tmp/project_R/runs/dtd_cupl_compare_20261020_113018`；本地可查看[实验报告](C:/Users/Administrator/Documents/ChatGPT/毕设/results/dtd_cupl_compare_20261020_113018/report.md)、[结果表](C:/Users/Administrator/Documents/ChatGPT/毕设/results/dtd_cupl_compare_20261020_113018/summary.csv)和[官方描述副本](C:/Users/Administrator/Documents/ChatGPT/毕设/results/dtd_cupl_compare_20261020_113018/prompts/official_full.json)。原运行的 10 组条件、预测、描述抽样索引、生成元数据与文件哈希均保留。

CuPL 方法与生成提示的背景依据为[原论文](https://arxiv.org/html/2209.03320v3)。论文发表于 ICCV 2023，预印本首次发布于 2022 年；不能将其主要 ViT-L/14 结果直接当作当前 ViT-B/16 的目标数值。
