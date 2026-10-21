# RDSS 文献对照与借鉴边界

核验日期：2026-10-21。本文维护文献信息与本项目的关系，不作为已复现清单；当前实施顺序见 [研究计划](semantic-adaptation-experiment-plan.md)。发表状态以链接的一手来源为准，预印本不写成正式录用，论文作者报告的优势不等于本机验证结果。

## 优先参考的2026工作

| 论文与出处 | 一手依据与阅读范围 | 可借鉴内容及边界 |
| --- | --- | --- |
| ARTTA — Anchored Reliability: Decoupling Estimation from Adaptation for Noisy Test-Time CLIP；ICIP 2026，DOI 10.1109/ICIP61757.2026.11630209 | [IEEE出版页](https://doi.org/10.1109/ICIP61757.2026.11630209)；已核验出版元数据及摘要，全文受访问限制 | 冻结锚点估计类别间隔可靠性，与在线适应分开；其本身更新LayerNorm，RDSS只借鉴可靠性原则，不引入该训练流程 |
| RADC — Risk-Aware Dual Caching for Vision-Language Test-Time Adaptation；2026-10-03预印本 | [arXiv:2610.06932](https://arxiv.org/abs/2610.06932)、[方法正文](https://arxiv.org/html/2610.06932v1)；核对摘要与风险接收方法节，未确认正式录用 | 用类别分离与多视图特征方差联合管理缓存；其语义前景缓存是视觉空间证据，不是本项目的文本语义缓存 |
| A²Memory — Learning to Memorize with Attributive and Associative Memory for Online Test-Time Adaptation of Vision-Language Models；ICML 2026 | [PMLR正式页面](https://proceedings.mlr.press/v306/zhang26gj.html)；已核验出版元数据和摘要，未逐式复现 | 属性中心表示、软分配与有限记忆保留；关联记忆含梯度优化，不能整套标为本项目免训练方法 |
| TaTa — Training-Free Test-Time Adaptation with Brownian Distance Covariance in Vision-Language Models；ICASSP 2026 | [arXiv:2601.23253](https://arxiv.org/abs/2601.23253)，作者页面标注Accepted in ICASSP 2026；DOI 10.1109/ICASSP55912.2026.11462803；已核对摘要/接收说明，未复现推导 | 布朗距离协方差、属性增强描述与动态聚类；BDC是统计工具备选，不能直接替换余弦后宣称有效；需先检查样本单位、在线因果性与成本 |
| MARC — Retrieval Geometry Shapes Cache-Based CLIP Adaptation；2026-09-20预印本，09-25修订 | [arXiv:2609.23409](https://arxiv.org/abs/2609.23409)、[正文](https://arxiv.org/html/2609.23409v2)；页面标注Under Review at ICLR，不视为录用 | 固定记忆比较检索空间，采用冻结DINOv2-B检索；属于需额外编码器的参考，暂不纳入最小RDSS主线 |

优先深入阅读ARTTA/RADC的可靠性定义和实验协议，再核对A²Memory的污染/保留分析；TaTa和MARC用于统计度量与检索对照的备选。以上均未在本项目复现，引用不表示已采用整套算法。

## 基础方法与最接近的工作

| 论文 | 与RDSS的关系 | 不能当作本项目原创的部分 |
| --- | --- | --- |
| [Efficient Test-Time Adaptation of Vision-Language Models（TDA），CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/papers/Karmanov_Efficient_Test-Time_Adaptation_of_Vision-Language_Models_CVPR_2024_paper.pdf) | 已有代码基础，正负视觉缓存与免反传适应 | 视觉缓存和正负伪标签机制 |
| [What Does a Platypus Look Like?（CuPL），ICCV 2023](https://arxiv.org/abs/2209.03320) | 类别描述库与全量融合依据；预印本始于2022年 | 用LLM生成描述本身 |
| [Does VLM Classification Benefit from LLM Description Semantics?（DisCLIP），AAAI 2025](https://ojs.aaai.org/index.php/AAAI/article/download/32638/34793) | 最接近的局部判别性描述选择；使用类别参考图像和classname-free描述 | 泛称“判别性选句”；需突出在线伪标签证据及可靠更新的具体区别 |
| [Summarization: Using MMR for Diversity-Based Reranking，1998](https://aclanthology.org/X98-1025/) | 相关性与冗余的成熟贪心平衡依据 | MMR式去冗余公式本身；不能未经证明声称当前目标具有其理论保证 |
| [COSMIC，CVPR 2025](https://arxiv.org/abs/2503.23388) | 跨模态语义缓存、类别关系图与额外DINOv2特征 | 首次语义缓存、首次跨模态缓存或首次双缓存协同 |

[RoTTA，CVPR 2023](https://arxiv.org/abs/2303.13899)提供类别平衡、时效与不确定性管理记忆的依据，其自身包含训练。[DynaPrompt，ICLR 2025](https://arxiv.org/abs/2501.16404)研究提示缓冲区的动态选择与更新，其对象是可学习提示而非自然语言描述子集。二者用于动态适应背景，不能把训练式提示优化改名成免训练RDSS。

## 对本项目的具体启发：尚未实现

冻结模板可作为不随语义适应漂移的可靠性参照；当前已有模板驱动门控只是部分符合这一原则，不是新增创新或正确性保证。拟在其上比较间隔、熵和多视图风险，保持R/D/S选择及语义缓存主线。

可研究“较大平均类别间隔、较小视图波动”是否对应更可信证据。多视图风险需要真实增强和重新编码；单视图或同类图片方差不能冒充同一图片的多视图不确定性。简化成何种权重、如何归一化、是否增加覆盖损失均需开发协议验证，不将自拟公式写成原论文公式。

伪标签软分配、BDC和额外检索空间仅为可选工具；使用前要确认不会利用未来样本、引入梯度训练或超出计算预算。借鉴原则不等于必须实现所有模块，也不能用更多数学名称替代实验依据。

## 引用与贡献表述

拟贡献集中为“在线可信视觉证据驱动的RDSS联合选择”和“可靠性/时效控制的动态语义缓存”，具体新定义及差异需由实现与消融支撑。论文不得声称首次生成描述、首次语义缓存、首次MMR或已证明全部语义冗余有害。对仅阅读摘要的工作先作相关工作定位，采用具体数学构造前应补读全文、附录及可用官方代码。
