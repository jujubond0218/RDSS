# 毕业设计项目与服务器环境

## 新会话先读

本地目录 `C:\Users\Administrator\Documents\ChatGPT\毕设` 用于组织对话，不是实验代码目录。实际代码和已安装环境均在 AutoDL 服务器。不要因为本地没有代码、当前 shell 显示 `(base)`，就重新克隆或安装。

服务器连接：`ssh -p 46939 root@region-9.autodl.pro`。密码不写入文件；已有会话凭据可继续使用，否则向用户获取。服务器实例、端口或 GPU 可能变化，操作前核实连接目标。

服务器仓库也保存本文件：`/root/autodl-tmp/project_R/RDSS/AGENTS.md`。环境说明变更时同步两处。

## 已有资源：优先复用

| 用途 | 路径或配置 |
| --- | --- |
| 项目根目录 | `/root/autodl-tmp/project_R` |
| 代码仓库 | `/root/autodl-tmp/project_R/RDSS` |
| Git origin | `https://github.com/jujubond0218/RDSS.git` |
| 基础方法 | TDA，官方上游 `https://github.com/kdiAAA/TDA` |
| 独立 Conda 环境 | `/root/autodl-tmp/project_R/env` |
| Python 解释器 | `/root/autodl-tmp/project_R/env/bin/python` |
| Conda 初始化脚本 | `/root/miniconda3/etc/profile.d/conda.sh` |
| 数据根目录 | `/root/autodl-tmp/project_R/data` |
| 项目运行 HOME | `/root/autodl-tmp/project_R/runtime-home` |
| ViT-B/16 权重 | `/root/autodl-tmp/project_R/runtime-home/.cache/clip/ViT-B-16.pt` |
| 日志 | `/root/autodl-tmp/project_R/logs` |
| Conda 包缓存 | `/root/autodl-tmp/project_R/cache/conda-pkgs` |
| pip 缓存 | `/root/autodl-tmp/project_R/cache/pip` |
| 临时目录 | `/root/autodl-tmp/project_R/tmp` |

2026-10-20 已安装并验证：Python 3.10.22、PyTorch 2.5.1+cu124、torchvision 0.20.1+cu124、NumPy 1.26.4、Pillow 10.4.0、WandB 0.15.12。当时 GPU 为 A100 PCIe 40GB，CUDA runtime 为 12.4。

安装的是源码实际使用的 baseline 运行依赖，未安装未使用的开发工具及 wilds 等额外依赖。直接依赖记录在 `/root/autodl-tmp/project_R/requirements-runtime.txt`；安装后完整 Python 包版本在 `/root/autodl-tmp/project_R/requirements-lock.txt`；Conda 基础包记录在 `/root/autodl-tmp/project_R/environment-conda-explicit.txt`。这些文件用于查询和有明确需求时重建，不是每个会话的安装步骤。

先检查现有解释器和 `python -m pip check`。已有环境可用时直接使用；出现错误时先诊断具体原因，不擅自删除、覆盖、升级或重建环境。不要向 `base`、其他项目环境或系统 Python 安装依赖，不修改系统 CUDA、驱动、全局 Conda/pip 配置或 shell 启动文件。

## 已准备的数据与模型

Caltech101 已按当前项目加载器要求放置：

```text
/root/autodl-tmp/project_R/data/caltech-101/
├── 101_ObjectCategories/
└── split_zhou_Caltech101.json
```

图片来自 Caltech 官方：https://data.caltech.edu/records/mzrjq-6wc02 。原始 ZIP 保存在 `project_R/downloads/caltech-101.zip`，MD5 为 `3138e1922a9193bfa496528edbbc45d0`，已校验。

划分文件来自公开研究代码的副本：https://github.com/linzhiqiu/cross_modal_adaptation/blob/main/splits/split_zhou_Caltech101.json 。SHA256 为 `52ce83205ff8ff8a1a18119aad6376120ae35c36e783a984bc560cbf10e73fa5`。已验证所有划分中的图片路径存在；测试集 2465 张、100 类，测试图片均通过 PIL 读取校验。不要自行重新划分或改变类别编号。

用户选择的模型是 **CLIP ViT-B/16**，不是 RN50。权重来自原项目指定的 OpenAI 下载地址，SHA256 为 `5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f`，已校验。不需要重复下载。

2026-10-20 另已准备 **DTD r1.0.1**，用于准确率较低且规模较小的后续实验。服务器目录为 `/root/autodl-tmp/project_R/data/dtd`，其中 `images/`、`labels/`、`imdb/` 来自官方包，`split_zhou_DescribableTextures.json` 是项目要求的 Zhou 划分。总计 5640 张、47 类，train/val/test 分别为 2820/1128/1692 张；解压后约 617MiB。原项目 `build_dataset('dtd', '/root/autodl-tmp/project_R/data')` 已成功读取 1692 张测试图、47 类；所有图片通过 PIL 结构检查，所有划分路径存在，三个划分无交集。2026-10-20 已完成 DTD 全量测试：TDA + ViT-B/16 的 Top-1 为 44.80%（1692 张），论文值为 47.40%，低 2.60 个百分点，尚未确定差异原因。测试循环约 21 秒，含启动约 33.04 秒。结果目录为 `/root/autodl-tmp/project_R/runs/dtd_vitb16_20261020_110941`，保存完整日志、配置、指标和资源采样；使用原 DTD 配置，无需补字段，未改动算法或环境。

DTD 原始压缩包保存在 `/root/autodl-tmp/project_R/downloads/dtd-r1.0.1.tar.gz`，下载自 `https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz`，MD5 为 `fff73e5086ae6bdbea199a49dfb8a4c1`，与 torchvision 使用的校验值一致。划分副本来自 `https://raw.githubusercontent.com/linzhiqiu/cross_modal_adaptation/main/splits/split_zhou_DescribableTextures.json`，SHA256 为 `f26ebeccd2b58e68d70f07a0dc39fe49bf7b69024bac34396b954dfd879699c38`。准备记录保存于数据目录的 `preparation.json`。不要改用 torchvision 默认划分替代这个项目划分，不需要重新下载或安装环境。

## 后续数据集：Oxford Flowers-102

2026-10-20 用户要求再换一个数据集，已另行准备 Oxford Flowers-102，保留原有 DTD 和 Caltech101。服务器目录为 `/root/autodl-tmp/project_R/data/oxford_flowers`，包含 `jpg/`、`imagelabels.mat`、`split_zhou_OxfordFlowers.json` 和 `preparation.json`。共 8189 张图片、102 类；项目使用的 Zhou 划分为 train/val/test = 4093/1633/2463，图片约 331 MiB。全部图片通过 PIL 结构检查，所有划分路径存在、无划分交集，原项目 `build_dataset('oxford_flowers', '/root/autodl-tmp/project_R/data')` 已成功读取 2463 张测试图、102 类。

原始包 `/root/autodl-tmp/project_R/downloads/102flowers.tgz` 来自 `https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz`，MD5 `52808999861908f626f3c1f4e79d11fa` 与 torchvision 记录一致。划分来自 `https://raw.githubusercontent.com/linzhiqiu/cross_modal_adaptation/main/splits/split_zhou_OxfordFlowers.json`，SHA256 `2f50c42fc1a17ca13c271cb4269b1347bdb43d2be672e34f74a4e4cbfbcca794`。不要换成 torchvision 默认划分。

使用现有环境与 ViT-B/16 即可，参数为 `--datasets oxford_flowers --config /root/autodl-tmp/project_R/RDSS/configs`，其余运行参数沿用已有命令。原 `configs/oxford_flowers.yaml` 的负缓存阈值字段完整。TDA 论文 Table 3 的 ViT-B/16 Flower102 为 71.42%，这是论文参考值，注意区分指标与划分。

2026-10-20 后续已完成 Flowers-102 全量配对实验，结果目录 `/root/autodl-tmp/project_R/runs/flowers_cupl_manual_20261020_120643`，本地 `results/` 中有同名副本。程序为 `/root/autodl-tmp/project_R/experiments/cupl_flowers/flowers_experiment.py`，输入在其 `prompts/`。复用此前 DTD 程序的原 TDA 缓存重放函数，不改原仓库代码。模板 TDA 71.50%（1761/2463），官方 CuPL 全量每类20条为74.18%（1827），助手根据每类两张验证图和原文本选出200条、每类1～2条后为74.10%（1825）。对应类别平均为71.01%/73.46%/73.23%。5份匹配每类条数的随机子集均值71.79%，范围70.16%～73.28%。共同测试流种子1，各组独立空缓存。筛选于所有组测试前冻结，未据测试结果改写。人工相对全量纠正119张、损坏121张，净少2张（-0.0812个百分点），没有证明筛选胜过全量。

筛选协议、逐类理由和原句、冻结哈希、验证接触图、逐样本预测均保存在结果目录。类别对齐有 `globe-flower -> globe flower`、`pink-yellow dahlia -> pink and yellow dahlia`、`ball moss -> air plant`，两种CuPL条件映射一致；最后一组名称并非严格同义，watercress也存在名称与验证图含义错位。人工筛选使用验证图片，不能称为完全零样本或在线无监督算法。当前未生成Flowers新版描述、未实现新的动态选择器；不要根据已观察的测试错误反复选句后当成独立验证。详细结果见本地 `docs/flowers102-cupl-manual-results.md`。

2026-10-20 用户随后要求重新人工筛选。第二版已另存于 `/root/autodl-tmp/project_R/experiments/cupl_flowers/manual_v2`，本地 `results/flowers_manual_v2_inputs` 有副本；记录见 `docs/flowers102-manual-v2-selection.md`。从相同官方2040条中保留712条，平均每类6.98条，79类保留7～8条；逐类记录原句索引、混淆类别和保留依据，不补写或改写原句。重点兼顾验证图外观、颜色变化、结构与相似类别区别。第二版是在第一版测试诊断已知后制作，在同一测试集复测应记录为探索性复测，不能声称全新独立盲测。旧脚本限制最多3条，不能直接覆盖旧输入运行712条版本。第一版与第二版输入均保留，未改算法或安装环境。

第二版现已完成全量测试，程序为上述目录内 `evaluate_v2.py`，复用第一版保存的2463张完整视觉流、顺序和原配置，各组独立空缓存。结果在 `/root/autodl-tmp/project_R/runs/flowers_cupl_manual_v2_20261020_150338`，本地results有同名副本；记录见 `docs/flowers102-manual-v2-results.md`。第二版Top-1 **75.40%（1857/2463）**，官方全量复跑74.18%（1827），第一版复跑74.10%（1825），两项控制均精确复现。第二版相对全量高1.2180个百分点，多答对30张（纠正80、新增错误50）；类别平均74.47%，全量73.46%。5份匹配第二版每类条数的随机子集均值73.40%，范围72.23%～73.89%，共同流种子1。输入712条和索引哈希保持冻结，未依据第二版结果继续改写。全部8份逐样本答对数及视觉流哈希已独立核对。75.40%是第二版结果，74.10%属于第一版。此为离线人工筛选的探索性复测，不是自动动态算法结果。

## 自生成分层描述库脚本

2026-10-20用户已明确授权为Flowers-102全部102类生成每类100条：概述20、细节60、显式对比20，总计10200条。当前提示版本 `layered-visual-bank-v4-broad-100`，输入 `configs/flowers102.classes.json`（原标签、空hint），不沿用此前植物知识澄清，不读取图片、标签、预测或实测混淆矩阵。比较对象由LLM提出。宽覆盖颜色、品种、形态、阶段、视角、部位和场景，允许语义重叠与对具体图片不适用的候选。

用户明确要求“不要审查”：不做内容筛选、事实纠正、去重或改写，不因内容问题重新付费生成。只检查导出需要的JSON结构与数量；额外元数据允许保留。不要预先为样本流净化候选池，也不要主动制造错误以保证实验优势。

本地脚本 `scripts/generate_description_bank.py`，服务器实验根目录 `/root/autodl-tmp/project_R/experiments/offline_description_bank`。沿用环境内requests，无新增依赖。API为 `https://api.aijws.com/v1/chat/completions`，模型ID `deepseek-v4.1-flash`；密钥已授权填入两端 `configs/description_api.local.json`，已忽略，服务器权限600。不要回显、提交或写进日志与文档。

全量已完成，最终输出为实验目录 `outputs/flowers102_broad100_20261020`，本地同名outputs目录存副本。102类、10200条、每层20/60/20已核对，status.complete=true、failures为空；没有审查文本内容或运行准确率实验。共613次HTTP返回，提供方usage为输入671228、输出554848、合计1226076 tokens。原始请求、响应、usage、断点与代码快照保留，旧元数据限制卡住的有效响应已直接复用，不重复生成。首轮输出另存 `outputs/flowers102_broad100_pass1_20261020`。最初PID603522，补齐PID604213和604710，均已退出；不要重新启动付费生成。完整日志与来源记录在最终输出内。

导出 `prompts_all.json` 是类名到100条文本列表，三个分层prompts文件为20/60/20条，`bank.json`保留层级与比较对象，`status.json`记录是否102类全部完成。未完成全量时不能冒称完整库；本次不启动分类准确率实验。当前脚本与旧120条版本提示不同，不向旧目录续跑。旧目录保留仅供查证历史，不混用为当前交付。

完整参数与说明见 `docs/flowers102-broad100-generation.md`。此前 `docs/hierarchical-description-bank.md` 为历史方案，命令不适用于当前全量库。

随后用户明确要求运行这份库的TDA＋全量融合，已完成。结果服务器 `/root/autodl-tmp/project_R/runs/flowers_broad100_all_20261020_165242`，本地results同名副本。脚本为实验目录 `scripts/evaluate_flowers_broad100.py`，复用原2463张视觉流与种子1、ViT-B/16及原配置，各类100句等权平均，缓存从空开始。TDA Top-1为74.4214%（1833/2463），纯文本73.1628%（1802），类别平均72.5036%；旧官方CuPL＋TDA为74.1778%（1827），本次净多6张。没有运行RDSS或筛选新文本，也没有重跑历史对照。输入哈希与交付一致，所有10200条实际未超77-token上限。保存指标、逐样本预测、文本特征、原型、配置、源码与日志；说明见 `docs/flowers102-broad100-all-results.md`。无需再次安装环境或重跑生成。

## 使用现有环境

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate /root/autodl-tmp/project_R/env
export PYTHONNOUSERSITE=1
unset PYTHONPATH
export CONDA_PKGS_DIRS=/root/autodl-tmp/project_R/cache/conda-pkgs
export PIP_CACHE_DIR=/root/autodl-tmp/project_R/cache/pip
export TMPDIR=/root/autodl-tmp/project_R/tmp
cd /root/autodl-tmp/project_R/RDSS
python -m pip check
```

需要启动完整 Caltech101 baseline 时使用以下命令。运行前检查 GPU 当前占用，避免干扰其他服务；当前文件记录启动方式，不表示新会话应自动启动实验。

```bash
HOME=/root/autodl-tmp/project_R/runtime-home \
PYTHONNOUSERSITE=1 PYTHONPATH= \
WANDB_MODE=disabled CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 \
/root/autodl-tmp/project_R/env/bin/python -B tda_runner.py \
  --config /root/autodl-tmp/project_R/runs/caltech101_vitb16_20261020_105422/configs \
  --datasets caltech101 \
  --data-root /root/autodl-tmp/project_R/data \
  --backbone 'ViT-B/16' \
  --wandb-log
```

保留 `--wandb-log` 并设置 `WANDB_MODE=disabled`：原代码测试循环无条件调用 `wandb.log()`，需要初始化，但本地验证不登录、不上传 WandB。`HOME` 仅对运行进程生效，使 CLIP 找到项目专属权重缓存，不修改真实用户 HOME。

当前仓库的 `configs/caltech101.yaml` 缺少 `negative.mask_threshold.upper`。完整测试使用上面结果目录内的配置补上 `upper: 1.0`，与官方当前配置一致；算法代码和仓库原配置未修改。不要直接使用缺字段的旧配置，也不要关闭负缓存来绕过错误。最初 16 张图片的检查没有覆盖到负缓存检索触发条件。

2026-10-20 验证结果：`pip check` 通过，CUDA 可用，ViT-B/16 加载成功。完整 Caltech101 baseline 已完成：2465 张、100 类，Top-1 为 94.08%，论文 Table 3 对应值为 94.24%，相差 -0.16 个百分点；接近但没有精确复现。测试循环约 31 秒，含启动约 42.86 秒。本次结果目录为 `/root/autodl-tmp/project_R/runs/caltech101_vitb16_20261020_105422`，含 `report.md`、`metrics.json`、`resources.csv`、`test.log`、`manifest.json` 和实际运行配置。RDSS 改进尚未实现。

## 已完成的 CuPL 对照实验

2026-10-20 已完成 DTD + ViT-B/16 的配对描述实验。结果目录为 `/root/autodl-tmp/project_R/runs/dtd_cupl_compare_20261020_113018`，主要文件为 `report.md`、`summary.csv`、`metrics.json`、`*_predictions.csv`、`prompt_audit.json`、`prompts/`、`visual_stream.pt` 和 `manifest.json`。本地副本位于当前工作区 `results/dtd_cupl_compare_20261020_113018`（被忽略）。

可复用程序是 `/root/autodl-tmp/project_R/experiments/cupl_dtd/compare_prompts.py`，描述输入在同目录 `prompts/`，使用现有独立环境和项目运行 HOME，不需要安装依赖。旧描述固定为 CuPL 官方提交 `719619c829792ff8f37298266e941b7f077c7f5c` 中的 `all_prompts/full_prompts/texture_prompts_full.json`，每类 60 条。新描述由用户要求的低成本 `gpt-6-luna` 子智能体生成，47 类各 10 条，生成指令、审阅修改与元数据均保留；不能称为已证实的“高质量描述”。

测试流种子 1，各组共用同一视觉特征与顺序、分别清空并更新视觉缓存。TDA 模板基线 44.80%；旧 CuPL 全量 54.08%；旧池完全去重 54.26%；旧池每类抽取 10 条（固定种子 0/1/2）分别为 53.37%/51.65%/51.89%，均值 52.30%；新版静态 10 条 50.53%；新版全部均匀复制一次仍为 50.53%；只将首条额外复制 10 次为 44.62%；新版历史视觉原型动态 Top-3 为 50.12%。简单动态对照使用静态新版伪标签、归一化熵 <=0.2 门控，当前预测后才更新，k=3 固定，没有用真实标签或未来样本；它不是完整 RDSS。

当前证据支持静态 CuPL 有用，但没有证明新版生成一定更好或动态选择一定更好。均匀重复不改变均值原型；偏向性重复可能改变权重并损害本次表现。不要把人为重复压力测试当成所有语义冗余有害的证明，也不要按测试结果反复改写描述或选择参数；后续方法开发使用验证集。预运行 `dtd_cupl_compare_20261020_112916` 因评估前文本清理被中止，保留 `ABORTED.txt`，不要混入正式比较结果。

## 网络与项目边界

此前服务器可访问 Caltech 数据站点、OpenAI 权重站点和 PyTorch 官方 wheel 源，但直连 GitHub HTTPS 超时，`/etc/network_turbo` 提示该地区不支持内置加速。网络状态应实测，不假定永久不可用。代码曾经本机 Git 克隆后经 SSH 中转，保留完整 `.git` 和个人仓库 origin。后续已在功能分支实现固定语义缓存基线，并按用户要求提交AGENTS.md；当前状态见下方功能分支记录。

若确需新增依赖，先检查现有依赖能力，只安装到上述独立环境，并同步依赖记录。不要盲目执行原始 `requirements.txt`、重新克隆仓库、重新下载已有数据或覆盖用户改动。

研究主题是“双缓存协同的免训练 CLIP 测试时适应”。TDA 自带正负视觉缓存；课题拟新增文本语义缓存，并使用 RDSS（相关性、混淆类别判别性、冗余抑制）选择语义。不要混淆这两种“双缓存”。

## 用户工作约定

回答优先使用少量完整段落，列表通常不超过 5 项；解释清楚整体流程、因果和关键区别。每次最终答复以“这是我的回答qwq：”开头。

不保留过时兼容层，不写 migration 或 fallback。选择满足当前需求的最简单实现，不做预防性抽象或多余配置层。先跑通最小端到端流程，再增加能力，不为未完成的复杂度拆掉能运行的部分。组件保持模块化、关注点分离。

先检查项目已有依赖，优先成熟且维护中的库；先研究成熟产品和已验证模式，不无故重写。架构决策考虑长期使用，不采用“先这样以后再换”的临时方案。

没有用户明确要求，不修改代码、安装环境、提交或 push。当前已授权的数据准备、独立环境安装和本文档编写不等于授权未来任意变更；新任务按用户实际授权范围执行。

## 固定语义缓存功能分支

用户随后明确要求提交并push此基线，再新建分支完成首轮RDSS；该要求覆盖下文此前暂不提交代码的状态。基线分支为 `codex/fixed-semantic-cache-baseline`，后续RDSS分支为 `codex/rdss-first-round`。首轮参数固定后做配对测试，不启动描述生成或安装依赖。

2026-10-20用户明确授权新建功能分支、修改代码并运行固定语义缓存基线，要求直接提交AGENTS.md。实际仓库分支为 `codex/fixed-semantic-cache-baseline`，起点main的e697fb0；现有AGENTS初次单独提交22063ca。功能代码保留在该分支工作区，不自动提交或push。

新增 `semantic_cache.py`、`experiments/evaluate_fixed_semantic.py` 和 `tests/test_fixed_semantic.py`。模板原型保持不变，伪标签、熵和视觉缓存写入仅由模板分支驱动。每类100条已有单位文本向量等权平均并归一化，形成固定语义原型；两路文本均使用100倍相似度尺度。最终分数为模板TDA分数加eta乘语义分数，首次eta=1。没有动态筛选或修改原TDA源码，没有新依赖。

4项单元测试通过，Flowers-102完整2463张配对评估完成。结果服务器目录 `/root/autodl-tmp/project_R/runs/flowers_fixed_semantic_20261020_194908`，本地results同名副本。模板TDA对照71.4982%（1761），固定语义基线73.6500%（1814），类别平均72.3345%。相对模板纠正92、损坏39，净多53张（+2.1518个百分点）。eta=0的2463条预测与此前模板TDA逐张一致；语义原型与此前100句融合原型逐元素一致；融合求和检查通过。之前74.4214%（1833）是替换模板的另一实验条件。

复用ViT-B/16既有图像特征、种子1顺序、描述向量和原Flowers配置，不重复编码、不调用API、不安装环境。重放约5.98秒，总脚本约6.69秒，不含特征提取。保存metrics、逐样本预测、视觉缓存轨迹、各路logits、语义原型、输入与源码快照。运行入口为仓库内 `experiments/evaluate_fixed_semantic.py --semantic-weight 1`，使用已有独立解释器和本文的运行环境变量。说明见 `docs/flowers102-fixed-semantic-results.md`。
