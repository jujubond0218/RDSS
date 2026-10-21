# 服务器环境、数据与资源

更新：2026-10-21。本文件只维护可复用环境、数据来源和运行约定；当前代码与结果见 [项目状态](../PROJECT_STATUS.md)，各轮实验见 [实验索引](../EXPERIMENT_INDEX.md)。本地目录用于资料与同步，实际计算在服务器。已有资源无需重新准备。

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

2026-10-20 验证结果：`pip check` 通过，CUDA 可用，ViT-B/16 加载成功。完整 Caltech101 baseline 已完成：2465 张、100 类，Top-1 为 94.08%，论文 Table 3 对应值为 94.24%，相差 -0.16 个百分点；接近但没有精确复现。测试循环约 31 秒，含启动约 42.86 秒。本次结果目录为 `/root/autodl-tmp/project_R/runs/caltech101_vitb16_20261020_105422`，含 `report.md`、`metrics.json`、`resources.csv`、`test.log`、`manifest.json` 和实际运行配置。该记录为原始TDA验证；后续RDSS实现见 [项目状态](../PROJECT_STATUS.md)。

## 网络与项目边界

此前服务器可访问 Caltech 数据站点、OpenAI 权重站点和 PyTorch 官方 wheel 源，但直连 GitHub HTTPS 超时，`/etc/network_turbo` 提示该地区不支持内置加速。网络状态应实测，不假定永久不可用。代码曾经本机 Git 克隆后经 SSH 中转，保留完整 `.git` 和个人仓库 origin。分支和检查点历史见 [实验索引](../EXPERIMENT_INDEX.md)，当前状态见项目状态页。

若确需新增依赖，先检查现有依赖能力，只安装到上述独立环境，并同步依赖记录。不要盲目执行原始 `requirements.txt`、重新克隆仓库、重新下载已有数据或覆盖用户改动。

研究主题是“双缓存协同的免训练 CLIP 测试时适应”。TDA 自带正负视觉缓存，项目另有文本语义缓存；研究定义与当前进度分别见研究计划和项目状态，不将两种“双缓存”混为一谈。

## Git 同步与认证

实际仓库 `/root/autodl-tmp/project_R/RDSS`，镜像 `C:\Users\Administrator\Documents\ChatGPT\毕设\tmp\rdss-sync`，origin 为 `https://github.com/jujubond0218/RDSS.git`。当前分支与授权状态以 [项目状态](../PROJECT_STATUS.md) 为准，不自动提交或推送。Git 凭据保存在本机系统凭据管理器，不复制到服务器；服务器可通过已有 SSH 中转 Git bundle 同步。

本机 Git Credential Manager 2.7.3 的设备登录若长时间无响应，可仅对该次进程设置 `DOTNET_SYSTEM_NET_HTTP_SOCKETSHTTPHANDLER_HTTP2SUPPORT=0`，再执行 `git credential-manager github login --username jujubond0218 --device --no-ui`。该方法来自此前成功登录记录，不代表需要重新登录；不修改系统网络配置。

## 描述库与连续域数据

Flowers 当前库为每类100条、20/60/20，共10200条，来源、完成状态和原始响应见 [生成记录](../experiments/flowers102-broad100-generation.md)。保留模型原文，不审查、去重或改写，不因内容问题重新付费生成。API 为 `https://api.aijws.com/v1/chat/completions`，模型 `deepseek-v4.1-flash`；私有配置被忽略，服务器权限600，不回显密钥。

PACS 最小流数据为 `/root/autodl-tmp/project_R/data/pacs_minimal`；来源、版本、哈希和700张抽样见 [PACS报告](../experiments/pacs-minimal-shift.md)。Parquet准备仅使用已安装PyArrow的base解释器，不向base安装任何包；推理使用项目独立环境。NICO最小流为 `/root/autodl-tmp/project_R/data/nico_minimal`，1120张、20类、六上下文子集，来源与唯一性记录见 [NICO报告](../experiments/nico-minimal-shift.md)。`nico_minimal_incomplete_20261020` 是未完成的历史抽样，禁止作为正式数据。域回归使用新图片，域边界不输入RDSS。
