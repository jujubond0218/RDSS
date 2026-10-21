# PACS 最小连续域变化实验

> 历史实验：本文准确率与协议属于2026-10-20首轮源码。当前同名运行脚本已接入第二版，不能直接按新代码复跑并称为首轮复现；使用保存的源码快照或Git检查点。当前方法见 [第二版报告](rdss-v2.md)。

2026-10-20；分支 `codex/rdss-first-round`。本轮源码与文档已随实验检查点归档。复用现有RDSS实现和已有环境，没有安装依赖或调用付费API。

## 数据与测试流

PACS原始项目：https://dali-dl.github.io/project_iccv2017.html 。镜像来源：https://huggingface.co/datasets/flwrlabs/pacs ，固定版本 `394113073258ead631f617d2e13bb377c0715c4b`，共9991张、7类、4域。191395900字节Parquet经hf-mirror.com下载后，SHA256与Hugging Face LFS公布的 `4fc041ee92eec6043fe6e2859e8bdd138e5f958bc621afd153879812cbe65ff5` 完全一致。

7类为dog、elephant、giraffe、guitar、horse、house、person。固定种子1，从每个域每类抽样，构造 `photo → art_painting → cartoon → sketch → photo`。每阶段每类20张，共140张；照片第二次使用另一批20张，合计700张互不重复。阶段内打乱，各方法使用完全相同顺序，域切换时不重置缓存。

图片位于 `/root/autodl-tmp/project_R/data/pacs_minimal/images`，抽样清单、图像哈希、原始行号在同目录manifest.json。完整源文件在 `/root/autodl-tmp/project_R/downloads/pacs_39411307.parquet`。

## 方法与描述

先做28张TDA冒烟测试，再跑700张原始TDA；之后重新从空视觉缓存开始，配对计算TDA、TDA+全量语义、TDA+RDSS。配对流的700条TDA预测与单独TDA运行逐张一致。

CLIP ViT-B/16，模板固定 `a photo of a <class>.`。TDA正负缓存超参数直接沿用已有Flowers配置，未在PACS上搜索参数。RDSS沿用k=8、近期可信队列32、衰减0.9、熵阈值0.2、模板/TDA一致门控、混淆top3、Rel+Disc-0.1Red，语义融合权重1；先预测再更新。

本轮在当前对话中编写静态描述素材，脚本 `pacs_description_bank.py` 渲染700条描述。每类5条概述、15条细节、5条类别对比，分别渲染为照片、油画、卡通、素描四种风格，得到每类100条：20概述、60细节、20对比，每域25条。素材在看测试图片或预测结果前固定，覆盖颜色、姿态、部件、视角等变化；未按本次成绩改写。全量和RDSS共用同一库，冷启动原型相同。域标签只用于构造流与事后统计，不输入RDSS筛选。

## 结果

28张TDA冒烟：28/28。700张完整最小测试：

| 阶段 | TDA | TDA+全量 | TDA+RDSS |
| --- | ---: | ---: | ---: |
| 照片1 | 100.00% | 100.00% | 100.00% |
| 油画 | 99.29% | 99.29% | 97.14% |
| 卡通 | 99.29% | 100.00% | 99.29% |
| 素描 | 87.86% | 88.57% | 89.29% |
| 照片2 | 100.00% | 100.00% | 100.00% |
| 总计700张 | 97.2857%（681） | 97.5714%（683） | 97.1429%（680） |

RDSS相比固定全量纠正2张、新增错误5张，净少3张；相比TDA少1张。该最小子集整体较容易，主要错误集中在素描阶段。

## 动态行为

RDSS共接收656次语义更新，其中8次伪标签错误。统计各阶段开始35张、结束35张内的可信更新，选出的8条描述中当前域风格描述占比：

| 域阶段 | 开始段 | 结束段 |
| --- | ---: | ---: |
| 照片1 | 76.47% | 80.71% |
| 油画 | 39.92% | 88.26% |
| 卡通 | 35.66% | 80.51% |
| 素描 | 4.63% | 29.74% |
| 照片2 | 18.21% | 74.26% |

油画、卡通和再次进入照片时，选句风格随近期数据发生变化；进入素描后切换较慢，该阶段全部更新中仍有73.26%选句为卡通风格，21.96%为素描风格。这提示后续需要检查近期证据的切换速度，以及相关性打分对卡通/素描的区分，而不是只继续增加描述噪声。风格占比是行为统计，不能直接等同每条描述的分类贡献。

## 复现与输出

数据准备使用服务器已经安装PyArrow的base解释器；CLIP与TDA/RDSS运行仍使用原独立环境。不要把PyArrow安装到项目环境来重复准备。

```bash
cd /root/autodl-tmp/project_R/RDSS
# 数据目录不存在且源Parquet已下载时执行一次
/root/miniconda3/bin/python -B experiments/prepare_pacs_minimal.py
HOME=/root/autodl-tmp/project_R/runtime-home PYTHONNOUSERSITE=1 PYTHONPATH= WANDB_MODE=disabled CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 /root/autodl-tmp/project_R/env/bin/python -B experiments/evaluate_pacs_minimal.py
```

结果服务器目录 `/root/autodl-tmp/project_R/runs/pacs_minimal_shift_20261020_210546`；日志 `/root/autodl-tmp/project_R/logs/pacs_minimal_shift_20261020.log`；本地results同名副本。含report、metrics、逐样本predictions、semantic_updates、每阶段选句索引、描述原句与类型/风格元数据、特征与原型、数据清单、参数和源码快照。

总脚本约12.71秒，其中配对缓存重放3.84秒；总耗时不含下载和数据抽取。700张图像哈希检查、TDA逐张对照、冷启动一致性通过；CSV正确数和源码快照哈希另行复核通过。
