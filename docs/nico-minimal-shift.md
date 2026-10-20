# NICO++ 最小连续上下文实验

2026-10-20；分支 `codex/rdss-first-round`。新增脚本、本文及AGENTS已随实验检查点归档。算法参数与首轮RDSS一致，没有安装依赖或调用付费API。

## 数据来源与抽样

原始论文仓库 https://github.com/xxgege/NICO-plus 的Dropbox入口显示暂时禁用，清华云盘入口显示链接不存在。本次使用公开重组子集 https://huggingface.co/datasets/2018LZY/NICOplusplus ，固定版本 `66bea9905018646633b3c3664027e7e80cdcb687`，属于20类、六上下文子集。压缩包1374607687字节，下载后SHA256与该镜像LFS公布的 `c5531759655342b7adf8b2ede51a97954e8114816106d64a100ed6774bd3eafa` 一致。

保留子集全部20类：cactus、car、chair、crab、dolphin、elephant、fox、giraffe、gun、kangaroo、lion、mailbox、pumpkin、racket、sailboat、sheep、spider、tent、tortoise、umbrella。

原计划每类每上下文10张；实际文件检查发现秋季dolphin只有8张。首次准备在计分前停止，半成品保留为 `/root/autodl-tmp/project_R/data/nico_minimal_incomplete_20261020`，并有INCOMPLETE.txt说明。随后统一固定为每类每阶段8张，保留全部类别和上下文，未据模型成绩选择类别或样本。

流为 `grass → water → rock → autumn → dim → outdoor → grass`。每阶段20类×8张=160张，共1120张。回归草地使用另一批图片；种子1，阶段内打乱，内容SHA256全局去重，不在域切换时清缓存。原始ZIP成员路径、图像哈希、类别与阶段均记录在manifest.json。

数据目录 `/root/autodl-tmp/project_R/data/nico_minimal`；源压缩包 `/root/autodl-tmp/project_R/downloads/nico_66bea990.zip`。准备入口 `experiments/prepare_nico_minimal.py`，使用已有独立解释器即可，不需要PyArrow。

## 描述与算法

`experiments/nico_description_bank.py` 在当前对话中编写每类10条外观素材和5个静态对比类别，再通过提示组合生成每类100条，共2000条。组成是20条通用外观表述、60条上下文细节（每个上下文10条）、20条显式类别对比。它们包含相同基础素材的上下文及措辞变体，并非每类100条独立生成的事实。对比类别预先静态指定，未利用测试混淆矩阵。三组计分前描述库即固定。

CLIP ViT-B/16，模板 `a photo of a <class>.`。TDA使用已有Flowers正负缓存参数。RDSS仍为k=8、近期可信队列32、衰减0.9、模板归一化熵<=0.2并与TDA类别一致才更新、混淆top3、Rel+Disc-0.1Red；语义权重1。全量与RDSS用同一个100句池，冷启动相同。先预测再更新语义，域标签只用于排流与事后诊断。

先跑28张TDA冒烟，再跑1120张单独TDA，最后清空视觉状态、共享同一TDA缓存做三路配对。配对TDA的1120条预测与单独运行逐张一致。

## 结果

28张冒烟24/28（85.7143%）。1120张结果：

| 方法 | 正确数 | Top-1 |
| --- | ---: | ---: |
| TDA | 1037 | 92.5893% |
| TDA+全量描述 | 1044 | 93.2143% |
| TDA+RDSS | 1036 | 92.5000% |

| 阶段 | TDA | 全量描述 | RDSS |
| --- | ---: | ---: | ---: |
| 草地1 | 94.38% | 95.00% | 93.75% |
| 水边 | 86.88% | 87.50% | 86.25% |
| 岩石 | 93.75% | 94.38% | 93.75% |
| 秋季 | 93.75% | 93.12% | 91.88% |
| 昏暗 | 90.00% | 91.88% | 90.62% |
| 户外 | 93.75% | 95.62% | 96.25% |
| 草地2 | 95.62% | 95.00% | 95.00% |

RDSS相对全量纠正3张、新增错误11张，净少8张（-0.7143个百分点）；相对TDA少1张。语义门控接收979次更新，其中20次伪标签错误。

上下文词命中统计写入metrics和semantic_updates。通用/对比描述标为neutral，而且grass、水边等场景与outdoor语义可能重叠，因此“描述标记上下文是否等于当前标签”只是选句分布统计。

## 验证、输出与复现

源ZIP的LFS哈希一致、1120张内容哈希互不重复、每类每阶段均8张、配对TDA逐张一致、冷启动原型一致，均通过；另行复核CSV正确数和源码快照哈希。总脚本20.50秒，配对缓存重放5.70秒；不含下载与数据准备。

服务器结果 `/root/autodl-tmp/project_R/runs/nico_minimal_shift_20261020_220735`，日志 `/root/autodl-tmp/project_R/logs/nico_minimal_shift_20261020.log`；本地results同名副本。包含report、metrics、逐样本predictions、逐次semantic_updates、各阶段选句、2000条原句与类型/上下文元数据、图像/文本特征、参数、原型、来源清单与源码快照。

```bash
cd /root/autodl-tmp/project_R/RDSS
# 源ZIP已校验、正式数据目录不存在时执行一次
/root/autodl-tmp/project_R/env/bin/python -B experiments/prepare_nico_minimal.py
HOME=/root/autodl-tmp/project_R/runtime-home PYTHONNOUSERSITE=1 PYTHONPATH= WANDB_MODE=disabled CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 /root/autodl-tmp/project_R/env/bin/python -B experiments/evaluate_nico_minimal.py
```
