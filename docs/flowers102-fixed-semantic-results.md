# Flowers-102 固定语义缓存基线

2026-10-20在服务器仓库新建分支 `codex/fixed-semantic-cache-baseline`，起点为main的e697fb0。AGENTS.md先单独提交为22063ca，补充结果后再提交为a3cb845。用户随后明确授权将基线代码提交并push，再新建 `codex/rdss-first-round` 完成首轮RDSS。

本次保持原始提示模板、CLIP ViT-B/16及TDA视觉缓存逻辑。每类已有100条描述向量等权平均并归一化，形成固定语义原型。当前图片与原型的100倍余弦相似度为语义logits，最终分数为：

```text
final = template_logits + positive_visual_logits - negative_visual_logits
        + eta * fixed_semantic_logits
eta = 1.0
```

正负视觉缓存的伪标签、预测熵、概率图、写入和检索都仅由模板分支驱动。eta=0对照与eta=1条件共享同一视觉状态，语义分数不进入视觉缓存更新。固定语义缓存始终使用全部100条描述，没有在线挑选或文本更新。

| 实验条件 | Top-1 | 正确数／2463 | 类别平均Top-1 |
| --- | ---: | ---: | ---: |
| 模板TDA，本次配对对照 | 71.50% | 1761 | 71.01% |
| 模板TDA＋固定语义缓存，eta=1 | **73.65%** | **1814** | **72.33%** |
| 旧替换模板实验，历史参考 | 74.42% | 1833 | 72.50% |

新基线相对模板TDA纠正92张、新增错误39张，净多53张，提高2.1518个百分点。相对旧替换式条件少19张、低0.7714个百分点；两者的原始文本分支和视觉缓存写入依据不同，旧结果仅作历史参考。本次eta固定为1，没有搜索测试集上的融合权重。

4项CPU单元测试通过，覆盖100条等权聚合、类别到分数列的顺序、查询不改动固定缓存与源向量、零权重保持TDA以及加性融合。完整2463张评估中，eta=0所有逐样本预测与此前模板TDA完全一致；固定语义原型与旧全量100条原型逐元素一致；每个样本的融合求和均通过精确检查。预测CSV独立计数为模板1761、新基线1814。

实验复用原先保存的2463张图像特征、种子1顺序、描述向量及原Flowers配置；无图像/文本重新编码、生成API调用或新增依赖。重放5.98秒，总脚本6.69秒，不是包含特征提取的端到端时间。本进程PyTorch峰值分配11.67MiB、保留32MiB，不代表GPU总显存峰值。

服务器代码：`/root/autodl-tmp/project_R/RDSS/semantic_cache.py`、`experiments/evaluate_fixed_semantic.py`、`tests/test_fixed_semantic.py`。原 `tda_runner.py` 未修改。

服务器结果：`/root/autodl-tmp/project_R/runs/flowers_fixed_semantic_20261020_194908`；本地副本：`results/flowers_fixed_semantic_20261020_194908`。保存 `metrics.json`、`predictions.csv`、`visual_cache_trace.csv`、`logits.pt`、`semantic_prototypes.pt`、描述原文、源码快照和来源哈希。视觉流和描述向量原件路径、哈希记录在manifest中，避免重复复制大文件。

在仓库目录使用既有环境运行：

```bash
HOME=/root/autodl-tmp/project_R/runtime-home \
PYTHONNOUSERSITE=1 PYTHONPATH= WANDB_MODE=disabled \
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 \
/root/autodl-tmp/project_R/env/bin/python -B \
  experiments/evaluate_fixed_semantic.py --semantic-weight 1
```
