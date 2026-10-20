# Flowers-102 属性错配噪声实验

2026-10-20，分支 `codex/rdss-first-round`。入口 `experiments/evaluate_semantic_noise.py`。本轮源码与记录已随实验检查点归档，后续状态见PROJECT_STATUS.md。

## 构造与公平比较

保留原始每类100条描述，另外追加每类25条或50条。随机种子20261020；每条噪声随机取其他花类前20条概述之一，仅将其中的供体类别名替换成目标类别名，原来属性、背景和上下文均保留。这是人工属性迁移/错配噪声，单条是否不适配由图像决定，没有根据测试图片或正确率选取供体。全部5100条供体、原句、替换后文本和索引保存在noise_provenance.json。

原库不覆盖，25条是50条的前缀。两个噪声库分别125条/类（人工噪声占20%）、150条/类（占33.33%）。全量融合与RDSS用完全相同候选库。只编码新增5100条文本，复用已有ViT-B/16图像特征及原始描述向量，不调用付费API，不安装依赖。例：供体句被改为“...a pink primrose flower...orange tubular corolla...”，保留了供体的橙色筒状花冠属性。

RDSS仍使用k=8、近期队列32、衰减0.9、熵阈值0.2、模板/TDA一致门控、混淆top3、判别权重1、冗余权重0.1、语义融合权重1。每类冷启动原型使用对应库全部句子融合，固定方法和RDSS相同。所有条件共享同一TDA视觉状态；先预测，再更新语义。原库的模板、固定语义、RDSS三路2463条预测与上一轮逐张一致。

## 结果

| 每类追加 | 总条数/类 | 全量融合Top-1 | RDSS Top-1 | 全量变化 | RDSS变化 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 100 | 73.6500%（1814） | 73.2034%（1803） | 0 | 0 |
| 25 | 125 | 73.8124%（1818） | 73.6500%（1814） | +0.1624 pp | +0.4466 pp |
| 50 | 150 | 73.6500%（1814） | 73.2846%（1805） | 0.0000 pp | +0.0812 pp |

共2463张，原始TDA为71.4982%（1761）。这里的变化均相对各方法自己的原库成绩。

每种RDSS条件接收1482次更新。在这些更新各自选中的8条描述中，人工噪声占比为：追加25条时11.35%，追加50条时19.71%，低于对应池中的20%和33.33%。该统计按更新次数加权，包含同一描述在不同更新中再次入选，不是独立原句去重计数。

本轮全量融合没有被这两档人工属性错配压低；RDSS在噪声条件下也没有下降。RDSS有降低人工噪声选入比例的现象。这轮噪声保留正确类别名称，一些迁移属性也可能与目标图像相容，因而噪声不等于每条都产生负面影响。

## 资源与复现

文本编码及准备约7.15秒，缓存特征重放约22.42秒，总计29.94秒；重放PyTorch峰值已分配显存约72.30MiB，耗时不包含重新提取图像特征。CSV正确数和源码哈希另行复核通过。

服务器结果：`/root/autodl-tmp/project_R/runs/flowers_semantic_noise_20261020_202831`；本地results同名副本。包括metrics、predictions、noise_provenance、prompts_add0/25/50、noise_embeddings、semantic_updates、final_selections、logits、参数与源码快照。

```bash
cd /root/autodl-tmp/project_R/RDSS
HOME=/root/autodl-tmp/project_R/runtime-home PYTHONNOUSERSITE=1 PYTHONPATH= WANDB_MODE=disabled CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 /root/autodl-tmp/project_R/env/bin/python -B experiments/evaluate_semantic_noise.py
```
