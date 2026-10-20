# 首轮 RDSS：实现与复现

代码分支：`codex/rdss-first-round`，基于固定语义基线提交 `8f66b71299b6cc415f458ec8150a2d3efb6a43c7`。基线已通过本机 Git 登录成功推送到 origin。首轮 RDSS 实现提交为 `a54f6d6a0ed914c1c37a50b7a8189e8f3a2b92ee`，本分支同步至 origin。

## 整体流程

原始提示模板和 TDA 正负视觉缓存保持原有计算方式。文本描述库直接复用 Flowers-102 每类 100 条原始描述及已编码向量，不重新生成、不清洗。每类动态语义原型冷启动时与全量融合基线完全相同。

图片先使用当前状态完成预测：`final = template_TDA + 100 * image_feature @ semantic_prototypes`。之后，只有模板预测归一化熵不超过 0.2，且模板与 TDA 的类别预测一致，才接收它作为语义筛选证据。当前图片的语义更新只影响之后的预测。

TDA 原本保留低熵视觉样本，不能保证跟随近期变化，因此另设每类最多 32 张的近期可信视觉特征队列，保留原始 TDA 状态不变。近期队列以 `可靠度(1-熵) * 0.9^同类后续接收次数` 加权。新证据到达后，重新为该类全部 100 条候选描述打分；旧描述没有被删除，之后仍能再次入选。

## 描述筛选

相关性 Rel 是候选描述与该类近期证据的加权平均余弦相似度。混淆候选取当前模板分数最高的其他 3 类；只使用已经出现且有可信证据的混淆类别。判别性 Disc 为 Rel 减去候选描述与这些混淆类证据的最高相似度，没有混淆证据时 Disc 为零。

贪心选取 8 条，评分为 `Rel + Disc - 0.1 * Red`。Red 是与本轮已经选中描述的最大非负余弦相似度；首条没有冗余项。将 8 条单位描述向量平均并归一化，更新该类语义原型。其他类别保持当前状态。

实现分别位于 `rdss.py`（近期证据与筛选）、`visual_cache.py`（共享的模板 TDA 视觉状态）和 `experiments/evaluate_rdss.py`（配对实验）。没有新依赖，也没有修改原始 `tda_runner.py`。

## 首轮对比

固定语义基线、只按相关性选择的动态 Top-k、完整 RDSS 均与同一个模板 TDA 状态融合。动态 Top-k 与 RDSS 的队列、门控、k、更新时序完全相同，只把判别和冗余权重设为零。所有参数在正式预测前写入 `parameters.json`。

使用同一份 ViT-B/16 图像特征、种子 1 的 2463 张顺序、每类 100 条描述向量和 Flowers 配置，复用已保存特征；运行时间不包含图像和文本编码。真实标签只用于最终计分与输出预测表。

## 运行

```bash
cd /root/autodl-tmp/project_R/RDSS
HOME=/root/autodl-tmp/project_R/runtime-home PYTHONNOUSERSITE=1 PYTHONPATH= WANDB_MODE=disabled CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 /root/autodl-tmp/project_R/env/bin/python -B -m unittest discover -s tests -p 'test_*.py' -v
HOME=/root/autodl-tmp/project_R/runtime-home PYTHONNOUSERSITE=1 PYTHONPATH= WANDB_MODE=disabled CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 /root/autodl-tmp/project_R/env/bin/python -B experiments/evaluate_rdss.py
```

入口当前直接复用本项目已存的 Flowers 特征路径，运行前按 AGENTS.md 检查已有资源。结果目录指针为 `/root/autodl-tmp/project_R/runs/latest_flowers_rdss_first.txt`。

## 本轮结果

服务器目录：`/root/autodl-tmp/project_R/runs/flowers_rdss_first_20261020_201102`。

| 条件 | Top-1 | 正确数/2463 | 类别平均正确率 |
| --- | ---: | ---: | ---: |
| 模板 TDA | 71.4982% | 1761 | 71.0076% |
| TDA + 固定全量描述语义缓存 | 73.6500% | 1814 | 72.3345% |
| TDA + 动态相关性 Top-k | 73.6906% | 1815 | 73.0705% |
| TDA + RDSS | 73.2034% | 1803 | 72.3759% |

RDSS 对固定基线纠正 52 张、新增错误 63 张，净少正确 11 张，差 -0.4466 个百分点；对动态相关性 Top-k 净少正确 12 张。本轮不继续根据同一测试集分数改参数。动态 Top-k 与完整 RDSS 的差别仅在判别和冗余打分，下一步可从这些项的选句变化定位损失来源。

两个动态条件均接收 1482 个可信样本、更新 87 类；RDSS 发生 1471 次有序选择列表变化，其中 1395 次更新有已出现混淆类的视觉证据。选择列表变化包含顺序改变，不能直接解读为同等次数的语义原型变化。15 个类别未收到通过门控的样本，保持全量融合冷启动原型。

11 项单元测试通过，包含属性切换与回归、容量边界、可靠性门控、近期加权、判别性和冗余抑制。共用视觉缓存模块重构后重跑固定基线，保持 1761/1814；本轮又断言模板和固定分支的全部 2463 条预测与之前基线逐张一致。

结果包含 `metrics.json`、`parameters.json`、`predictions.csv`、`semantic_updates.jsonl`、`final_selections.json`、`visual_cache_trace.csv`、各条件 logits、初末语义原型和输入/源码快照。逐次选句记录使用原始描述索引，可以回到 `prompts_all.json` 定位；`final_selections.json` 同时保存原句。

本轮缓存特征重放约 46.99 秒，脚本总计约 47.94 秒，PyTorch 峰值已分配显存约 32.27 MiB；服务器同时有其他 GPU 任务，这不是独占 GPU 的端到端耗时。
