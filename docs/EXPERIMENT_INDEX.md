# 历史实验与复现索引

更新：2026-10-21。本页只索引历史事实、来源和复现入口，不维护另一套研究计划或重复完整结果表。当前状态见 [PROJECT_STATUS.md](PROJECT_STATUS.md)，后续方案见 [研究计划](semantic-adaptation-experiment-plan.md)。历史结果文件和源码快照保留，不因文档整理覆盖。

## 结果与来源

服务器结果根目录为 `/root/autodl-tmp/project_R/runs`，本地对应 `results/`，均不提交大文件。表中的程序位置是原实验来源；当前同名仓库脚本若已更新，按对应报告的历史快照或Git检查点复现。

| 实验 | 主要报告 | 服务器结果目录名 |
| --- | --- | --- |
| Caltech101/DTD原始TDA | [环境与baseline记录](environment/ENVIRONMENT.md) | `caltech101_vitb16_20261020_105422`、`dtd_vitb16_20261020_110941` |
| DTD官方CuPL、新池、重复与简单动态对照 | [DTD早期探索](experiments/dtd-cupl-exploration.md) | `dtd_cupl_compare_20261020_113018` |
| Flowers CuPL全量/人工第一版 | [第一版结果](experiments/flowers102-cupl-manual-results.md) | `flowers_cupl_manual_20261020_120643` |
| Flowers人工第二版 | [筛选输入](experiments/flowers102-manual-v2-selection.md)、[结果](experiments/flowers102-manual-v2-results.md) | `flowers_cupl_manual_v2_20261020_150338` |
| Flowers broad100库与全量替换文本实验 | [生成来源](experiments/flowers102-broad100-generation.md)、[全量结果](experiments/flowers102-broad100-all-results.md) | `flowers_broad100_all_20261020_165242` |
| 固定语义增量融合 | [固定语义基线](experiments/flowers102-fixed-semantic-results.md) | `flowers_fixed_semantic_20261020_194908` |
| 首轮RDSS | [首轮方法与结果](experiments/rdss-first-round.md) | `flowers_rdss_first_20261020_201102` |
| 人工属性错配 | [噪声实验](experiments/flowers102-semantic-noise.md) | `flowers_semantic_noise_20261020_202831` |
| PACS连续域最小流 | [PACS报告](experiments/pacs-minimal-shift.md) | `pacs_minimal_shift_20261020_210546` |
| NICO连续上下文最小流 | [NICO报告](experiments/nico-minimal-shift.md) | `nico_minimal_shift_20261020_220735` |
| 首轮组件诊断与域流拆分 | [原因诊断](experiments/rdss-diagnosis.md) | `rdss_diagnosis_20261020_223013`，包含 `domain_components/` |
| 第二版及递进消融 | [第二版报告](experiments/rdss-v2.md) | `rdss_v2_20261021_003009` |

Flowers broad100最终库在 `/root/autodl-tmp/project_R/experiments/offline_description_bank/outputs/flowers102_broad100_20261020`，本地outputs同名副本；已有完整库，不重启付费生成。旧120条提示、名称澄清与生成目录仅作为 [历史方案](archive/hierarchical-description-bank.md) 查证，不混入当前100条库。

DTD原始对照程序在 `/root/autodl-tmp/project_R/experiments/cupl_dtd/compare_prompts.py`；Flowers人工对照在 `/root/autodl-tmp/project_R/experiments/cupl_flowers`；第二版人工输入为其 `manual_v2` 子目录。Flowers第一版最多3条的旧脚本不能直接运行712条第二版输入。中止的DTD预运行 `dtd_cupl_compare_20261020_112916` 有ABORTED标记，不计为正式结果。

## Git与检查点验证

固定语义基线分支 `codex/fixed-semantic-cache-baseline`，实现提交 `8f66b71299b6cc415f458ec8150a2d3efb6a43c7`；首轮RDSS实现提交 `a54f6d6a0ed914c1c37a50b7a8189e8f3a2b92ee`。后续归档检查点 `d3cb465bdb84d59e3df240a87944ca83ac1be395` 已推送到 `origin/codex/rdss-first-round`。第二版开发后用户已另行授权归档提交与推送，再新建优化分支；活动状态以项目状态页为准，不从历史推断任意新操作授权。

2026-10-20归档时23项测试通过，统一诊断重放精确复现三数据集，验证目录 `rdss_diagnosis_20261020_232159`，日志 `project_R/logs/rdss_checkpoint_tests_20261020.log` 与 `rdss_checkpoint_replay_20261020.log`。第二版历史入口验证目录 `rdss_diagnosis_20261021_003512`，31项测试与产物审计细节见第二版报告。这里记录既有检查，不表示文档整理再次运行实验。

## 历史结果不可互换

人工Flowers第二版75.40%是使用验证图且已知前次测试诊断后的探索性复测，描述池与视觉状态协议均不同；不是在线自动RDSS结果。broad100全量替换文本74.4214%与保留模板后增量语义融合73.6500%也是不同条件。PACS/NICO为最小公开流，不代表完整数据集。所有失败、人工噪声和未完成准备均按原标记保留。
