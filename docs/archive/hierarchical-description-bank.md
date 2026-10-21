# 分层描述库早期方案与文献来源

历史方案日期：2026-10-20；整理日期：2026-10-21。本文件只保留早期调研与生成条件的区别，不再提供旧版执行命令。当前交付已完成，每类100条、20/60/20，全部102类共10200条，唯一现行来源记录见 [broad100生成记录](../experiments/flowers102-broad100-generation.md)。旧120条方案、名称hint和目录不能用于当前库的续跑。

## 分层比较的参考

早期提到的“gpt-power”对应讨论中的 **ChatGPT-Powered Hierarchical Comparisons for Image Classification**，Zhiyuan Ren、Yiyang Su、Xiaoming Liu，NeurIPS 2023。见 [正式页面](https://papers.neurips.cc/paper_files/paper/2023/hash/dc81297c791bb989deade65c6bd8c1d8-Abstract-Conference.html)、[论文PDF](https://proceedings.neurips.cc/paper_files/paper/2023/file/dc81297c791bb989deade65c6bd8c1d8-Paper-Conference.pdf)、[作者代码](https://github.com/Zhiyuan-R/ChatGPT-Powered-Hierarchical-Comparisons-for-Image-Classification)。

论文通过CLIP特征递归组织类别并生成有区分度的层级描述，再融合沿层级的图文分数。本项目仅借鉴分层外观与显式比较的生成思路，没有复现递归知识树、遍历或树分类器；不能将本项目描述库写成该论文完整实现。

## 历史条件与当前库的区别

早期v3方案为概述20、细节60、对比40，每类120条，共12240条；它与最终v4宽覆盖100条提示不同。早期曾为pink primrose提供外部植物知识hint，将其解释为Oenothera speciosa，来源为 [北卡州立大学推广资料](https://plants.ces.ncsu.edu/plants/oenothera-speciosa/) 和 [得克萨斯大学植物资料](https://w3.biosci.utexas.edu/prc/K12/pages/Oenothera%20speciosa.html)。这是历史人工外部知识条件，最终v4使用原标签和空hint，没有沿用这些澄清。

历史目录 `flowers102_description_bank`、`flowers102_deepseek_v41_flash`、`flowers102_explicit_pairs_v3`、`flowers102_explicit_pairs_grounded_v3` 和 `flowers102_redone` 保留查证，不合并或覆盖。提示、数量和元数据处理历次不同，不能将旧响应冒充当前库；原始生成快照与请求记录保留在对应输出。

此前大写模型名返回model_not_found，实际可用ID确认是 `deepseek-v4.1-flash`。最终接口、私有配置、原始响应复用、请求和usage数量均在现行生成记录维护，不在这里重复安装/API命令。用户“不审查”约定继续有效，不筛选、事实纠正、去重、改写，也不因内容问题重新付费生成。
