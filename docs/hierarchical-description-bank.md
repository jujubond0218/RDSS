# 分层描述库生成：论文调研与API脚本

日期：2026-10-20。当前交付目录为 `outputs/flowers102_redone`。用户最新要求“不审查”：停止内容审阅、筛选或改写，保存生成模型原文，不因内容问题重生成。带名称澄清版本已返回的付费响应直接复用，额外顶层元数据保留，补齐剩余请求。尚未生成全部102类或测试新版准确率。

## 找到的论文

用户提到的“gpt-power”很可能是 **ChatGPT-Powered Hierarchical Comparisons for Image Classification**，作者 Zhiyuan Ren、Yiyang Su、Xiaoming Liu，NeurIPS 2023。论文见[正式会议页面](https://papers.neurips.cc/paper_files/paper/2023/hash/dc81297c791bb989deade65c6bd8c1d8-Abstract-Conference.html)、[论文PDF](https://proceedings.neurips.cc/paper_files/paper/2023/file/dc81297c791bb989deade65c6bd8c1d8-Paper-Conference.pdf)，代码见[作者仓库](https://github.com/Zhiyuan-R/ChatGPT-Powered-Hierarchical-Comparisons-for-Image-Classification)。

论文第3.2节从初始类别描述的CLIP特征出发，用聚类递归组织类别，再在各组内生成有区分度的描述。较大组先总结组的共性，再针对该共性描述目标类的区别；较小组直接比较目标类和其他类。第3.3节再融合沿层级得到的图文相似度，并非把所有层级文本无条件平均。

这与本项目的“概貌、细节、混淆类别对比”有直接联系，但本次脚本仅借鉴分层与差异比较的生成思路：没有复现论文的递归CLIP聚类、知识树遍历或分数融合，也没有把原TDA分类器改成树分类器。生成完成后仍可保持TDA流程，比较同一描述池的全量平均、随机同条数子集及适配选择。

## 本次实现

本地脚本为 `scripts/generate_description_bank.py`。HTTP使用现有项目已安装的 `requests`，不引入OpenAI SDK或新依赖，使用Chat Completions请求格式。服务器副本位于 `/root/autodl-tmp/project_R/experiments/offline_description_bank`，解释器仍为 `/root/autodl-tmp/project_R/env/bin/python`。

接口为 `https://api.aijws.com/v1/chat/completions`。用户指定的模型显示名称为 `DeepSeek-V4.1-Flash`；首次使用该大小写字符串返回404、`model_not_found`。随后读取 `/v1/models` 返回200，可用ID为 `deepseek-v4-pro` 与 **`deepseek-v4.1-flash`**。两端配置已改为后者，真实规划与描述请求均已成功，返回模型同为 `deepseek-v4.1-flash`。没有更换到其他模型。站点价格、credit字段单位及计费限额尚未核实，不根据字段名猜测费用。

`configs/description_api.example.json` 是公开模板，`configs/description_api.local.json` 是私有配置，已加入 `.gitignore`，输出目录 `outputs/` 也被忽略。用户已明确授权填写密钥并调用生成，密钥已写入本地和服务器私有配置，未写入文档或生成记录。服务器私有配置权限为600。

`configs/flowers102.classes.json` 包含原项目的102个类别名和空 `hint`，没有复制CuPL描述或上一轮人工子集。每类可手动填写 `hint` 说明名称含义；脚本会把该信息记录在生成清单中，不读取图片、测试标签、特征、预测或准确率。此前发现的watercress、ball moss命名歧义没有被暗中改名，需在生成前审阅，必要时明确填写hint并记录依据。若hint来自目标域验证图，应如实记录这种数据访问。

本次另存 `configs/flowers102.grounded.classes.json`：保留全部102个原标签，仅为首类pink primrose填写有来源的hint，将其明确为Oenothera speciosa，而不是粉色Primula。四瓣浅杯形花、颜色与脉纹依据[北卡州立大学推广资料](https://plants.ces.ncsu.edu/plants/oenothera-speciosa/)；八枚雄蕊、四裂柱头等结构依据[得克萨斯大学植物资料](https://w3.biosci.utexas.edu/prc/K12/pages/Oenothera%20speciosa.html)。这是人工提供外部植物知识的生成条件，应记录在方法中，不称为纯类别名输入或混淆矩阵驱动生成。未使用测试图片、标签或预测挑选比较对象。

每类先调用一次API规划宽泛上位概念、2～5个可视属性维度以及最多4个候选混淆类别。候选来自同一数据集类别名，由语言模型基于名称和知识提出，不是根据测试集错误率选取，也不是已经测得的CLIP混淆邻居。

| 层级 | 默认每类条数 | 主要内容 |
| --- | ---: | --- |
| overview | 20 | 类别整体、轮廓和主要外观 |
| detail | 60 | 部位、形状、颜色、纹理、排列、视角及合理变体 |
| contrast | 40 | 正文点名目标类和比较类，明确说明同一外观维度上的两者差异 |

默认每类120条，共12240条。每次生成最多20条，一类需1个规划请求和6个描述请求，全量共714个请求，不含HTTP重试。`--overview`、`--detail`、`--contrast`可调整数量。v3逐句均匀指定比较对象，要求对比句以目标类为主语，在正文中点名两类、写出目标特征与对方在同一维度上的不同特征。`compare_to`仅用于审计，不代替正文对比。每句话保留层级、属性、适用范围及对比对象。

生成提示要求真实、可见且多样的描述，不为填数量制造错误。合理颜色、姿态与部位变体可以自然地产生对当前图像不适用的候选，但并不保证全量融合一定下降。不能以制造更好看的主结果为目的，暗中将已知错误句混入常规基线。若研究人为语义噪声，应建立另一个明确标记噪声类型、比例和注入规则的压力测试，不冒充自然描述库。本次脚本没有实现人工错误注入。

## 使用方法

在服务器上编辑 `configs/description_api.local.json`，只需在 `api_key` 填写自己的密钥；不需要把密钥发到对话中。先在不联网的dry-run中核对规模，再只生成一类检查中转站兼容性和输出质量。

```bash
cd /root/autodl-tmp/project_R/experiments/offline_description_bank
/root/autodl-tmp/project_R/env/bin/python -B scripts/generate_description_bank.py --classes configs/flowers102.grounded.classes.json --dry-run --output outputs/flowers102_redone
/root/autodl-tmp/project_R/env/bin/python -B scripts/generate_description_bank.py --classes configs/flowers102.grounded.classes.json --limit 1 --max-requests 7 --output outputs/flowers102_redone
```

样例配置默认温度0.8、最大输出4096 tokens、请求超时120秒。上面的7次是HTTP尝试上限，若发生重试也计数；达到上限退出码为2，已经成功的请求保留。真正使用前需按站点价格确认费用，本次没有报价或声称费用可控的依据。

以下仅为用户明确授权全量生成之后的命令，不表示当前已启动或已获授权。相同配置、相同输出目录会自动跳过已完成请求：

```bash
/root/autodl-tmp/project_R/env/bin/python -B scripts/generate_description_bank.py --classes configs/flowers102.grounded.classes.json --max-requests 1000 --output outputs/flowers102_redone
```

旧目录 `outputs/flowers102_description_bank` 保存错误模型ID诊断，`outputs/flowers102_deepseek_v41_flash` 保存v1原始请求与120条未过滤输出，`outputs/flowers102_explicit_pairs_v3`保存v3首轮存在类别身份错误的输出，均保留供核对。本次修正版另存 `outputs/flowers102_explicit_pairs_grounded_v3`。改变数量、提示代码、模型、温度或类别信息后需使用新目录，脚本拒绝混合不同设置的断点。密钥不计入公开配置哈希。

未完成的HTTP超时请求不自动重试，因为服务器可能已经完成并计费；429和部分5xx最多尝试3次，每次都计入请求次数并保存响应。401、403、404、重定向、结构无效、截断响应或非法JSON直接停止。原始响应保留供排查，不自动换模型、补造句子或静默跳过错误。重新运行会重试未成功保存的任务，可能再次计费。`max-requests`是请求尝试数上限，不是货币费用上限。

## 输出与实验接口

| 文件 | 内容 |
| --- | --- |
| `prompts_all.json` | 类别到全部文本列表，可用于类内平均原型 |
| `prompts_overview.json` / `prompts_detail.json` / `prompts_contrast.json` | 按层级导出的文本列表 |
| `bank.json` | 每句ID、层级、属性、scope、compare_to及类别规划 |
| `jobs/`、`raw/` | 成功请求断点、完整请求体、原始响应、状态和返回usage |
| `manifest.json`、`generator_snapshot.py`、`status.json` | 设置、类别清单、代码快照、完成状态和重复数量 |

原始日志不保存Authorization头，响应若意外回显密钥也会脱敏。完全重复句只统计、不暗中删除，避免改变“全量融合”的含义。导出仅包含全部层级生成完毕的类别，必须确认 `status.json` 的 `complete` 为true后才能做全数据集比较；单类试跑不是完整描述库。

当前只检查导出所需的JSON结构、数量与字段类型，容许额外元数据，不按字数或比较类名拒收文本。按用户要求已删除 `content_audit.json` 导出，后续不进行内容筛选、改写或事实审查。本轮未做CLIP长度检查或准确率测试；`scope`与`compare_to`为生成模型元数据。

建议固定一次生成结果，比较TDA模板、TDA+自生成全量、随机匹配条数、只按相关性选择，以及相关性与区分度联合选择；CuPL官方池可保留为另一份来源对照。所有选择方法共享冻结候选池、同一视觉流和TDA配置。开发使用验证集，已有Flowers测试诊断已被查看，后续同测试集结果按探索性复测记录。

## 验证范围

修正版本地通过11项无联网测试，包含额外元数据保留、断点续跑、导出、请求数上限、密钥脱敏、HTTP错误及明确对比提示和比较对象均匀分配。原始请求与响应保留供查证；没有以测试通过声称文本内容正确。
