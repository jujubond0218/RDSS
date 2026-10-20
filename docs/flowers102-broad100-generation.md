# Flowers-102 每类100条宽覆盖描述库

2026-10-20，用户明确授权生成全部102类。目标是为RDSS提供未经当前样本流预筛选的大候选池，而不是生成经过人工净化的类别原型。

输入为 `configs/flowers102.classes.json`：102个原类别名、空hint。不使用上一轮的植物知识澄清，不读取图片、测试标签、特征、预测或混淆矩阵。混淆类别由LLM根据全部类别名提出；不能称为实测CLIP混淆类别。

每类20条概述、60条细节、20条显式类别对比，共100条；总计10200条。提示版本 `layered-visual-bank-v4-broad-100`。覆盖常见和少见颜色、品种、形态、部位、生长阶段、视角、遮挡、光照和场景，容许语义重叠及不适配具体图像的描述。对比句要求在正文点名两类，并描述同一维度上的差异，比较对象均匀分配。

按用户要求，不做内容审查、事实纠正、筛选、去重或改写，不以文本内容问题触发重生成；原文、自然产生的不适配、冗余及潜在模型错误全部保留。没有主动编造错误以保证全量融合下降。该库不能被描述为经事实核查的知识库。

模型仍为 `deepseek-v4.1-flash`，使用已有私有配置与环境，未安装依赖。四个类别并发生成，一类6次请求（规划1、概述1、细节3、对比1），正常共612次。首次运行HTTP尝试上限650，网络状态码重试计入上限。原始响应、usage、任务断点及代码快照保留；不保存密钥。

```bash
cd /root/autodl-tmp/project_R/experiments/offline_description_bank
/root/autodl-tmp/project_R/env/bin/python -B scripts/generate_description_bank.py \
  --classes configs/flowers102.classes.json \
  --overview 20 --detail 60 --contrast 20 --workers 4 --max-requests 650 \
  --output outputs/flowers102_broad100_20261020
```

本次已全部完成，生成进程已退出。PID最初为603522，最后补齐PID604710。日志为 `description_bank_broad100_20261020.log`、`description_bank_broad100_repair_20261020.log` 和 `description_bank_broad100_finish_20261020.log`，均在 `/root/autodl-tmp/project_R/logs`，输出目录中也保存副本。服务器最终目录 `/root/autodl-tmp/project_R/experiments/offline_description_bank/outputs/flowers102_broad100_20261020`，本地副本 `outputs/flowers102_broad100_20261020`。

`prompts_all.json`为类别到100条原文的映射；三个 `prompts_<level>.json`为分层原文；`bank.json`保留描述ID、层级与比较对象。`status.json`记录完成类数及条数，`failures.json`记录无法完成的请求任务。生成中的导出仅含完整类别；只有102类全部完成才是本次全量交付。

首轮运行仍含旧的规划元数据限制，导致部分类别被中断，原始响应全部保留。随后删除比较对象是否属于数据集、数量与属性轴数量的限制，不审查这些内容；补齐流程从相同请求体的已付费响应直接恢复。先为已结束的前72类补齐，再合并其余类别断点完成全量；不重复生成已经取得的有效JSON内容。因为源码解析部分有变更，首轮清单与代码快照单独保留，生成提示未在中途改变。

当前修正版在本地通过12项离线测试，覆盖原始响应复用不再次付费、导出、断点、请求上限、额外元数据保留、比较对象分配与密钥脱敏。首轮启动前两端通过11项测试。不以程序测试声称文本事实正确，未运行本描述库的分类准确率实验。

最终两端修正版均通过12项离线测试。全库仅核对结构与数量：102类全部完整，各类100条，三个层级分别20/60/20，失败列表为空。共613次HTTP请求及613份返回，比正常612次多1次无效JSON返回；已返回的有效内容复用，没有因名称或属性元数据问题重复付费生成。提供方返回的usage合计输入671228、输出554848、总计1226076 tokens，不据此猜测货币费用。细目见输出中的 `generation_summary.json`。

首轮旧检查及其输出另存服务器 `outputs/flowers102_broad100_pass1_20261020`；最终库保留首轮清单与源码快照，说明相同提示下的响应复用来源。最终压缩包 `/root/autodl-tmp/project_R/downloads/flowers102_broad100_20261020.zip`，SHA256为 `8baadafa85f03b876650736fd07bf6f11df50d9c277cf70d7066376903013fcf`。
