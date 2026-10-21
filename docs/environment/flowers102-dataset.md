# Oxford Flowers-102 数据准备记录

日期：2026-10-20。根据“再换一个数据集”的要求，为后续描述选择实验增加 Flowers-102，保留 DTD。数据准备阶段没有修改算法、安装环境、提交或推送。此后已应用户要求完成全量 TDA + CuPL 和人工筛选对照，见 [实验结果](../experiments/flowers102-cupl-manual-results.md)。

## 选择原因及实验边界

花卉具有颜色、花瓣形状和排列等可视属性，可用来研究描述与目标图像之间的适配关系。其类别存在相似外观和类内变化，适合作为纹理数据集之外的补充，但数据集自身不能保证“筛选几条优于全部融合”。这一点仍应使用同一描述池下的全量、随机 K 条、适配 K 条对照验证，具体方法见 `semantic-adaptation-experiment-plan.md`。

[TDA 论文 Table 3](https://arxiv.org/html/2403.18293v1) 报告 ViT-B/16 在 Flower102 上为 71.42%，低于 Caltech101 的 94.24%。这些是论文参考值。后续本机配对实验的模板 TDA 为71.50%、官方 CuPL 为74.18%、人工筛选为74.10%（均为样本 Top-1）；已准备官方描述，尚未生成新版描述。

## 数据位置与校验

服务器目录：`/root/autodl-tmp/project_R/data/oxford_flowers`。

| 内容 | 结果 |
| --- | --- |
| 图片 | `jpg/`，8189 张、102 类，346809251 字节，约 331 MiB |
| 项目划分 | `split_zhou_OxfordFlowers.json` |
| train / val / test | 4093 / 1633 / 2463 张 |
| 附带原始标签 | `imagelabels.mat`，项目加载使用划分 JSON 中的标签和类别名 |
| 机器可读记录 | `preparation.json` |

图片来自 [Oxford 官方数据站点](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/)。原始包保留在 `/root/autodl-tmp/project_R/downloads/102flowers.tgz`，MD5 为 `52808999861908f626f3c1f4e79d11fa`，已与 [torchvision 0.20.1 的校验记录](https://github.com/pytorch/vision/blob/v0.20.1/torchvision/datasets/flowers102.py) 核对。

划分来自 [公开研究代码中的 Zhou 划分副本](https://github.com/linzhiqiu/cross_modal_adaptation/blob/main/splits/split_zhou_OxfordFlowers.json)，SHA256 为 `2f50c42fc1a17ca13c271cb4269b1347bdb43d2be672e34f74a4e4cbfbcca794`。保持当前类别编号和划分，不换成 torchvision 默认划分。

已验证压缩包完整性、全部图片的 PIL 结构检查、全部划分路径存在、划分内部无重复路径、三个划分无交集。原项目加载器已读取 2463 张测试图、102 类。结构检查不代表已经完整解码每张图片进行推理。

## 后续运行方式

直接复用既有独立环境、ViT-B/16 权重与原项目配置，无需安装依赖。以下是原模板 TDA 的运行命令；后续配对实验使用独立脚本和共同特征重放，具体见实验结果记录：

```bash
cd /root/autodl-tmp/project_R/RDSS
HOME=/root/autodl-tmp/project_R/runtime-home \
PYTHONNOUSERSITE=1 PYTHONPATH= \
WANDB_MODE=disabled CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 \
/root/autodl-tmp/project_R/env/bin/python -B tda_runner.py \
  --config /root/autodl-tmp/project_R/RDSS/configs \
  --datasets oxford_flowers \
  --data-root /root/autodl-tmp/project_R/data \
  --backbone 'ViT-B/16' \
  --wandb-log
```

正式执行前检查 GPU 当前占用，并把日志、配置和指标保存到新的结果目录。后续描述筛选应遵循既有实验记录，使用验证集确定参数，不依据测试集结果反复改写描述。
