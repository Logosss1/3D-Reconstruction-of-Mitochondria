[English](README.md) | **简体中文**

# Simplified Convolutional Occupancy Network Based Mitochondria 3D Reconstruction

![image](https://github.com/Logosss1/3D-Reconstruction-of-Mitochondria/blob/main/reconstruction_rotation_showcase_gt.gif)
<img width="600" alt="线粒体重建效果图" src="https://github.com/Logosss1/3D-Reconstruction-of-Mitochondria/blob/main/hela3_mito_gt_montage.png" />


---

## 项目概述

本项目面向本科毕业论文与科研展示场景，研究对象为**线粒体的三维重建**。项目基于简化卷积占据网络（simplified convolutional occupancy network, ConvONet-style pipeline），从 FIB-SEM / OpenOrganelle 体系下的三维电子显微镜体数据中学习空间占据函数，并通过 Marching Cubes 从预测概率场中提取显式网格，用于结果分析、论文写作和可视化展示。

与直接体素分割不同，本项目将三维形状恢复表述为对查询点占据概率的预测问题。该设计使模型能够在连续空间中估计目标结构，并最终生成可检查、可导出、可演示的三维网格结果。

## 研究目标

本项目对应的论文工作聚焦于以下问题：

1. 简化卷积占据网络是否能够在线粒体 crop 上生成连贯、可检查的三维网格；
2. 不同数据域，尤其是 HeLa2 与 HeLa3，是否会导致显著不同的重建表现；
3. 体素重叠指标与表面距离指标是否会在同一设计改动下呈现不一致趋势。

因此，本仓库既是代码实现，也是论文实验流程的工程化载体。

## 方法流程

<p align="center">
  <img src="thesis/images/prototype_reconstruction.png" alt="Prototype reconstruction" width="31%" />
  <img src="thesis/images/representative_mito_gt.png" alt="Representative mito ground truth" width="31%" />
  <img src="thesis/images/segmentation_result_comparison.png" alt="Segmentation comparison" width="31%" />
</p>

整体流程可概括为四个阶段：

1. **数据读取与裁剪**  
   从 OpenOrganelle / crop export Zarr 中读取原始 EM 体数据与线粒体标签，构造训练与推理所需的三维 crop。

2. **占据学习**  
   使用轻量 3D 卷积编码器提取局部体特征；对归一化查询坐标进行采样，并通过 MLP 预测每个点是否属于线粒体内部。

3. **显式表面重建**  
   在三维空间上密集评估占据概率，形成概率体后利用 Marching Cubes 提取等值面，得到 `.obj` 网格。

4. **定量验证与可视化**  
   使用 mesh-vs-label 评估流程计算 voxel IoU、voxel Dice 与 Chamfer 类指标，同时导出 PNG、OBJ 与 GIF 以支持论文和答辩展示。

## 当前仓库中的真实主流程

> 当前仓库的主实验流程以 [hela2_mito_pipeline.py](hela2_mito_pipeline.py) 为入口；旧版以 `sample.zarr` 和 `evaluate.py` 为核心的演示流程仍保留在仓库中，但**不再代表当前论文主线**。

核心脚本如下：

- [hela2_mito_pipeline.py](hela2_mito_pipeline.py)：主流程调度入口，支持 `joint`、`per-crop`、`infer-all`
- [train.py](train.py)：训练 ConvONet 风格占据预测模型
- [generate.py](generate.py)：进行推理、重建概率体并提取三维网格
- [validate_crop.py](validate_crop.py)：对单个 crop 的 mesh 与标签进行定量验证
- [export_hela2_np_s0_mito_masked.py](export_hela2_np_s0_mito_masked.py)：导出可用于训练的 crop Zarr 数据

## 仓库结构

```text
Mito3D_Reconstruction_Thesis/
├─ environment.yml
├─ hela2_mito_pipeline.py
├─ train.py
├─ generate.py
├─ validate_crop.py
├─ export_hela2_np_s0_mito_masked.py
├─ run_hela2_mito_training.py
├─ data/
├─ src/
├─ scripts/
├─ checkpoints/
├─ reconstruction/
├─ thesis/
├─ Thesis_Final_Result.png
├─ preview_result.png
└─ Mito3D_Rotation_Showcase.gif
```

## 环境配置

建议优先使用仓库内的 [environment.yml](environment.yml) 创建 Conda 环境：

```bash
conda env create -f environment.yml
conda activate mito3d_env
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

如果在 Windows + NVIDIA GPU 环境中遇到 CUDA 不可用，可检查 PyTorch 是否被安装为 CPU 版本，或是否发生了 `pip`/`conda` 混装导致的 `numpy` 兼容问题。

## 快速开始

### 1. Joint training

```bash
python hela2_mito_pipeline.py joint --epochs 300
```

该模式在指定数据域上训练一个联合模型。默认情况下，HeLa2 与 HeLa3 会使用各自的默认 checkpoint 命名策略。

### 2. Per-crop training

```bash
python hela2_mito_pipeline.py per-crop --only-crop 9 --epochs 300
```

该模式针对单个 crop 单独训练、单独生成网格并可进一步执行验证，适合论文中的逐视野分析。

### 3. Batch inference on a trained checkpoint

```bash
python hela2_mito_pipeline.py infer-all --checkpoint checkpoints/model_hela2_all_mito.pth --validate
```

该模式会对多个 crop 批量执行推理，并在启用 `--validate` 时输出逐 crop 评估结果与汇总统计。

### 4. 单脚本入口（按需）

```bash
python train.py --data_root data --dataset jrc_hela-2 --crop_id 94 --epochs 300
python generate.py --data_root data --dataset jrc_hela-2 --crop_id 94 --checkpoint checkpoints/model_final.pth
python validate_crop.py --data_root data --dataset jrc_hela-2 --crop_id 94 --mesh reconstruction/final_mitochondria.obj
```

## 数据、许可与输出

仓库中的原创代码采用 MIT License。数据集、派生文件、第三方依赖和项目来源边界分别记录在 [`DATA_SOURCES.md`](DATA_SOURCES.md)、[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) 与 [`PROVENANCE.md`](PROVENANCE.md) 中。

### 输入数据

项目当前主要围绕以下两类数据入口组织：

- **OpenOrganelle / OME-Zarr 路径**：直接读取 HeLa 数据及其 crop 标签；
- **crop export Zarr 路径**：将线粒体相关 crop 导出为更直接的训练/推理输入。

### 典型输出

- `checkpoints/*.pth`：训练得到的模型权重；
- `reconstruction/**`：推理结果、拼图、指标统计与验证输出；
- `.obj`：三维网格模型；
- `.png`：论文和汇报可直接使用的高分辨率图像；
- `.gif`：旋转展示动画。

## 实验可视化

<p align="center">
  <img src="thesis/images/hela2_joint_montage.png" alt="HeLa2 joint montage" width="48%" />
  <img src="thesis/images/hela3_joint_montage.png" alt="HeLa3 joint montage" width="48%" />
</p>

<p align="center">
  <img src="thesis/images/training_loss_comparison.png" alt="Training loss comparison" width="48%" />
  <img src="thesis/images/validation_loss_comparison.png" alt="Validation loss comparison" width="48%" />
</p>

这些图像来自当前论文与实验输出，分别对应：

- 代表性重建结果；
- HeLa2 / HeLa3 联合推理拼图；
- 训练与验证损失曲线；
- 重建结果与标签之间的视觉比较。

## 与论文的关系

本仓库服务于 **Yuhao Lu** 在 Wenzhou-Kean University 完成的本科论文 **“Simplified Convolutional Occupancy Network Based 3D Reconstruction for Mitochondria”** 的实验实现与结果展示，导师为 **Gao Zhiqiang & Xie Zhiwu**，论文时间为 2026 年 4 月。论文仅说明为学校图书馆存档，不在此声称存在公开 DOI 或在线出版页面。论文摘要中强调：

- 模型采用轻量 3D 卷积编码器 + 点查询 MLP；
- 训练目标使用 BCE 与 Dice loss 以缓解前景稀疏问题；
- 推理阶段将稠密占据概率体转化为显式网格；
- HeLa2 与 HeLa3 数据域在结果上呈现明显差异；
- overlap 指标与 surface-distance 指标可能在同一设计改动下出现分歧。


## 局限性说明

- 当前仓库中的部分旧脚本与旧说明仍然保留，可能反映项目早期 demo 阶段；
- 不同数据域的结果差异较大，尤其是更困难视野上的表现仍有限；
- 某些可视化脚本更偏论文展示用途，而非大规模自动化生产流程；
- 指标解释需要结合具体 crop、具体推理模式与后处理设置，不能孤立理解单一数值。

## Citation

如果你在学术写作或展示中使用本仓库，请优先引用对应论文版本，并在必要时说明所使用的数据域、训练模式与验证设置。机器可读的引用信息见 [`CITATION.cff`](CITATION.cff)。

> Lu, Yuhao. *Simplified Convolutional Occupancy Network Based 3D Reconstruction for Mitochondria*. Bachelor's thesis, Wenzhou-Kean University, 2026. Supervisor: Xie Zhiwu.
