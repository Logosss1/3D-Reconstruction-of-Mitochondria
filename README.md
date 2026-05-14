## Mito3D Reconstruction Thesis

本项目是一个面向毕业论文展示的 **线粒体 3D 重建与可视化** 小型工程。  
它通过隐式表示网络（ConvONet）从 3D EM 体数据中重建线粒体的三维形状，并生成论文 / 答辩可用的图片和动画。
![image](https://github.com/Logosss1/3D-Reconstruction-of-Mitochondria/blob/main/reconstruction_rotation_showcase_gt.gif)

---

## 1. 功能概览

- **仿真数据生成**：构造类似 OpenOrganelle jrc_hela-2 的 3D 线粒体数据（raw + label，Zarr 格式）。
- **隐式网络训练**：基于 ConvONet 学习体素占据函数 \( f(x,y,z) \)，判断空间点是否在线粒体内部。
- **3D 网格重建**：在 3D 空间密集采样，利用模型输出 + Marching Cubes 提取等值面，导出 `.obj`。
- **可视化**：
  - 高分辨率 3D 散点图（PNG），可直接插入论文 / 答辩 PPT。
  - 旋转 GIF 动画（线粒体 3D 展示），便于演示效果。
- **定量指标展示（示例）**：IoU、Chamfer Distance 的示意输出，用于论文“实验结果”章节说明。

---

## 2. 环境配置

建议使用 Conda 创建独立环境（仓库中已有 `environment.yml`）：

```bash
# 1. 创建环境
conda env create -f environment.yml

# 2. 激活环境（名称与 environment.yml 一致）
conda activate mito3d_env

# 3. 确认 PyTorch / CUDA 是否可用（可选）
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

### Windows + NVIDIA GPU

若 `torch.cuda.is_available()` 为 `False`，常见原因是装到了 **CPU 版 PyTorch**，或 **pip 安装的 numpy 与 conda 混用**（会出现 `numpy` 变成 namespace、`_multiarray_umath` 报错）。

在已激活 `mito3d_env` 后，用 conda 统一安装 **CUDA 11.8** 构建（与当前项目 PyTorch 2.0.1 一致）：

```bash
conda install -y pytorch=2.0.1=py3.9_cuda11.8_cudnn8_0 torchvision=0.15.2=py39_cu118 pytorch-cuda=11.8 numpy=1.26.4 --override-channels -c pytorch -c nvidia -c defaults
```

验证：

```bash
python -c "import numpy, torch; print(numpy.__version__); print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

> 如未使用 `environment.yml`，需确保安装：
> - Python 3.8+
> - PyTorch + CUDA（可选）
> - numpy, zarr, scikit-image, trimesh, matplotlib, pillow 等依赖。

### Hela2：直接用 `hela2.zarr`（crop 标注 + 全局 EM）

训练默认 **不再** 导出中间 `sample.zarr`，而是从 `data/hela2.zarr/jrc_hela-2.zarr/recon-1/` 读取：

- **标注**：`labels/groundtruth/crop{CROP}/mito/s0`
- **背景 EM**：`em/fibsem-uint8/s0`（体素比 label 粗，脚本会按 OME-Zarr 的 `scale` 把 EM **重采样到与 label 相同的三维形状**）

默认 **`--crop_id 94`**。训练仍用随机 **`encoder_spatial`³** 子块（默认 128）；也可用合并 Zarr：`train.py --zarr data/sample_hela2/sample.zarr`。

```bash
# 默认 crop94 + hela2.zarr
python train.py --data_root data --dataset jrc_hela-2 --crop_id 94 --encoder_spatial 128 --epochs 300
python run_hela2_mito_training.py --crop_id 94

# 推理（同域 EM）
python generate.py --data_root data --crop_id 94 --infer_max_spatial 128 --mc_resolution 128
```

---

## 3. 项目结构

```text
Mito3D_Reconstruction_Thesis/
├─ environment.yml              # 推荐的 Conda 环境配置
├─ train.py                     # 训练 ConvONet（hela2 crop 或 --zarr）
├─ generate.py                  # 重建 3D 网格 + 生成高清结果图
├─ validate_crop.py             # 验证 crop 推理结果
├─ hela2_mito_pipeline.py       # Hela2 线粒体端到端流水线
├─ run_hela2_mito_training.py   # 一键：hela2.zarr 上训练
├─ export_hela2_np_s0_mito_masked.py  # 导出 Hela2/Hela3 掩码数据
├─ data/                        # 训练/验证数据（Zarr 格式）
├─ src/                         # 共享模块（模型、数据集、后处理等）
├─ scripts/                     # 训练/推理辅助脚本
├─ checkpoints/                 # 模型权重
├─ result/                      # 当前分支推理结果
├─ reconstruction/              # 导出重建分支（原 other_version）
│  ├─ src/                      #   重建分支源码
│  ├─ result/                   #   重建结果（含 hela23_all_crops）
│  └─ ...
├─ others/                      # 非核心辅助文件
│  ├─ scripts/                  #   prepare_data.py, evaluate.py, make_video.py 等
│  ├─ docs/                     #   旧版 README 等
│  ├─ data/                     #   测试数据（_zv2_test 等）
│  └─ misc/                     #   杂项
└─ thesis/                      # 论文 LaTeX 源码
```

---

## 4. 使用流程

### 4.1 生成仿真数据

```bash
python others/scripts/prepare_data.py
```

- **输出**：
  - 在 `data/sample_hela2/` 目录下生成 `sample.zarr`，包含：
    - `raw`：模拟 EM 灰度体数据
    - `label`：线粒体体素掩码

### 4.2 训练隐式网络

```bash
python train.py
```

- **主要逻辑**：
  - 从 `sample.zarr` 读取 `raw` 和 `label`。
  - 使用 ConvONet 对 `(raw, 3D 坐标)` 学习 occupancy。
  - 采用混合采样：一半正样本（label>0 的点），一半随机背景。
  - 损失函数：`BCEWithLogitsLoss + Dice Loss`。
- **输出**：
  - `checkpoints/model_final.pth` —— 训练好的模型权重。

### 4.3 重建 3D 网格 & 生成图片

```bash
python generate.py
```

- **主要逻辑**：
  - 加载 `model_final.pth`。
  - 在整个 3D 空间上密集采样点，利用 ConvONet 计算每点概率。
  - 使用 Marching Cubes 提取等值面，得到 `trimesh.Trimesh` 网格。
  - 绘制高质量 3D 散点图，保存 PNG。
- **输出**：
  - `final_mitochondria.obj` —— 3D 模型，可用 MeshLab / Blender 等查看。
  - `preview_result.png` —— 结果预览图。
  - `Thesis_Final_Result.png` —— 用于论文 / 答辩的高清图。

### 4.4 定量指标展示（示例）

```bash
python others/scripts/evaluate.py
```

- 当前脚本中 IoU、Chamfer Distance 为**模拟数值**，用于展示论文中可报告的指标格式。
- 可根据实际需要替换为真实计算逻辑（例如基于点采样的 IoU / Chamfer Distance）。

### 4.5 生成旋转 GIF 动画

```bash
python others/scripts/make_video.py
```

- **主要逻辑**：
  - 加载 `final_mitochondria.obj`。
  - 以 3D 散点形式绘制点云。
  - 使用 `matplotlib.animation.FuncAnimation` 逐帧旋转视角。
- **输出**：
  - `Mito3D_Rotation_Showcase.gif` —— 旋转展示动画，可直接插入 PPT。

---

## 5. 代码核心简述

- **隐式表示 (Implicit Representation)**  
  使用函数 \( f_\theta(x, y, z) \rightarrow [0,1] \) 表示“该点是否在目标内部”，而不是直接预测体素网格。
- **条件输入 (Conditioning on EM Volume)**  
  - 用 3D 卷积对整块 EM 体数据编码，得到特征图。
  - 对每个查询点，用 `grid_sample` 在特征图上抽取局部特征，再与点坐标拼接送入 MLP。
- **表面提取 (Marching Cubes)**  
  - 在连续空间上评估 \( f_\theta \)，构建概率体。
  - 使用 Marching Cubes 在给定阈值下提取等值面，得到 3D 网格。
- **可视化与展示**  
  - 通过高分辨率 PNG、OBJ 和 GIF，将 3D 重建结果以多种形式呈现，方便论文和答辩展示。

---

## 6. 注意事项

- 若使用真实 OpenOrganelle 数据，可参考 `src/dataset.py` 中的 `MitoDataset` 类进行扩展，将 `train.py` 换成标准 DataLoader 训练流程。
- 若在无 GPU 的环境下运行，训练可能较慢；可以降低：
  - 训练轮数（如从 300 降到 50）
  - 每次采样点数（如从 10000 降到 2000）
- 若在生成阶段 Marching Cubes 提取不到表面，可适当调整：
  - `src/post_process.py` 中的 `resolution`（如 64, 128, 160）
  - 阈值策略（`threshold` 或基于 `max_p` 的比例）。

---

## 7. 致谢（可选）

- 仿真数据与网络结构设计参考了公开的 3D 细胞超微结构数据集（如 OpenOrganelle）及隐式表示相关工作。  
- 本代码主要用于教学与毕业设计示例，可根据需要在此基础上扩展为更完整的科研项目。
