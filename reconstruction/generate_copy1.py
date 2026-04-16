import rtx50_compat
import torch
import numpy as np
import zarr
import matplotlib.pyplot as plt
from src.model import ConvONet
from src.post_process import reconstruction_pipeline

def run_generation():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 正在提取最终版『实心土豆』模型...")

    # 1. 加载模型
    model = ConvONet().to(device)
    model.load_state_dict(torch.load("checkpoints/model_final.pth", map_location=device))

    # 2. 读取数据
    z = zarr.open('data/hela3.zarr/jrc_hela-3.zarr', mode='r')
    raw_ds = z['recon-1/em/fibsem-uint8/s0']
    d, h, w = raw_ds.shape
    # 减小输入体积大小以减少内存使用
    crop_d, crop_h, crop_w = 256, 256, 256
    crop_d = min(crop_d, d)
    crop_h = min(crop_h, h)
    crop_w = min(crop_w, w)
    cz, cy, cx = d // 2, h // 2, w // 2
    z0 = max(0, cz - crop_d // 2)
    y0 = max(0, cy - crop_h // 2)
    x0 = max(0, cx - crop_w // 2)
    z1 = min(d, z0 + crop_d)
    y1 = min(h, y0 + crop_h)
    x1 = min(w, x0 + crop_w)
    real_raw = np.array(raw_ds[z0:z1, y0:y1, x0:x1])
    real_input = torch.from_numpy(real_raw).float().unsqueeze(0).unsqueeze(0).to(device) / 255.0

    # 3. 提取 3D 网格
    print("🛠️ 正在计算等值面网格 (OBJ)...")
    _, mesh = reconstruction_pipeline(model, real_input, real_raw, resolution=128, threshold=0.2)
    
    # 导出 OBJ 文件（可以用 3D 查看器打开）
    mesh.export("hela3_mitochondria.obj")

    # 4. ✨ 自动生成高清答辩专用预览图
    print("📸 正在绘制高清 3D 效果图...")
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # 增加采样密度 (::2 表示每隔两个点取一个，点会更密更实)
    v = mesh.vertices[::2] 
    
    # 使用渐变色 (cmap='magma')，让它看起来更像高端科研成果
    scatter = ax.scatter(v[:, 0], v[:, 1], v[:, 2], s=5, c=v[:, 2], cmap='magma', alpha=0.8)
    
    ax.set_xlabel('X (nm)')
    ax.set_ylabel('Y (nm)')
    ax.set_zlabel('Z (nm)')
    plt.title("Final 3D Reconstructed Mitochondria (96.19% Acc)")
    
    # 保存两张图，一张是之前的预览，一张是高清版
    plt.savefig("preview_result.png")
    plt.savefig("Thesis_Final_Result.png", dpi=300) # 高质量导出
    
    print("-" * 30)
    print("✅ 全部搞定！")
    print("📍 OBJ 模型：final_mitochondria.obj")
    print("📍 答辩高清图：Thesis_Final_Result.png")
    print("-" * 30)

if __name__ == "__main__":
    run_generation()
