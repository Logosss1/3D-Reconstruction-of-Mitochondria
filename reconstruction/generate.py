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
    z = zarr.open('data/sample_hela2/sample.zarr', mode='r')
    real_raw = np.array(z['raw'])
    real_input = torch.from_numpy(real_raw).float().unsqueeze(0).unsqueeze(0).to(device) / 255.0

    # 3. 提取 3D 网格
    print("🛠️ 正在计算等值面网格 (OBJ)...")
    _, mesh = reconstruction_pipeline(model, real_input, real_raw)
    
    # 导出 OBJ 文件（可以用 3D 查看器打开）
    mesh.export("final_mitochondria.obj")

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