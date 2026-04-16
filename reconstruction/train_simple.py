import os

# 设置环境变量以启用CUDA兼容性
os.environ['CUDA_DISABLE_COMPATIBILITY_CHECK'] = '1'
os.environ['TORCH_USE_CUDA_DSA'] = '1'

# 现在导入rtx50_compat和torch
import rtx50_compat
import torch
import torch.nn as nn
import torch.optim as optim
import zarr
import numpy as np
import os
from src.model import ConvONet

def dice_loss(pred, target):
    pred = torch.sigmoid(pred)
    smooth = 1e-5
    intersect = (pred * target).sum()
    return 1 - (2. * intersect + smooth) / (pred.sum() + target.sum() + smooth)

def train():
    # 使用GPU
    device = torch.device('cuda')
    print(f"🚀 开始『实心化』训练...")
    print(f"使用设备: {device}")
    print(f"✅ GPU设备: {torch.cuda.get_device_name(0)}")
    print("✅ CUDA能力: {}".format(torch.cuda.get_device_capability(0)))
    
    # 测试GPU基本功能
    print("\n测试GPU基本功能...")
    try:
        test_tensor = torch.randn(1, 1, 64, 64, 64)
        test_tensor_gpu = test_tensor.to(device)
        print('✅ 大张量成功移动到GPU')
    except Exception as e:
        print(f'❌ GPU测试失败: {e}')
        return
    
    # 使用hela2的高分辨率数据
    print("\n📁 加载 hela2 高分辨率数据...")
    z = zarr.open("data/hela2.zarr/jrc_hela-2.zarr", mode='r')
    
    # 读取原始图像 (高分辨率)
    raw_ds = z['recon-1/em/fibsem-uint8/s0']
    d, h, w = raw_ds.shape
    print(f"原始数据形状: {d}x{h}x{w}")
    
    # 裁剪一个小区域以减少内存使用
    crop_size = 64
    cz, cy, cx = d // 2, h // 2, w // 2
    z0 = max(0, cz - crop_size // 2)
    y0 = max(0, cy - crop_size // 2)
    x0 = max(0, cx - crop_size // 2)
    z1 = min(d, z0 + crop_size)
    y1 = min(h, y0 + crop_size)
    x1 = min(w, x0 + crop_size)
    
    print(f"裁剪区域: Z[{z0}:{z1}], Y[{y0}:{y1}], X[{x0}:{x1}]")
    
    # 读取和处理数据
    raw_crop = np.array(raw_ds[z0:z1, y0:y1, x0:x1])
    print(f"原始裁剪数据形状: {raw_crop.shape}")
    
    raw = torch.from_numpy(raw_crop).float().unsqueeze(0).unsqueeze(0)
    print(f"张量形状: {raw.shape}")
    
    # 移动到GPU
    raw = raw.to(device) / 255.0
    print("✅ 原始数据成功移动到GPU")
    
    # 读取标注数据
    label_ds = z['recon-1/labels/masks/foreground']
    label_crop = np.array(label_ds[z0:z1, y0:y1, x0:x1])
    print(f"标注裁剪数据形状: {label_crop.shape}")
    
    label = torch.from_numpy(label_crop).float().unsqueeze(0).unsqueeze(0)
    print(f"标注张量形状: {label.shape}")
    
    # 移动到GPU
    label = label.to(device)
    print("✅ 标注数据成功移动到GPU")
    
    # 初始化模型
    model = ConvONet().to(device)
    print("✅ 模型成功移动到GPU")
    
    optimizer = optim.Adam(model.parameters(), lr=5e-4)
    
    # 预提取有“肉”的坐标，防止空盒子
    pos_indices = torch.from_numpy(np.argwhere(label_crop > 0)).float().to(device)
    print(f"✅ 找到 {len(pos_indices)} 个实心像素，开始训练...")
    
    # 创建检查点目录
    os.makedirs('checkpoints', exist_ok=True)
    
    # 开始训练
    print("\n🚀 开始训练...")
    for epoch in range(1, 11):  # 只训练几个epoch测试
        optimizer.zero_grad()
        
        # 混合采样：一半肉，一半背景
        num_points = 10000
        if len(pos_indices) > 0:
            # 确保有足够的正样本
            num_pos = min(num_points // 2, len(pos_indices))
            idx = torch.randint(0, len(pos_indices), (num_pos,))
            p_pos = pos_indices[idx] / crop_size
            # 生成负样本
            p_rand = torch.rand(num_points - num_pos, 3).to(device)
            points = torch.cat([p_pos, p_rand], dim=0).unsqueeze(0)
        else:
            # 如果没有正样本，只使用随机点
            points = torch.rand(1, num_points, 3).to(device)
        
        # 提取真实标签
        d_crop, h_crop, w_crop = label.shape[-3:]
        iz = torch.clamp((points[0,:,0]*d_crop).long(), 0, d_crop-1)
        ih = torch.clamp((points[0,:,1]*h_crop).long(), 0, h_crop-1)
        iw = torch.clamp((points[0,:,2]*w_crop).long(), 0, w_crop-1)
        target = label[0, 0, iz, ih, iw].unsqueeze(0)

        # 前向传播
        out = model(raw, points)
        loss = nn.BCEWithLogitsLoss()(out, target) + dice_loss(out, target)
        
        # 反向传播
        loss.backward()
        optimizer.step()

        if epoch % 2 == 0:
            print(f"Epoch {epoch}/10 | Loss: {loss.item():.4f}")

    # 保存最终模型
    torch.save(model.state_dict(), "checkpoints/model_final.pth")
    print("\n✅ 训练完成！权重已存入 checkpoints/model_final.pth")

if __name__ == "__main__":
    train()