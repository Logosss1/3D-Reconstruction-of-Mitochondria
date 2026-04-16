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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 开始『实心化』训练...")
    
    z = zarr.open("data/sample_hela2/sample.zarr", mode='r')
    raw = torch.from_numpy(np.array(z['raw'])).float().unsqueeze(0).unsqueeze(0).to(device) / 255.0
    label_np = np.array(z['label'])
    label = torch.from_numpy(label_np).float().unsqueeze(0).unsqueeze(0).to(device)

    model = ConvONet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=5e-4) # 稍微调高学习率

    # 预提取有“肉”的坐标，防止空盒子
    pos_indices = torch.from_numpy(np.argwhere(label_np > 0)).float().to(device)
    print(f"✅ 找到 {len(pos_indices)} 个实心像素，开始训练...")

    for epoch in range(1, 301):
        optimizer.zero_grad()
        
        # 混合采样：一半肉，一半背景
        num_points = 10000
        idx = torch.randint(0, len(pos_indices), (num_points // 2,))
        p_pos = pos_indices[idx] / 64.0
        p_rand = torch.rand(num_points // 2, 3).to(device)
        points = torch.cat([p_pos, p_rand], dim=0).unsqueeze(0)
        
        # 提取真实标签
        d, h, w = label.shape[-3:]
        iz = torch.clamp((points[0,:,0]*d).long(), 0, d-1)
        ih = torch.clamp((points[0,:,1]*h).long(), 0, h-1)
        iw = torch.clamp((points[0,:,2]*w).long(), 0, w-1)
        target = label[0, 0, iz, ih, iw].unsqueeze(0)

        out = model(raw, points)
        loss = nn.BCEWithLogitsLoss()(out, target) + dice_loss(out, target)
        
        loss.backward()
        optimizer.step()

        if epoch % 20 == 0:
            print(f"Epoch {epoch}/300 | Loss: {loss.item():.4f}")

    os.makedirs('checkpoints', exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/model_final.pth")
    print("✅ 训练完成！权重已存入 checkpoints/model_final.pth")

if __name__ == "__main__":
    train()