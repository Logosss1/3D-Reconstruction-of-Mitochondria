import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvONet(nn.Module):
    def __init__(self):
        super().__init__()
        # 3D 卷积层：从原始 EM 图像提取生物特征
        self.conv = nn.Sequential(
            nn.Conv3d(1, 16, 3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.Conv3d(16, 32, 3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU()
        )
        # 解码层：结合特征和空间位置判断“肉”在哪里
        self.fc = nn.Sequential(
            nn.Linear(32 + 3, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x, p):
        # 提取图像全局特征图
        feat = self.conv(x) 
        
        # 关键：将采样点 p (0~1) 映射到图像特征空间 (-1~1)
        # 这样模型在判断每个点时，都会“看一眼”那个位置的原始图像
        grid_p = p.flip(-1).unsqueeze(1).unsqueeze(1) * 2 - 1
        sampled_feat = F.grid_sample(feat, grid_p, align_corners=True, mode='bilinear')
        sampled_feat = sampled_feat.squeeze(2).squeeze(2).permute(0, 2, 1)
        
        # 拼接特征和坐标
        combined = torch.cat([sampled_feat, p], dim=-1)
        return self.fc(combined).squeeze(-1)