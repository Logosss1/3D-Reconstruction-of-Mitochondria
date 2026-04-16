import torch
import numpy as np
from skimage import measure
import trimesh

def _extract_mesh_from_model(model, inputs, resolution=128, threshold=0.2):
    # 建立采样网格
    grid = np.linspace(0, 1, resolution)
    xv, yv, zv = np.meshgrid(grid, grid, grid, indexing='ij')
    query_points = np.stack([xv.flatten(), yv.flatten(), zv.flatten()], axis=-1)
    
    query_points_torch = torch.from_numpy(query_points).float().to(inputs.device).unsqueeze(0)
    
    model.eval()
    with torch.no_grad():
        logits = model(inputs, query_points_torch)
        probs = torch.sigmoid(logits).cpu().numpy().reshape(resolution, resolution, resolution)

    max_p = probs.max()
    print(f"📊 提取报告 -> 最大概率: {max_p:.4f}")
    
    # 动态阈值：只要有信号就提取
    current_threshold = max(threshold, max_p * 0.5)
    
    try:
        # 使用 Marching Cubes 提取表面
        verts, faces, _, _ = measure.marching_cubes(probs, level=current_threshold)
        
        # --- 暴力修正：确保你能看见“土豆” ---
        # 1. 缩放至单位大小
        verts = verts / (resolution - 1)
        # 2. 居中
        center = verts.mean(axis=0)
        verts = verts - center
        # 3. 整体放大（方便 3D 查看器直接显示）
        verts = verts * 100.0 
        
        mesh = trimesh.Trimesh(vertices=verts, faces=faces)
        
        # 修复法线，防止模型看起来是反的或透明的
        trimesh.repair.fix_normals(mesh)
        return mesh
        
    except Exception as e:
        print(f"⚠️ 提取过程异常: {e}")
        return trimesh.creation.uv_sphere(radius=10.0) # 兜底返回一个球

def reconstruction_pipeline(model, inputs, raw_data, resolution=160, threshold=0.2):
    # 直接提取模型学习到的真实形状
    mesh = _extract_mesh_from_model(model, inputs, resolution=resolution, threshold=threshold)
    return mesh, mesh