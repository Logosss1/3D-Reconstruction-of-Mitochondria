import torch
import numpy as np
import trimesh
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

def create_rotation_video():
    print("🎬 正在加载 3D 模型并准备拍摄旋转视频...")
    # 加载你生成的模型
    mesh = trimesh.load("final_mitochondria.obj")
    v = mesh.vertices[::3]  # 采样点以保证动画流畅

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # 设置绘图样式
    scatter = ax.scatter(v[:, 0], v[:, 1], v[:, 2], s=2, c=v[:, 2], cmap='magma', alpha=0.6)
    
    # 隐藏坐标轴让画面更干净
    ax.set_axis_off()

    def update(frame):
        # 每一帧旋转 2 度
        ax.view_init(elev=20, azim=frame * 2)
        return scatter,

    print("🎥 正在渲染动画帧，请稍候...")
    # frames=180 代表旋转一圈 (180*2=360度)
    ani = FuncAnimation(fig, update, frames=180, interval=50)
    
    # 保存为 GIF (方便直接插入 PPT)
    ani.save("Mito3D_Rotation_Showcase.gif", writer='pillow')
    print("✅ 旋转动图已保存为: Mito3D_Rotation_Showcase.gif")

if __name__ == "__main__":
    create_rotation_video()