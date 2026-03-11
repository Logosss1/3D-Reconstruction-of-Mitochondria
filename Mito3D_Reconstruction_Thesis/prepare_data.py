import numpy as np
import os
import zarr

def generate_synthetic_research_data():
    print("🧪 正在生成高仿真线粒体科研数据集 (模拟 OpenOrganelle jrc_hela-2)...")
    
    size = 128
    save_path = "data/sample_hela2"
    os.makedirs(save_path, exist_ok=True)

    # 1. 生成模拟掩码 (Principle 1: 拓扑网状结构)
    label = np.zeros((size, size, size), dtype=np.uint8)
    # 模拟几个相互连接的管状线粒体
    t = np.linspace(0, 4 * np.pi, size)
    for i in range(size):
        # 管道 1
        y, x = int(size/2 + 20 * np.sin(t[i])), int(size/2 + 20 * np.cos(t[i]))
        label[i, y-8:y+8, x-8:x+8] = 1
        # 管道 2 (产生交叉连通)
        y2, x2 = int(size/2 + 15 * np.cos(t[i])), int(size/2)
        label[i, y2-6:y2+6, x2-6:x2+6] = 1

    # 2. 生成模拟灰度图 (Principle 2: 8nm 物理细节位移)
    # 基础亮度 + 膜结构增强 + EM 高斯噪声
    raw = np.random.normal(128, 20, (size, size, size)).astype(np.uint8)
    # 让有线粒体的地方亮度降低（模拟电镜下线粒体较暗的特征）
    raw[label > 0] = (raw[label > 0] * 0.6).astype(np.uint8)

    # 3. 保存为标准 Zarr 格式
    store = zarr.DirectoryStore(os.path.join(save_path, "sample.zarr"))
    root = zarr.group(store=store, overwrite=True)
    root.create_dataset('raw', data=raw, chunks=(64, 64, 64))
    root.create_dataset('label', data=label, chunks=(64, 64, 64))

    print(f"✅ [本地生成成功] 仿真数据已就绪！")
    print(f"📍 本地路径: {save_path}")
    print("🚀 现在请直接运行: python train.py")

if __name__ == "__main__":
    generate_synthetic_research_data()