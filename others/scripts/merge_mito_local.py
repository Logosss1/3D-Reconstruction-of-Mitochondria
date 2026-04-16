import os
import numpy as np
import pandas as pd
import zarr
from scipy.ndimage import label as cc_label

def parse_vec(s):
    # "[2.62;2.0;2.0]" -> [2.62,2.0,2.0]
    s = str(s).strip()
    s = s.strip("[]")
    return np.array([float(x) for x in s.split(";")], dtype=np.float64)

def mito_s0_path(data_root, dataset, crop_id, label_root="labels/groundtruth"):
    """
    dataset: 'jrc_hela-2' or 'jrc_hela-3'
    data_root: .../data
    """
    # hela2.zarr / hela3.zarr 对应 dataset
    hela_bucket = "hela2.zarr" if dataset.endswith("hela-2") else "hela3.zarr"
    jrc_bucket = dataset + ".zarr"
    base = os.path.join(data_root, hela_bucket, jrc_bucket, "recon-1", label_root, f"crop{crop_id}", "mito")
    return os.path.join(base, "s0")

def merge_crops_to_local(data_root, manifest_csv, np_s0_csv, dataset, crop_ids):
    man = pd.read_csv(manifest_csv)
    # 可选：用 np_s0_csv 做 crop_ids 过滤（你也可以直接传 crop_ids）
    # npdf = pd.read_csv(np_s0_csv)

    sub = man[(man["dataset"] == dataset) & (man["class_label"] == "mito")]
    sub = sub[sub["crop_name"].isin(crop_ids)].drop_duplicates(subset=["crop_name"])

    placements = []
    for _, row in sub.iterrows():
        crop_id = int(row["crop_name"])
        tz, ty, tx = parse_vec(row["translation"])
        vz, vy, vx = parse_vec(row["voxel_size"])
        dz, dy, dx = parse_vec(row["shape"]).astype(int)

        z0 = tz / vz
        y0 = ty / vy
        x0 = tx / vx
        placements.append((crop_id, z0, y0, x0, dz, dy, dx))

    z0s = np.array([p[1] for p in placements], dtype=np.float64)
    y0s = np.array([p[2] for p in placements], dtype=np.float64)
    x0s = np.array([p[3] for p in placements], dtype=np.float64)
    min_z0, min_y0, min_x0 = z0s.min(), y0s.min(), x0s.min()

    # local size
    z1s = np.array([p[1] + p[4] for p in placements], dtype=np.float64)
    y1s = np.array([p[2] + p[5] for p in placements], dtype=np.float64)
    x1s = np.array([p[3] + p[6] for p in placements], dtype=np.float64)

    local_Z = int(np.ceil(z1s.max() - min_z0))
    local_Y = int(np.ceil(y1s.max() - min_y0))
    local_X = int(np.ceil(x1s.max() - min_x0))

    local_mask = np.zeros((local_Z, local_Y, local_X), dtype=np.uint8)

    for crop_id, z0, y0, x0, dz, dy, dx in placements:
        z_start = int(round(z0 - min_z0))
        y_start = int(round(y0 - min_y0))
        x_start = int(round(x0 - min_x0))

        mito_path = mito_s0_path(data_root, dataset, crop_id)
        z = zarr.open(mito_path, mode="r")
        m = (np.array(z) > 0).astype(np.uint8)  # 二值化：只保留是否在线粒体

        # 粘贴（重叠取 max）
        local_mask[z_start:z_start+dz, y_start:y_start+dy, x_start:x_start+dx] = np.maximum(
            local_mask[z_start:z_start+dz, y_start:y_start+dy, x_start:x_start+dx],
            m
        )

    return local_mask, (min_z0, min_y0, min_x0), (local_Z, local_Y, local_X)

def connected_components_3d(binary_mask):
    # 26-connectivity: 3x3x3 全 ones
    structure = np.ones((3,3,3), dtype=np.uint8)
    inst_map, num = cc_label(binary_mask.astype(bool), structure=structure)
    return inst_map, num

if __name__ == "__main__":
    data_root = r"D:\1AAA\mit_rec\cursor\Mito3D_Reconstruction_Thesis\data"
    manifest_csv = r"C:\Users\Lenovo\Desktop\train_crop_manifest.csv"
    np_s0_csv = r"C:\Users\Lenovo\Desktop\np_s0_with_data.csv"

    dataset = "jrc_hela-2"
    crop_ids = [94]  # 先从一个 crop 验证；再换成多个 crop_ids

    local_mask, origin, shape = merge_crops_to_local(
        data_root, manifest_csv, np_s0_csv, dataset, crop_ids
    )
    inst_map, num = connected_components_3d(local_mask)

    print("Local mask shape:", shape)
    print("Num instances in local volume:", num)