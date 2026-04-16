import csv
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import trimesh
import zarr
from skimage import measure

from src.model import ConvONet


def make_mesh_from_volume(
    volume: np.ndarray,
    level_candidates: List[float],
) -> trimesh.Trimesh:
    best_mesh = None
    best_v = -1
    for lv in level_candidates:
        try:
            verts, faces, _, _ = measure.marching_cubes(volume, level=float(lv))
            if len(verts) > best_v:
                best_v = len(verts)
                verts = verts / max(volume.shape[0] - 1, 1)
                verts = verts - verts.mean(axis=0, keepdims=True)
                verts = verts * 100.0
                mesh = trimesh.Trimesh(vertices=verts, faces=faces)
                trimesh.repair.fix_normals(mesh)
                best_mesh = mesh
        except Exception:
            continue
    if best_mesh is None:
        best_mesh = trimesh.creation.uv_sphere(radius=10.0)
    return best_mesh


def model_predict_mesh(
    model: ConvONet,
    raw_zyx: np.ndarray,
    device: torch.device,
    max_dim: int = 256,
    resolution: int = 128,
) -> trimesh.Trimesh:
    d, h, w = raw_zyx.shape
    if max(d, h, w) > max_dim:
        z0 = max(0, d // 2 - max_dim // 2)
        y0 = max(0, h // 2 - max_dim // 2)
        x0 = max(0, w // 2 - max_dim // 2)
        z1 = min(d, z0 + max_dim)
        y1 = min(h, y0 + max_dim)
        x1 = min(w, x0 + max_dim)
        vol = raw_zyx[z0:z1, y0:y1, x0:x1]
    else:
        vol = raw_zyx

    x = torch.from_numpy(vol.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device) / 255.0

    grid = np.linspace(0.0, 1.0, resolution, dtype=np.float32)
    xv, yv, zv = np.meshgrid(grid, grid, grid, indexing="ij")
    q = np.stack([xv.reshape(-1), yv.reshape(-1), zv.reshape(-1)], axis=-1)
    q_t = torch.from_numpy(q).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        logits = model(x, q_t)
        probs = torch.sigmoid(logits).squeeze(0).cpu().numpy().reshape(resolution, resolution, resolution)

    pmax = float(probs.max())
    # More robust than a single fixed threshold
    levels = [
        max(0.02, pmax * 0.5),
        max(0.02, pmax * 0.35),
        max(0.02, pmax * 0.2),
        0.05,
        0.02,
    ]
    return make_mesh_from_volume(probs, levels)


def gt_mesh_from_label(label_zyx: np.ndarray) -> trimesh.Trimesh:
    fg = (label_zyx > 0).astype(np.float32)
    return make_mesh_from_volume(fg, [0.5, 0.3, 0.1])


def save_mesh_preview(mesh: trimesh.Trimesh, png_path: Path, title: str) -> None:
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    v = mesh.vertices[::2] if len(mesh.vertices) > 2 else mesh.vertices
    ax.scatter(v[:, 0], v[:, 1], v[:, 2], s=4, c=v[:, 2], cmap="magma", alpha=0.85)
    ax.set_title(title)
    ax.set_axis_off()
    plt.tight_layout()
    fig.savefig(png_path, dpi=220)
    plt.close(fig)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = Path("checkpoints/model_hela23_exports_v2_best.pth")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")

    model = ConvONet().to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    crop_paths = sorted(
        list(Path("data/crop_exports_hela2_mito_bg").glob("*.zarr"))
        + list(Path("data/crop_exports_hela3_mito_bg").glob("*.zarr"))
    )
    if not crop_paths:
        raise RuntimeError("No crop export zarr found.")

    out_dir = Path("result/hela23_all_crops")
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Tuple[str, int, int]] = []
    for p in crop_paths:
        z = zarr.open(str(p), mode="r")
        raw = np.asarray(z["raw"][:], dtype=np.float32)
        label = np.asarray(z["label"][:], dtype=np.uint8)
        stem = p.stem

        pred_mesh = model_predict_mesh(model, raw, device=device, max_dim=256, resolution=128)
        gt_mesh = gt_mesh_from_label(label)

        pred_obj = out_dir / f"{stem}_pred.obj"
        gt_obj = out_dir / f"{stem}_gt.obj"
        pred_png = out_dir / f"{stem}_pred.png"
        gt_png = out_dir / f"{stem}_gt.png"

        pred_mesh.export(str(pred_obj))
        gt_mesh.export(str(gt_obj))
        save_mesh_preview(pred_mesh, pred_png, f"{stem} - pred")
        save_mesh_preview(gt_mesh, gt_png, f"{stem} - gt")

        rows.append((stem, len(pred_mesh.vertices), len(gt_mesh.vertices)))
        print(f"✅ {stem}: pred_v={len(pred_mesh.vertices)} gt_v={len(gt_mesh.vertices)}")

    report = out_dir / "recon_summary.csv"
    with report.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["crop", "pred_vertices", "gt_vertices"])
        w.writerows(rows)
    print("Done:", report)


if __name__ == "__main__":
    import os

    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    main()

