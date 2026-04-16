"""Validate one crop: load GT mito label + predicted mesh, print/save metrics (thesis table)."""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import trimesh

from src.crop_export_zarr import load_crop_export_label_only
from src.hela2_zarr_crop import load_crop_em_mito_aligned
from src.mito_metrics import compute_mesh_label_metrics


def main() -> None:
    ap = argparse.ArgumentParser(description="Mesh vs mito label metrics for one crop.")
    ap.add_argument("--data_root", type=str, default="data")
    ap.add_argument("--dataset", type=str, default="jrc_hela-2")
    ap.add_argument("--crop_id", type=int, required=True)
    ap.add_argument("--zarr_root", type=str, default=None)
    ap.add_argument(
        "--crop_export_dir",
        type=str,
        default=None,
        help="GT label from {dataset}_crop{id}_mito.zarr instead of Open Organelle.",
    )
    ap.add_argument("--mesh", type=str, required=True, help="Path to OBJ from generate.py")
    ap.add_argument(
        "--out_json",
        type=str,
        default=None,
        help="Optional path to write metrics JSON (same folder as figures recommended).",
    )
    args = ap.parse_args()

    if args.crop_export_dir:
        label_bin = load_crop_export_label_only(
            args.crop_export_dir, args.dataset, args.crop_id
        )
    else:
        _, label_vol = load_crop_em_mito_aligned(
            args.data_root,
            args.dataset,
            args.crop_id,
            zarr_root=args.zarr_root,
        )
        label_bin = (label_vol > 0).astype(np.uint8)

    loaded = trimesh.load(args.mesh, force="mesh")
    if isinstance(loaded, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(loaded.geometry.values()))
    else:
        mesh = loaded
    metrics = compute_mesh_label_metrics(mesh, label_bin)
    metrics["crop_id"] = args.crop_id
    metrics["mesh_path"] = os.path.abspath(args.mesh)

    line = (
        f"crop {args.crop_id}: voxel_IoU={metrics['voxel_iou']:.4f} "
        f"voxel_Dice={metrics['voxel_dice']:.4f} "
        f"chamfer_pred_to_gt={metrics['chamfer_mean_pred_to_gt_vox']:.4f} (voxels)"
    )
    print(line)

    if args.out_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.out_json)) or ".", exist_ok=True)
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print(f"Wrote {args.out_json}")


if __name__ == "__main__":
    main()
