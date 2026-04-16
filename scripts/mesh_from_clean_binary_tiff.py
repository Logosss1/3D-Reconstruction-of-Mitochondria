"""
Create a cleaned mesh from a cleaned binary occupancy TIFF.

Workflow:
  1) Run generate.py with --save_mc_prob_tiff (native required) to export mc_prob_float32.tif.
  2) Run scripts/postprocess_binary_tiff.py to threshold + remove small components.
     It outputs a cleaned binary uint8 TIFF: *_bin_t{threshold}.tif (values 0/255).
  3) This script loads the cleaned binary TIFF and runs marching cubes on it,
     then exports OBJ + preview PNGs (same filenames as generate.py).
"""

from __future__ import annotations

import argparse
import os

import numpy as np
from skimage import measure
import trimesh


def _maybe_render_preview_pngs(
    mesh: trimesh.Trimesh,
    out_dir: str,
    prefix: str,
    *,
    preview_vertex_stride: int,
    preview_point_size: float,
    preview_alpha: float,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print("matplotlib not installed; skipped PNG previews.")
        return

    v = mesh.vertices[:: max(1, int(preview_vertex_stride))]
    if len(v) == 0:
        print("Empty mesh; skipped preview PNGs.")
        return

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        v[:, 0],
        v[:, 1],
        v[:, 2],
        s=float(preview_point_size),
        c=v[:, 2],
        cmap="magma",
        alpha=float(preview_alpha),
    )
    ax.set_xlabel("X (vox)")
    ax.set_ylabel("Y (vox)")
    ax.set_zlabel("Z (vox)")
    plt.title("3D reconstruction (voxel coords, centered)")

    png1 = os.path.join(out_dir, f"{prefix}preview_result.png")
    png2 = os.path.join(out_dir, f"{prefix}Thesis_Final_Result.png")
    plt.savefig(png1)
    plt.savefig(png2, dpi=300)
    plt.close(fig)
    print(f"Saved {png1}, {png2}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Marching cubes on cleaned binary TIFF -> OBJ + preview PNGs.")
    ap.add_argument("--in_tiff", type=str, required=True, help="Cleaned binary uint8 TIFF (0/255).")
    ap.add_argument("--out_dir", type=str, required=True, help="Output folder for OBJ and PNGs.")
    ap.add_argument("--prefix", type=str, default="", help="Prefix like crop1_ (matches generate.py naming).")
    ap.add_argument("--level", type=float, default=0.5, help="Iso-level on binary probs (default 0.5).")
    ap.add_argument("--preview_vertex_stride", type=int, default=2)
    ap.add_argument("--preview_point_size", type=float, default=5.0)
    ap.add_argument("--preview_alpha", type=float, default=0.8)
    args = ap.parse_args()

    import tifffile

    in_tiff = args.in_tiff
    if not os.path.isfile(in_tiff):
        raise SystemExit(f"--in_tiff not found: {in_tiff}")

    os.makedirs(args.out_dir, exist_ok=True)

    vol_u8 = tifffile.imread(in_tiff)
    if vol_u8.ndim != 3:
        raise SystemExit(f"Expected 3D TIFF, got shape={vol_u8.shape}")

    mask = vol_u8.astype(np.float32)
    # postprocess outputs 0/255; normalize to 0/1 for marching cubes
    mask = mask / 255.0

    pmin = float(mask.min())
    pmax = float(mask.max())
    if pmax - pmin < 1e-8:
        print(f"Binary volume is constant (min=max={pmin}); skipping mesh export.")
        return

    try:
        verts, faces, _, _ = measure.marching_cubes(mask, level=float(args.level))
    except Exception as e:
        print(f"marching_cubes failed: {e}")
        return

    if len(faces) == 0 or len(verts) == 0:
        print("marching_cubes returned empty mesh; skipping.")
        return

    verts = verts.astype(np.float64)
    verts = verts - verts.mean(axis=0, keepdims=True)  # match generate.py native behavior
    mesh = trimesh.Trimesh(vertices=verts, faces=faces)
    trimesh.repair.fix_normals(mesh)

    obj_path = os.path.join(args.out_dir, f"{args.prefix}final_mitochondria.obj")
    mesh.export(obj_path)
    print(f"Saved {obj_path} (verts={len(mesh.vertices)}, faces={len(mesh.faces)})")

    _maybe_render_preview_pngs(
        mesh,
        args.out_dir,
        args.prefix,
        preview_vertex_stride=args.preview_vertex_stride,
        preview_point_size=args.preview_point_size,
        preview_alpha=args.preview_alpha,
    )


if __name__ == "__main__":
    main()

