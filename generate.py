import argparse
import os
from typing import Optional

import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover
    plt = None
import torch
import zarr
from skimage import measure
import trimesh

from src.model import ConvONet
from src.post_process import reconstruction_pipeline
from src.volume_utils import crop_volume, downsample_for_encoder


def _load_model_checkpoint(
    model: ConvONet, checkpoint_path: str, device: torch.device
) -> Optional[dict]:
    """
    Load either:
    1) legacy raw state_dict, or
    2) full checkpoint dict with model_state_dict.
    Returns metadata dict for logging when available.
    """
    obj = torch.load(checkpoint_path, map_location=device)
    if isinstance(obj, dict) and "model_state_dict" in obj:
        model.load_state_dict(obj["model_state_dict"])
        meta = {
            "epoch": obj.get("epoch"),
            "args": obj.get("args"),
            "has_optimizer_state": "optimizer_state_dict" in obj,
            "has_scaler": "scaler" in obj,
        }
        return meta
    model.load_state_dict(obj)
    return None


def _parse_roi(
    roi: Optional[str], shape: tuple[int, int, int]
) -> tuple[slice, slice, slice]:
    if not roi:
        return slice(0, shape[0]), slice(0, shape[1]), slice(0, shape[2])
    vals = [int(x.strip()) for x in roi.split(",") if x.strip()]
    if len(vals) != 6:
        raise ValueError("--binary_cleanup_roi needs z0,y0,x0,dz,dy,dx")
    z0, y0, x0, dz, dy, dx = vals
    z1, y1, x1 = z0 + dz, y0 + dy, x0 + dx
    if min(vals) < 0 or z1 > shape[0] or y1 > shape[1] or x1 > shape[2]:
        raise ValueError(f"--binary_cleanup_roi out of bounds for shape {shape}")
    return slice(z0, z1), slice(y0, y1), slice(x0, x1)


def _cleanup_binary_from_probs(
    probs: np.ndarray,
    *,
    threshold: float,
    smooth_sigma: float,
    close_iters: int,
    open_iters: int,
    min_voxels: int,
    keep_largest_k: int,
    roi: Optional[str],
) -> np.ndarray:
    """
    Threshold + morphology cleanup on occupancy probabilities.

    Returns:
      clean_mask: bool array (same shape as probs).
    """
    from scipy.ndimage import binary_closing, binary_opening, gaussian_filter, label as ndi_label
    from skimage.morphology import remove_small_objects

    p = probs.astype(np.float32, copy=False)
    if smooth_sigma and smooth_sigma > 0:
        p = gaussian_filter(p, sigma=float(smooth_sigma), mode="nearest")
        p = np.clip(p, 0.0, 1.0)

    mask = p >= float(threshold)

    if close_iters and close_iters > 0:
        mask = binary_closing(mask, iterations=int(close_iters))
    if open_iters and open_iters > 0:
        mask = binary_opening(mask, iterations=int(open_iters))
    if min_voxels and min_voxels > 0:
        mask = remove_small_objects(mask, min_size=int(min_voxels))

    if keep_largest_k and keep_largest_k > 0:
        lab, n = ndi_label(mask)
        if n > 0:
            sizes = np.bincount(lab.ravel())
            sizes[0] = 0
            keep_labels = np.argsort(sizes)[-int(keep_largest_k) :]
            keep = np.zeros_like(mask, dtype=bool)
            for lid in keep_labels:
                if lid > 0:
                    keep |= lab == lid
            mask = keep

    zsl, ysl, xsl = _parse_roi(roi, mask.shape)
    roi_mask = np.zeros_like(mask, dtype=bool)
    roi_mask[zsl, ysl, xsl] = True
    mask &= roi_mask
    return mask


def _mesh_from_binary_mask_native(
    mask: np.ndarray,
) -> trimesh.Trimesh:
    # For a binary volume (0/1), iso=0.5 gives the boundary.
    probs = mask.astype(np.float32, copy=False)
    if probs.min() == probs.max():
        # empty or constant; return trivial mesh
        return trimesh.creation.uv_sphere(radius=10.0)
    verts, faces, _, _ = measure.marching_cubes(probs, level=0.5)
    verts = verts - verts.mean(axis=0, keepdims=True)
    mesh = trimesh.Trimesh(vertices=verts, faces=faces)
    trimesh.repair.fix_normals(mesh)
    return mesh


def _native_probs_tiled(
    model: ConvONet,
    real_raw: np.ndarray,
    device: torch.device,
    *,
    tile: int,
    overlap: int,
    query_chunk: int,
) -> np.ndarray:
    """Predict full native-size P(occ) volume with tiled, no-downsample inference."""
    d, h, w = real_raw.shape
    out = np.zeros((d, h, w), dtype=np.float32)
    cnt = np.zeros((d, h, w), dtype=np.float32)

    step = max(1, tile - overlap)
    z_starts = list(range(0, d, step))
    y_starts = list(range(0, h, step))
    x_starts = list(range(0, w, step))

    def _clamp_start(s: int, n: int) -> int:
        return max(0, min(s, max(0, n - tile)))

    for z0 in z_starts:
        for y0 in y_starts:
            for x0 in x_starts:
                zz = _clamp_start(z0, d)
                yy = _clamp_start(y0, h)
                xx = _clamp_start(x0, w)
                z1, y1, x1 = min(d, zz + tile), min(h, yy + tile), min(w, xx + tile)
                raw_tile = real_raw[zz:z1, yy:y1, xx:x1]
                td, th, tw = raw_tile.shape
                inp = (
                    torch.from_numpy(raw_tile)
                    .float()
                    .unsqueeze(0)
                    .unsqueeze(0)
                    .to(device)
                    / 255.0
                )
                gz, gy, gx = np.meshgrid(
                    np.linspace(0.0, 1.0, td, dtype=np.float32),
                    np.linspace(0.0, 1.0, th, dtype=np.float32),
                    np.linspace(0.0, 1.0, tw, dtype=np.float32),
                    indexing="ij",
                )
                pts = np.stack([gz.ravel(), gy.ravel(), gx.ravel()], axis=-1)
                probs_flat = np.empty((pts.shape[0],), dtype=np.float32)
                with torch.no_grad():
                    for s in range(0, pts.shape[0], query_chunk):
                        e = min(pts.shape[0], s + query_chunk)
                        p = torch.from_numpy(pts[s:e]).to(device).unsqueeze(0)
                        logit = model(inp, p)
                        probs_flat[s:e] = torch.sigmoid(logit).squeeze(0).detach().cpu().numpy()
                probs_tile = probs_flat.reshape(td, th, tw)
                out[zz:z1, yy:y1, xx:x1] += probs_tile
                cnt[zz:z1, yy:y1, xx:x1] += 1.0
    cnt = np.maximum(cnt, 1.0)
    return out / cnt


def _mesh_from_probs_native(
    probs: np.ndarray,
    *,
    level_mode: str,
    fixed_level: float,
    percentile: float,
    adaptive_floor: float,
) -> trimesh.Trimesh:
    flat = probs.ravel()
    pmin, pmax = float(flat.min()), float(flat.max())
    if level_mode == "fixed":
        level = float(fixed_level)
    elif level_mode == "percentile":
        level = float(np.percentile(flat, float(np.clip(percentile, 0.0, 100.0))))
    else:
        level = max(float(adaptive_floor), pmax * 0.5)
    if not (pmin < level < pmax):
        eps = max(1e-6, (pmax - pmin) * 1e-4)
        level = float(np.clip(level, pmin + eps, pmax - eps))
    print(
        f"Native MC: level_mode={level_mode} level={level:.4f} "
        f"min={pmin:.4f} max={pmax:.4f} mean={float(flat.mean()):.4f}"
    )
    verts, faces, _, _ = measure.marching_cubes(probs, level=level)
    verts = verts - verts.mean(axis=0, keepdims=True)
    mesh = trimesh.Trimesh(vertices=verts, faces=faces)
    trimesh.repair.fix_normals(mesh)
    return mesh


def run_generation():
    parser = argparse.ArgumentParser(description="Infer mesh from trained ConvONet.")
    parser.add_argument(
        "--zarr",
        type=str,
        default=None,
        help="Zarr with 'raw'. If omitted, uses hela2.zarr (--data_root, --crop_id).",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="data",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="jrc_hela-2",
    )
    parser.add_argument(
        "--crop_id",
        type=int,
        default=94,
    )
    parser.add_argument(
        "--zarr_root",
        type=str,
        default=None,
        help="Optional jrc_hela-2.zarr root path (contains recon-1).",
    )
    parser.add_argument(
        "--crop_export_dir",
        type=str,
        default=None,
        help="Directory with {dataset}_crop{crop_id}_mito.zarr; used when --zarr is omitted.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/model_final.pth",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="result",
        help="Directory for mesh and PNG outputs (created if missing).",
    )
    parser.add_argument(
        "--name_prefix",
        type=str,
        default="",
        help="Prefix for output files, e.g. crop94_ -> crop94_final_mitochondria.obj",
    )
    parser.add_argument(
        "--infer_max_spatial",
        type=int,
        default=128,
        help="Downsample raw so max(D,H,W) equals this before encoder.",
    )
    parser.add_argument(
        "--mc_resolution",
        type=int,
        default=128,
        help="Marching-cubes grid resolution.",
    )
    parser.add_argument(
        "--mc_level_mode",
        type=str,
        choices=("fixed", "percentile", "adaptive"),
        default="fixed",
        help=(
            "Iso-surface level for marching cubes: "
            "fixed=mc_fixed_level; percentile=mc_percentile of voxel probs; "
            "adaptive=legacy max(mc_adaptive_floor, max_prob*0.5)."
        ),
    )
    parser.add_argument(
        "--mc_fixed_level",
        type=float,
        default=0.5,
        help="Used when mc_level_mode=fixed (typical 0.3–0.7).",
    )
    parser.add_argument(
        "--mc_percentile",
        type=float,
        default=90.0,
        help="Used when mc_level_mode=percentile (0–100; higher => stricter / smaller surface).",
    )
    parser.add_argument(
        "--mc_adaptive_floor",
        type=float,
        default=0.2,
        help="Floor when mc_level_mode=adaptive.",
    )
    parser.add_argument(
        "--mesh_strip_mc_shell_cells",
        type=float,
        default=3.0,
        help=(
            "Strip mesh faces in the outer N cells of the mc_resolution^3 grid (unit cube), "
            "before scaling to full Z,Y,X — removes 'box walls' on large crops; 0 disables."
        ),
    )
    parser.add_argument(
        "--mesh_strip_boundary_voxels",
        type=float,
        default=0.0,
        help=(
            "Extra strip in full voxels from the data box after MC scaling; 0=off. "
            "Usually unnecessary if mesh_strip_mc_shell_cells is set."
        ),
    )
    parser.add_argument(
        "--mc_prob_smooth_sigma",
        type=float,
        default=0.0,
        help=(
            "Gaussian smoothing sigma (voxels in MC grid) on occupancy probs before "
            "marching cubes; 0=off. Try 0.5–1.0 to reduce grainy MC surfaces."
        ),
    )
    parser.add_argument(
        "--preview_vertex_stride",
        type=int,
        default=2,
        help="Subsample mesh vertices for scatter preview: vertices[::stride].",
    )
    parser.add_argument(
        "--preview_point_size",
        type=float,
        default=5.0,
        help="Scatter marker size for preview PNG.",
    )
    parser.add_argument(
        "--preview_alpha",
        type=float,
        default=0.8,
        help="Scatter alpha for preview PNG.",
    )
    parser.add_argument(
        "--save_binary_preview",
        action="store_true",
        help=(
            "Save a 2-panel PNG: max-Z projection of P(occ) vs binary (P≥threshold). "
            "Matches advisor-style 'binarize at half then look' on the MC grid."
        ),
    )
    parser.add_argument(
        "--mc_binary_threshold",
        type=float,
        default=0.5,
        help="Threshold for --save_binary_preview right panel (default 0.5).",
    )
    parser.add_argument(
        "--save_mc_prob_tiff",
        action="store_true",
        help=(
            "Save MC occupancy probability grid as float32 TIFF stack (Fiji/ImageJ: File › Import › Bio-Formats "
            "or plain TIFF stack). Same resolution as --mc_resolution."
        ),
    )
    parser.add_argument(
        "--save_mc_prob_tiff_binary",
        action="store_true",
        help=(
            "Save MC grid as binary uint8 TIFF (0/255) using P>=mc_binary_threshold. "
            "Use with --save_mc_prob_tiff for both float and mask."
        ),
    )
    parser.add_argument(
        "--raw_subcrop",
        type=str,
        default=None,
        help=(
            "Optional Z,Y,X subvolume from loaded raw: 'z0,y0,x0,dz,dy,dx' integers. "
            "Use to inspect a spatial block not aligned with training random crops (same crop_id, new window)."
        ),
    )
    parser.add_argument(
        "--native_required",
        action="store_true",
        help=(
            "Strict native mode: do not downsample input; reconstruct on native D,H,W grid "
            "via tiled inference. Slower but keeps original voxel size."
        ),
    )
    parser.add_argument(
        "--native_tile",
        type=int,
        default=160,
        help="Tile edge size in native_required mode (lower if OOM).",
    )
    parser.add_argument(
        "--native_overlap",
        type=int,
        default=24,
        help="Tile overlap voxels in native_required mode.",
    )
    parser.add_argument(
        "--query_chunk",
        type=int,
        default=250000,
        help="Point-query chunk size in native_required mode (lower if OOM).",
    )

    # Native-only binary cleanup: remove tiny components before marching cubes.
    parser.add_argument(
        "--binary_cleanup",
        action="store_true",
        help=(
            "Native-only: after probs are computed, binarize and run morphology cleanup "
            "(remove_small_objects) before marching cubes. Outputs OBJ/PNG only (no TIFF)."
        ),
    )
    parser.add_argument(
        "--binary_cleanup_threshold",
        type=float,
        default=0.5,
        help="Threshold for --binary_cleanup (P>=thr => 1).",
    )
    parser.add_argument(
        "--binary_cleanup_min_voxels",
        type=int,
        default=300,
        help="Remove connected components smaller than this many voxels.",
    )
    parser.add_argument(
        "--binary_cleanup_close_iters",
        type=int,
        default=1,
        help="Binary closing iterations (0 disables).",
    )
    parser.add_argument(
        "--binary_cleanup_open_iters",
        type=int,
        default=0,
        help="Binary opening iterations (0 disables).",
    )
    parser.add_argument(
        "--binary_cleanup_keep_largest_k",
        type=int,
        default=0,
        help="Keep only largest K components after cleanup (0 keeps all).",
    )
    parser.add_argument(
        "--binary_cleanup_smooth_sigma",
        type=float,
        default=0.0,
        help="Optional gaussian smoothing on probs before threshold (native-only).",
    )
    parser.add_argument(
        "--binary_cleanup_roi",
        type=str,
        default=None,
        help="Optional cleanup ROI: z0,y0,x0,dz,dy,dx (native-only).",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = ConvONet().to(device)
    ckpt_meta = _load_model_checkpoint(model, args.checkpoint, device)
    if ckpt_meta is None:
        print("Loaded legacy checkpoint format (raw state_dict).")
    else:
        print(
            "Loaded full checkpoint format "
            f"(epoch={ckpt_meta.get('epoch')}, "
            f"optimizer_state={ckpt_meta.get('has_optimizer_state')}, "
            f"scaler={ckpt_meta.get('has_scaler')})."
        )

    if args.zarr:
        z = zarr.open(args.zarr, mode="r")
        real_raw = np.array(z["raw"])
    elif args.crop_export_dir:
        from src.crop_export_zarr import load_crop_export_volume

        real_raw, _ = load_crop_export_volume(
            args.crop_export_dir, args.dataset, args.crop_id
        )
        print(f"Source: crop export {args.crop_id} (mito-grid raw+label Zarr)")
    else:
        from src.hela2_zarr_crop import load_crop_em_mito_aligned

        real_raw, _ = load_crop_em_mito_aligned(
            args.data_root,
            args.dataset,
            args.crop_id,
            zarr_root=args.zarr_root,
        )
        print(f"Source: hela2 crop {args.crop_id} EM (aligned to mito grid)")

    if args.raw_subcrop:
        parts = [int(x.strip()) for x in args.raw_subcrop.split(",") if x.strip()]
        if len(parts) != 6:
            raise SystemExit("--raw_subcrop needs exactly 6 integers: z0,y0,x0,dz,dy,dx")
        z0, y0, x0, dz, dy, dx = parts
        dmax, hmax, wmax = real_raw.shape
        if z0 + dz > dmax or y0 + dy > hmax or x0 + dx > wmax or min(parts) < 0:
            raise SystemExit(
                f"--raw_subcrop out of bounds: volume shape {real_raw.shape}, got {parts}"
            )
        real_raw = crop_volume(real_raw, z0, y0, x0, dz, dy, dx)
        print(
            f"Raw subcrop applied: offset (z,y,x)=({z0},{y0},{x0}) size (dz,dy,dx)=({dz},{dy},{dx})"
        )

    orig_shape = real_raw.shape
    print(f"Raw shape (Z,Y,X): {orig_shape}")

    real_input = torch.from_numpy(real_raw).float().unsqueeze(0).unsqueeze(0).to(device) / 255.0
    if args.native_required:
        model_in = real_input
        print("Native required: no downsample before encoder.")
    else:
        model_in, _ = downsample_for_encoder(real_input, args.infer_max_spatial)
        if model_in.shape != real_input.shape:
            print(
                f"Inference input resized for encoder: {tuple(model_in.shape)} "
                f"(original {tuple(real_input.shape)})"
            )

    print("Computing mesh (OBJ)...")
    os.makedirs(args.output_dir, exist_ok=True)
    prefix = args.name_prefix or ""
    binary_preview = None
    if args.save_binary_preview:
        binary_preview = os.path.join(
            args.output_dir, f"{prefix}mc_prob_binary_preview.png"
        )

    prob_tiff = None
    if args.save_mc_prob_tiff:
        prob_tiff = os.path.join(args.output_dir, f"{prefix}mc_prob_float32.tif")
    prob_tiff_bin = None
    if args.save_mc_prob_tiff_binary:
        prob_tiff_bin = os.path.join(args.output_dir, f"{prefix}mc_prob_binary_uint8.tif")

    if args.native_required:
        probs = _native_probs_tiled(
            model,
            real_raw,
            device,
            tile=max(16, int(args.native_tile)),
            overlap=max(0, int(args.native_overlap)),
            query_chunk=max(10000, int(args.query_chunk)),
        )
        if args.mc_prob_smooth_sigma and args.mc_prob_smooth_sigma > 0:
            from scipy.ndimage import gaussian_filter

            probs = gaussian_filter(
                probs, sigma=float(args.mc_prob_smooth_sigma), mode="nearest"
            ).astype(np.float32)
            probs = np.clip(probs, 0.0, 1.0)
            print(f"Native probs smoothed: sigma={args.mc_prob_smooth_sigma}")
        if prob_tiff is not None:
            import tifffile

            tifffile.imwrite(prob_tiff, np.ascontiguousarray(probs.astype(np.float32)), imagej=True)
            print(f"Saved native MC prob TIFF: {prob_tiff} shape={probs.shape}")
        if prob_tiff_bin is not None:
            import tifffile

            mb = ((probs >= float(args.mc_binary_threshold)).astype(np.uint8) * 255).astype(np.uint8)
            tifffile.imwrite(prob_tiff_bin, np.ascontiguousarray(mb), imagej=True)
            print(
                f"Saved native binary TIFF: {prob_tiff_bin} "
                f"shape={mb.shape} thr={args.mc_binary_threshold}"
            )
        if args.binary_cleanup:
            print(
                "Native binary cleanup enabled: "
                f"thr={args.binary_cleanup_threshold} min_voxels={args.binary_cleanup_min_voxels} "
                f"close_iters={args.binary_cleanup_close_iters} open_iters={args.binary_cleanup_open_iters} "
                f"keep_largest_k={args.binary_cleanup_keep_largest_k} smooth_sigma={args.binary_cleanup_smooth_sigma}"
            )
            clean_mask = _cleanup_binary_from_probs(
                probs,
                threshold=float(args.binary_cleanup_threshold),
                smooth_sigma=float(args.binary_cleanup_smooth_sigma),
                close_iters=int(args.binary_cleanup_close_iters),
                open_iters=int(args.binary_cleanup_open_iters),
                min_voxels=int(args.binary_cleanup_min_voxels),
                keep_largest_k=int(args.binary_cleanup_keep_largest_k),
                roi=getattr(args, "binary_cleanup_roi", None),
            )
            mesh = _mesh_from_binary_mask_native(clean_mask)
        else:
            mesh = _mesh_from_probs_native(
                probs,
                level_mode=args.mc_level_mode,
                fixed_level=args.mc_fixed_level,
                percentile=args.mc_percentile,
                adaptive_floor=args.mc_adaptive_floor,
            )
    else:
        mesh, _ = reconstruction_pipeline(
            model,
            model_in,
            real_raw,
            original_voxel_shape=orig_shape,
            resolution=args.mc_resolution,
            level_mode=args.mc_level_mode,
            fixed_level=args.mc_fixed_level,
            percentile=args.mc_percentile,
            adaptive_floor=args.mc_adaptive_floor,
            strip_boundary_voxels=args.mesh_strip_boundary_voxels,
            strip_mc_shell_cells=args.mesh_strip_mc_shell_cells,
            mc_prob_smooth_sigma=args.mc_prob_smooth_sigma,
            binary_preview_path=binary_preview,
            mc_binary_threshold=args.mc_binary_threshold,
            prob_tiff_path=prob_tiff,
            prob_tiff_binary_path=prob_tiff_bin,
        )
    obj_path = os.path.join(args.output_dir, f"{prefix}final_mitochondria.obj")
    mesh.export(obj_path)
    print(f"Saved {obj_path}")

    if plt is not None:
        print("Rendering preview figures...")
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection="3d")
        stride = max(1, int(args.preview_vertex_stride))
        v = mesh.vertices[::stride]
        ax.scatter(
            v[:, 0],
            v[:, 1],
            v[:, 2],
            s=args.preview_point_size,
            c=v[:, 2],
            cmap="magma",
            alpha=args.preview_alpha,
        )
        ax.set_xlabel("X (vox)")
        ax.set_ylabel("Y (vox)")
        ax.set_zlabel("Z (vox)")
        plt.title("3D reconstruction (voxel coords, centered)")
        png1 = os.path.join(args.output_dir, f"{prefix}preview_result.png")
        png2 = os.path.join(args.output_dir, f"{prefix}Thesis_Final_Result.png")
        plt.savefig(png1)
        plt.savefig(png2, dpi=300)
        print(f"Saved {png1}, {png2}")
    else:
        print("matplotlib not installed; skipped PNG previews.")


if __name__ == "__main__":
    run_generation()
