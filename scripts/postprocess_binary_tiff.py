"""
Post-process occupancy TIFF for clearer visualization and cleaner masks.

Input: float32 probability TIFF (0..1) from generate.py --save_mc_prob_tiff
Output:
  - binary uint8 TIFF (0/255)
  - optional cleaned uint8 TIFF
  - before/after max-projection PNG for quick review
"""

from __future__ import annotations

import argparse
import os

import numpy as np
from scipy.ndimage import binary_closing, binary_opening, gaussian_filter
from skimage.morphology import remove_small_objects


def _maxproj_png(path: str, vol: np.ndarray, title: str) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    img = np.max(vol.astype(np.float32), axis=0)
    plt.figure(figsize=(6, 5))
    plt.imshow(img, cmap="gray")
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def _parse_roi(roi: str | None, shape: tuple[int, int, int]) -> tuple[slice, slice, slice]:
    if not roi:
        return slice(0, shape[0]), slice(0, shape[1]), slice(0, shape[2])
    vals = [int(x.strip()) for x in roi.split(",") if x.strip()]
    if len(vals) != 6:
        raise ValueError("--roi needs z0,y0,x0,dz,dy,dx")
    z0, y0, x0, dz, dy, dx = vals
    z1, y1, x1 = z0 + dz, y0 + dy, x0 + dx
    if min(vals) < 0 or z1 > shape[0] or y1 > shape[1] or x1 > shape[2]:
        raise ValueError(f"--roi out of bounds for shape {shape}")
    return slice(z0, z1), slice(y0, y1), slice(x0, x1)


def main() -> None:
    ap = argparse.ArgumentParser(description="Threshold + clean occupancy TIFF (3D).")
    ap.add_argument("--in_tiff", type=str, required=True, help="float32 mc_prob tiff")
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--threshold", type=float, default=0.5, help="P>=threshold => foreground")
    ap.add_argument("--smooth_sigma", type=float, default=0.0, help="gaussian on prob before threshold")
    ap.add_argument("--min_voxels", type=int, default=300, help="remove components smaller than this")
    ap.add_argument("--close_iters", type=int, default=1, help="binary closing iterations")
    ap.add_argument("--open_iters", type=int, default=0, help="binary opening iterations")
    ap.add_argument(
        "--keep_largest_k",
        type=int,
        default=1,
        help="keep largest K connected components after cleanup (0 keeps all)",
    )
    ap.add_argument(
        "--roi",
        type=str,
        default=None,
        help="optional z0,y0,x0,dz,dy,dx; outside ROI is zeroed (roughly cuts irrelevant regions)",
    )
    args = ap.parse_args()

    import tifffile
    from scipy.ndimage import label

    prob = tifffile.imread(args.in_tiff).astype(np.float32, copy=False)
    os.makedirs(args.out_dir, exist_ok=True)

    p = prob
    if args.smooth_sigma > 0:
        p = gaussian_filter(p, sigma=float(args.smooth_sigma), mode="nearest")
        p = np.clip(p, 0.0, 1.0)

    mask = p >= float(args.threshold)
    if args.close_iters > 0:
        mask = binary_closing(mask, iterations=int(args.close_iters))
    if args.open_iters > 0:
        mask = binary_opening(mask, iterations=int(args.open_iters))
    if args.min_voxels > 0:
        mask = remove_small_objects(mask, min_size=int(args.min_voxels))

    if args.keep_largest_k > 0:
        lab, n = label(mask)
        if n > 0:
            sizes = np.bincount(lab.ravel())
            sizes[0] = 0
            keep_labels = np.argsort(sizes)[-int(args.keep_largest_k) :]
            keep = np.zeros_like(mask, dtype=bool)
            for lid in keep_labels:
                if lid > 0:
                    keep |= lab == lid
            mask = keep

    zsl, ysl, xsl = _parse_roi(args.roi, mask.shape)
    roi_mask = np.zeros_like(mask, dtype=bool)
    roi_mask[zsl, ysl, xsl] = True
    mask &= roi_mask

    out_bin = (mask.astype(np.uint8) * 255).astype(np.uint8)
    base = os.path.splitext(os.path.basename(args.in_tiff))[0]
    tiff_path = os.path.join(args.out_dir, f"{base}_bin_t{args.threshold:.2f}.tif")
    tifffile.imwrite(tiff_path, out_bin, imagej=True)

    _maxproj_png(
        os.path.join(args.out_dir, f"{base}_before_maxproj.png"),
        p,
        f"Before threshold (max-Z), thr={args.threshold:.2f}",
    )
    _maxproj_png(
        os.path.join(args.out_dir, f"{base}_after_maxproj.png"),
        out_bin,
        "After cleanup (max-Z)",
    )

    print(f"Saved: {tiff_path}")
    print(f"shape={out_bin.shape}, foreground_voxels={int(mask.sum())}")


if __name__ == "__main__":
    main()

