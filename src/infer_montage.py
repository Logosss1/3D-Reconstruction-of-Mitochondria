"""Build a single figure from per-crop Thesis_Final_Result.png (joint model batch infer)."""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np


def write_joint_montage(
    output_dir: str,
    crop_ids: list[int],
    out_filename: str = "joint_infer_montage.png",
    dpi: int = 200,
    title: str = "Joint model: per-crop 3D reconstruction (Thesis_Final_Result style)",
) -> str | None:
    """
    Grid layout: up to 4 columns, one row of titles. Uses crop{cid}_Thesis_Final_Result.png.
    Returns path to saved PNG or None if matplotlib unavailable / no images.
    """
    paths = []
    for cid in crop_ids:
        p = os.path.join(output_dir, f"crop{cid}_Thesis_Final_Result.png")
        paths.append((cid, p))

    n = len(paths)
    if n == 0:
        return None

    ncols = min(4, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 4.2 * nrows))
    axes_arr = np.atleast_1d(axes).ravel()

    for i, (cid, pth) in enumerate(paths):
        ax = axes_arr[i]
        if os.path.isfile(pth):
            img = mpimg.imread(pth)
            ax.imshow(img)
            ax.set_title(f"crop {cid}", fontsize=11)
        else:
            ax.text(0.5, 0.5, f"missing\n{pth}", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(f"crop {cid} (missing)", fontsize=11)
        ax.axis("off")

    for j in range(n, len(axes_arr)):
        axes_arr[j].axis("off")

    fig.suptitle(title, fontsize=13, y=1.02)
    plt.tight_layout()
    out_path = os.path.join(output_dir, out_filename)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path
