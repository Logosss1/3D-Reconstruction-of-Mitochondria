"""Helpers for large 3D volumes: random training crops and inference resize."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def random_crop_bounds(
    depth: int, height: int, width: int, crop_d: int, crop_h: int, crop_w: int
) -> tuple[int, int, int]:
    """Top-left corner (z0, y0, x0) for a crop of size (crop_d, crop_h, crop_w)."""
    cd = min(crop_d, depth)
    ch = min(crop_h, height)
    cw = min(crop_w, width)
    z0 = int(np.random.randint(0, max(1, depth - cd + 1)))
    y0 = int(np.random.randint(0, max(1, height - ch + 1)))
    x0 = int(np.random.randint(0, max(1, width - cw + 1)))
    return z0, y0, x0


def crop_volume(
    arr: np.ndarray, z0: int, y0: int, x0: int, cd: int, ch: int, cw: int
) -> np.ndarray:
    return np.ascontiguousarray(arr[z0 : z0 + cd, y0 : y0 + ch, x0 : x0 + cw])


def downsample_for_encoder(
    x: torch.Tensor, max_spatial: int
) -> tuple[torch.Tensor, tuple[int, int, int]]:
    """
    x: (B, C, D, H, W). If max(D,H,W) > max_spatial, resize so max side == max_spatial.
    Returns resized tensor and original (D, H, W).
    """
    if x.dim() != 5:
        raise ValueError("expected 5D tensor (B, C, D, H, W)")
    _, _, d, h, w = x.shape
    orig = (d, h, w)
    m = max(d, h, w)
    if m <= max_spatial:
        return x, orig
    scale = max_spatial / float(m)
    nd = max(1, int(round(d * scale)))
    nh = max(1, int(round(h * scale)))
    nw = max(1, int(round(w * scale)))
    y = F.interpolate(x, size=(nd, nh, nw), mode="trilinear", align_corners=True)
    return y, orig
