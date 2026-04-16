"""Fixed-seed validation batch for train.py (same subcrop + same query pattern each epoch)."""

from __future__ import annotations

import numpy as np
import torch

from src.volume_utils import crop_volume


def pick_val_subcrop_with_fg(
    raw_vol: np.ndarray,
    label_vol: np.ndarray,
    ps_step: int,
    rng: np.random.RandomState,
    max_tries: int = 256,
) -> tuple[np.ndarray, np.ndarray] | None:
    depth, height, width = raw_vol.shape
    cd = min(ps_step, depth)
    ch = min(ps_step, height)
    cw = min(ps_step, width)
    for _ in range(max_tries):
        z0 = int(rng.randint(0, max(1, depth - cd + 1)))
        y0 = int(rng.randint(0, max(1, height - ch + 1)))
        x0 = int(rng.randint(0, max(1, width - cw + 1)))
        label_np = crop_volume(label_vol, z0, y0, x0, cd, ch, cw)
        if label_np.max() > 0:
            raw_np = crop_volume(raw_vol, z0, y0, x0, cd, ch, cw)
            return raw_np, label_np
    return None


def tensors_for_bce_dice(
    raw_np: np.ndarray,
    label_np: np.ndarray,
    num_points: int,
    fg_query_fraction: float,
    device: torch.device,
    rng: np.random.RandomState,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    raw = torch.from_numpy(raw_np).float().unsqueeze(0).unsqueeze(0).to(device) / 255.0
    label = torch.from_numpy(label_np).float().unsqueeze(0).unsqueeze(0).to(device)
    d, h, w = label.shape[-3:]
    coord_scale = torch.tensor([max(d, 1), max(h, 1), max(w, 1)], device=device, dtype=torch.float32)

    pos_indices = torch.from_numpy(np.argwhere(label_np > 0)).float().to(device)
    fg_frac = float(min(0.95, max(0.05, fg_query_fraction)))
    n_fg = max(1, int(round(num_points * fg_frac)))
    n_bg = max(0, num_points - n_fg)
    if n_fg > len(pos_indices):
        n_fg = len(pos_indices)
        n_bg = num_points - n_fg
    idx = torch.from_numpy(rng.randint(0, len(pos_indices), size=(n_fg,))).long().to(device)
    p_pos = pos_indices[idx] / coord_scale
    p_rand = torch.from_numpy(rng.uniform(0.0, 1.0, size=(n_bg, 3))).float().to(device)
    points = torch.cat([p_pos, p_rand], dim=0).unsqueeze(0)

    iz = torch.clamp((points[0, :, 0] * d).long(), 0, d - 1)
    ih = torch.clamp((points[0, :, 1] * h).long(), 0, h - 1)
    iw = torch.clamp((points[0, :, 2] * w).long(), 0, w - 1)
    target = label[0, 0, iz, ih, iw].unsqueeze(0)
    return raw, points, target


def build_fixed_val_batch(
    raw_vol: np.ndarray,
    label_vol: np.ndarray,
    encoder_spatial: int,
    num_points: int,
    fg_query_fraction: float,
    val_seed: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    """One fixed subcrop + fixed query RNG stream (reproducible each epoch)."""
    depth, height, width = raw_vol.shape
    ps_step = min(encoder_spatial, depth, height, width)
    rng_bounds = np.random.RandomState(val_seed)
    rng_pts = np.random.RandomState(val_seed + 17_000_000)
    pair = pick_val_subcrop_with_fg(raw_vol, label_vol, ps_step, rng_bounds)
    if pair is None:
        return None
    raw_np, label_np = pair
    return tensors_for_bce_dice(
        raw_np, label_np, num_points, fg_query_fraction, device, rng_pts
    )
