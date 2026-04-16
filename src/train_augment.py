"""Train-time augmentation on 3D EM subcrops: optional geometry (rot90+flip) + intensity (contrast, noise, gamma, percentile stretch)."""

from __future__ import annotations

import numpy as np


def augment_spatial_3d(
    raw_np: np.ndarray,
    label_np: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Same rigid transform on raw (uint8) and label (float) so supervision stays aligned.
    Random 90° rotations in a plane + independent axis flips.
    """
    if raw_np.shape != label_np.shape:
        raise ValueError("raw and label must share shape")
    r = np.asarray(raw_np)
    lab = np.asarray(label_np, dtype=np.float32)
    axes_pair = tuple(rng.choice(np.array([(0, 1), (0, 2), (1, 2)])))
    k = int(rng.integers(0, 4))
    if k:
        r = np.rot90(r, k=k, axes=axes_pair)
        lab = np.rot90(lab, k=k, axes=axes_pair)
    for ax in range(3):
        if rng.random() < 0.5:
            r = np.flip(r, axis=ax)
            lab = np.flip(lab, axis=ax)
    return np.ascontiguousarray(r.astype(np.uint8, copy=False)), np.ascontiguousarray(lab)


def _percentile_stretch(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Random 1–99 style intensity stretch (histogram expansion)."""
    xf = x.astype(np.float32)
    lo_p = float(rng.uniform(0.5, 2.0))
    hi_p = float(rng.uniform(98.0, 99.5))
    lo, hi = np.percentile(xf, [lo_p, hi_p])
    if hi <= lo + 1e-3:
        return x
    y = (xf - lo) / (hi - lo) * 255.0
    return np.clip(y, 0.0, 255.0)


def augment_intensity_uint8(
    raw_np: np.ndarray,
    rng: np.random.Generator,
    *,
    contrast_scale: float,
    noise_std: float,
    gamma_span: float,
    percentile_stretch: bool,
) -> np.ndarray:
    """
    contrast_scale: 0 = skip mean-centered contrast scaling.
    gamma_span: 0 = skip; else gamma ~ U(1-s, 1+s) on normalized intensities.
    percentile_stretch: optional random percentile stretch (executed with ~50% prob if True).
    """
    x = raw_np.astype(np.float32)
    if percentile_stretch and rng.random() < 0.5:
        x = _percentile_stretch(x, rng)
    if gamma_span > 0:
        g = float(rng.uniform(1.0 - gamma_span, 1.0 + gamma_span))
        g = float(np.clip(g, 0.25, 4.0))
        xn = np.clip(x / 255.0, 1e-6, 1.0)
        x = np.power(xn, g) * 255.0
    if contrast_scale > 0:
        a = float(rng.uniform(1.0 - contrast_scale, 1.0 + contrast_scale))
        m = float(x.mean())
        x = a * (x - m) + m
    if noise_std > 0:
        x = x + rng.normal(0.0, noise_std, size=x.shape).astype(np.float32)
    return np.clip(x, 0.0, 255.0).astype(np.uint8)


def augment_raw_uint8(
    raw_np: np.ndarray,
    rng: np.random.Generator,
    *,
    contrast_scale: float,
    noise_std: float,
) -> np.ndarray:
    """Legacy intensity-only path (contrast + noise)."""
    return augment_intensity_uint8(
        raw_np,
        rng,
        contrast_scale=contrast_scale,
        noise_std=noise_std,
        gamma_span=0.0,
        percentile_stretch=False,
    )
