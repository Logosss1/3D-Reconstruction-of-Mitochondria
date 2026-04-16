from __future__ import annotations

import os

import torch
import numpy as np
from skimage import measure
import trimesh


def _print_occupancy_field_diag(probs: np.ndarray) -> None:
    """Inference-only: summarize sigmoid occupancy on the mc_resolution^3 grid."""
    flat = probs.ravel().astype(np.float64, copy=False)
    mean_p = float(flat.mean())
    std_p = float(flat.std())
    min_p = float(flat.min())
    max_p = float(flat.max())
    f05 = float((flat > 0.5).mean())
    f07 = float((flat > 0.7).mean())
    f09 = float((flat > 0.9).mean())
    print(
        "Occupancy field (mc grid): "
        f"mean={mean_p:.4f} std={std_p:.4f} min={min_p:.4f} max={max_p:.4f} | "
        f"frac>0.5={f05:.4f} frac>0.7={f07:.4f} frac>0.9={f09:.4f}"
    )
    if f05 > 0.85:
        print(
            "  hint: most voxels >0.5 — strong sign of occupancy collapse (foreground everywhere). "
            "Prefer training fixes (class balance, focal/weighted BCE) over only tuning MC level."
        )
    elif f05 > 0.6 and mean_p > 0.55:
        print(
            "  hint: field skewed high — consider more negative query points / focal loss / longer training."
        )


def _marching_cubes_level(
    probs: np.ndarray,
    *,
    level_mode: str,
    fixed_level: float,
    percentile: float,
    adaptive_floor: float,
) -> float:
    """
    level_mode:
      - fixed: use fixed_level (e.g. 0.5), independent of max_p
      - percentile: np.percentile(probs, percentile) — pushes surface toward high-confidence regions
      - adaptive: legacy max(adaptive_floor, max_p * 0.5)
    """
    flat = probs.ravel()
    min_p = float(flat.min())
    max_p = float(flat.max())
    mean_p = float(flat.mean())
    frac_mid = float((flat > 0.5).mean())

    if level_mode == "fixed":
        level = float(fixed_level)
    elif level_mode == "percentile":
        p = float(np.clip(percentile, 0.0, 100.0))
        level = float(np.percentile(flat, p))
    elif level_mode == "adaptive":
        level = max(float(adaptive_floor), max_p * 0.5)
    else:
        raise ValueError(f"Unknown level_mode: {level_mode!r}")

    print(
        f"Marching cubes: mode={level_mode} level={level:.4f} | "
        f"prob min={min_p:.4f} max={max_p:.4f} mean={mean_p:.4f} frac>0.5={frac_mid:.4f}"
    )
    if level < min_p or level > max_p:
        print(
            f"  warning: iso-level {level:.4f} outside prob range [{min_p:.4f}, {max_p:.4f}] "
            "(will clamp before marching cubes)"
        )
    return level


def _save_prob_binary_preview(
    probs: np.ndarray,
    path: str,
    *,
    threshold: float = 0.5,
) -> None:
    """Max-projection along Z of continuous probs vs binarized (>= threshold)."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipped binary/prob preview PNG.")
        return

    thr = float(threshold)
    binary = (probs >= thr).astype(np.float32)
    p_max = np.max(probs, axis=0)
    b_any = np.max(binary, axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    im0 = axes[0].imshow(p_max, cmap="magma", vmin=0.0, vmax=1.0, aspect="auto")
    axes[0].set_title("max over Z: P(occ)")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    axes[1].imshow(b_any, cmap="gray_r", vmin=0.0, vmax=1.0, aspect="auto")
    axes[1].set_title(f"max over Z: binary (P≥{thr})")
    for ax in axes:
        ax.set_xlabel("mc grid")
        ax.set_ylabel("mc grid")
    plt.suptitle("Occupancy diagnostics (mc grid, before mesh)")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved prob/binary preview: {path}")


def _clamp_marching_cubes_level(probs: np.ndarray, level: float) -> float:
    """
    skimage.measure.marching_cubes requires the iso-level to lie strictly within
    [min(volume), max(volume)]. When the model saturates (e.g. all probs > 0.5),
    a fixed level of 0.5 is invalid — clamp into the interior of the observed range.
    """
    lo = float(np.min(probs))
    hi = float(np.max(probs))
    span = hi - lo
    if span <= 0:
        print("  warning: constant probability field; marching cubes may still fail")
        return lo
    eps = max(1e-7, 1e-4 * span)
    lo_b, hi_b = lo + eps, hi - eps
    if lo_b >= hi_b:
        mid = 0.5 * (lo + hi)
        print(f"  note: ultra-narrow prob band; using iso-level {mid:.6f}")
        return mid
    if level < lo_b or level > hi_b:
        adj = float(np.clip(level, lo_b, hi_b))
        print(
            f"  note: iso-level {level:.6f} -> {adj:.6f} "
            f"(required inside ({lo:.6f}, {hi:.6f}))"
        )
        return adj
    return float(level)


def _strip_boundary_faces(
    verts: np.ndarray,
    faces: np.ndarray,
    shape_dhw: tuple[int, int, int],
    margin_vox: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Drop triangles that touch the outer shell of the axis-aligned volume box (Z,Y,X).
    Reduces marching-cubes 'wall' sheets along z=0,z=max, y=0, etc. in voxel space.
    """
    if margin_vox <= 0:
        return verts, faces
    d, h, w = int(shape_dhw[0]), int(shape_dhw[1]), int(shape_dhw[2])
    m = float(margin_vox)
    if min(d, h, w) < 2 * m + 1:
        print(
            f"  warning: mesh_strip_boundary_voxels={margin_vox} too large for shape {(d, h, w)}; skip strip"
        )
        return verts, faces
    zmax, ymax, xmax = float(d - 1), float(h - 1), float(w - 1)
    z, y, x = verts[:, 0], verts[:, 1], verts[:, 2]
    inside = (
        (z >= m)
        & (z <= zmax - m + 1e-9)
        & (y >= m)
        & (y <= ymax - m + 1e-9)
        & (x >= m)
        & (x <= xmax - m + 1e-9)
    )
    ok = inside[faces].all(axis=1)
    n0, nf = len(faces), int(ok.sum())
    faces_kept = faces[ok]
    if nf == 0:
        return verts, faces
    used = np.unique(faces_kept.ravel())
    remap = -np.ones(len(verts), dtype=np.int64)
    remap[used] = np.arange(len(used), dtype=np.int64)
    verts_new = verts[used]
    faces_new = remap[faces_kept.astype(np.int64)]
    print(f"Mesh boundary strip (margin={margin_vox} vox): kept {nf}/{n0} faces, {len(verts_new)} verts")
    return verts_new, faces_new


def _save_prob_volume_tiff(probs: np.ndarray, path: str) -> None:
    """
    Save 3D occupancy grid as a single multi-page TIFF for Fiji/ImageJ (Image › Stacks).
    Values are float32 in [0,1]; axes follow the internal MC grid (same order as marching_cubes input).
    """
    try:
        import tifffile
    except ImportError:
        print(
            "  warning: tifffile not installed; cannot save TIFF. "
            "Install with: pip install tifffile"
        )
        return
    d = os.path.dirname(os.path.abspath(path))
    if d:
        os.makedirs(d, exist_ok=True)
    vol = np.ascontiguousarray(probs.astype(np.float32))
    tifffile.imwrite(path, vol, imagej=True)
    print(f"Saved MC P(occ) volume TIFF for Fiji: {path} shape={tuple(vol.shape)} dtype=float32")


def _save_prob_volume_tiff_binary(
    probs: np.ndarray, path: str, threshold: float
) -> None:
    """Binarized MC grid: uint8 0 / 255 for Fiji (true binary mask)."""
    try:
        import tifffile
    except ImportError:
        print(
            "  warning: tifffile not installed; cannot save binary TIFF. "
            "Install with: pip install tifffile"
        )
        return
    d = os.path.dirname(os.path.abspath(path))
    if d:
        os.makedirs(d, exist_ok=True)
    thr = float(threshold)
    mask = ((probs >= thr).astype(np.uint8)) * 255
    mask = np.ascontiguousarray(mask)
    tifffile.imwrite(path, mask, imagej=True)
    print(
        f"Saved MC binary mask TIFF for Fiji: {path} shape={tuple(mask.shape)} "
        f"dtype=uint8 (P>={thr})"
    )


def _strip_unit_cube_shell(
    verts: np.ndarray,
    faces: np.ndarray,
    margin_norm: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Drop triangles with any vertex in the outer shell of the [0,1]^3 MC grid (before scaling to D,H,W).
    margin_norm: distance from each face of the unit cube (e.g. 3/127 for 3 MC cells at res=128).
    Large crops then lose ~margin_norm*(D-1) voxels from each side — removes inference-cube 'walls' evenly.
    """
    if margin_norm <= 0:
        return verts, faces
    hi = 1.0 - margin_norm
    if hi <= margin_norm + 1e-9:
        print("  warning: MC shell margin too large; skip unit-cube strip")
        return verts, faces
    v = verts
    ins = (
        (v[:, 0] >= margin_norm)
        & (v[:, 0] <= hi + 1e-12)
        & (v[:, 1] >= margin_norm)
        & (v[:, 1] <= hi + 1e-12)
        & (v[:, 2] >= margin_norm)
        & (v[:, 2] <= hi + 1e-12)
    )
    ok = ins[faces].all(axis=1)
    n0, nf = len(faces), int(ok.sum())
    fk = faces[ok]
    if nf == 0:
        return verts, faces
    used = np.unique(fk.ravel())
    remap = -np.ones(len(verts), dtype=np.int64)
    remap[used] = np.arange(len(used), dtype=np.int64)
    print(
        f"Mesh MC-shell strip (margin_norm={margin_norm:.5f}): kept {nf}/{n0} faces, {len(used)} verts"
    )
    return verts[used], remap[fk.astype(np.int64)]


def _extract_mesh_from_model(
    model,
    inputs,
    resolution=128,
    original_voxel_shape=None,
    *,
    level_mode: str = "fixed",
    fixed_level: float = 0.5,
    percentile: float = 90.0,
    adaptive_floor: float = 0.2,
    strip_boundary_voxels: float = 0.0,
    strip_mc_shell_cells: float = 0.0,
    mc_prob_smooth_sigma: float = 0.0,
    binary_preview_path: str | None = None,
    mc_binary_threshold: float = 0.5,
    prob_tiff_path: str | None = None,
    prob_tiff_binary_path: str | None = None,
):
    """
    original_voxel_shape: optional (D, H, W) of the *source* volume before any
    inference resize. Vertices are mapped into that voxel frame when provided.
    """
    grid = np.linspace(0, 1, resolution)
    xv, yv, zv = np.meshgrid(grid, grid, grid, indexing="ij")
    query_points = np.stack([xv.flatten(), yv.flatten(), zv.flatten()], axis=-1)

    query_points_torch = torch.from_numpy(query_points).float().to(inputs.device).unsqueeze(0)

    model.eval()
    with torch.no_grad():
        logits = model(inputs, query_points_torch)
        probs = torch.sigmoid(logits).cpu().numpy().reshape(resolution, resolution, resolution)

    if mc_prob_smooth_sigma and mc_prob_smooth_sigma > 0:
        from scipy.ndimage import gaussian_filter

        probs = gaussian_filter(
            probs, sigma=float(mc_prob_smooth_sigma), mode="nearest"
        )
        probs = np.clip(probs, 0.0, 1.0)
        print(
            f"MC prob grid: gaussian_filter sigma={mc_prob_smooth_sigma} (mode=nearest)"
        )

    _print_occupancy_field_diag(probs)

    if binary_preview_path:
        _save_prob_binary_preview(
            probs,
            binary_preview_path,
            threshold=mc_binary_threshold,
        )

    if prob_tiff_path:
        _save_prob_volume_tiff(probs, prob_tiff_path)

    if prob_tiff_binary_path:
        _save_prob_volume_tiff_binary(
            probs, prob_tiff_binary_path, threshold=mc_binary_threshold
        )

    current_threshold = _marching_cubes_level(
        probs,
        level_mode=level_mode,
        fixed_level=fixed_level,
        percentile=percentile,
        adaptive_floor=adaptive_floor,
    )
    current_threshold = _clamp_marching_cubes_level(probs, current_threshold)

    try:
        verts, faces, _, _ = measure.marching_cubes(probs, level=current_threshold)

        verts = verts / max(resolution - 1, 1)

        if strip_mc_shell_cells > 0:
            mn = float(strip_mc_shell_cells) / max(resolution - 1, 1)
            v0, f0 = verts.copy(), faces.copy()
            verts, faces = _strip_unit_cube_shell(verts, faces, mn)
            if len(faces) == 0:
                print("  warning: MC-shell strip removed all faces; using unstripped mesh")
                verts, faces = v0, f0

        if original_voxel_shape is not None:
            d, h, w = original_voxel_shape
            verts = verts * np.array([d - 1, h - 1, w - 1], dtype=np.float64)
            if strip_boundary_voxels > 0:
                v0, f0 = verts.copy(), faces.copy()
                verts, faces = _strip_boundary_faces(verts, faces, (d, h, w), strip_boundary_voxels)
                if len(faces) == 0:
                    print("  warning: boundary strip removed all faces; using unstripped mesh")
                    verts, faces = v0, f0
            center = verts.mean(axis=0)
            verts = verts - center
        else:
            center = verts.mean(axis=0)
            verts = verts - center
            verts = verts * 100.0

        mesh = trimesh.Trimesh(vertices=verts, faces=faces)
        trimesh.repair.fix_normals(mesh)
        return mesh

    except Exception as e:
        print(f"Mesh extraction failed: {e}")
        return trimesh.creation.uv_sphere(radius=10.0)


def reconstruction_pipeline(
    model,
    inputs,
    raw_data,
    original_voxel_shape=None,
    resolution=128,
    *,
    level_mode: str = "fixed",
    fixed_level: float = 0.5,
    percentile: float = 90.0,
    adaptive_floor: float = 0.2,
    strip_boundary_voxels: float = 0.0,
    strip_mc_shell_cells: float = 0.0,
    mc_prob_smooth_sigma: float = 0.0,
    binary_preview_path: str | None = None,
    mc_binary_threshold: float = 0.5,
    prob_tiff_path: str | None = None,
    prob_tiff_binary_path: str | None = None,
):
    mesh = _extract_mesh_from_model(
        model,
        inputs,
        resolution=resolution,
        original_voxel_shape=original_voxel_shape,
        level_mode=level_mode,
        fixed_level=fixed_level,
        percentile=percentile,
        adaptive_floor=adaptive_floor,
        strip_boundary_voxels=strip_boundary_voxels,
        strip_mc_shell_cells=strip_mc_shell_cells,
        mc_prob_smooth_sigma=mc_prob_smooth_sigma,
        binary_preview_path=binary_preview_path,
        mc_binary_threshold=mc_binary_threshold,
        prob_tiff_path=prob_tiff_path,
        prob_tiff_binary_path=prob_tiff_binary_path,
    )
    return mesh, mesh