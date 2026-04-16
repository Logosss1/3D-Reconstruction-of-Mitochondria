"""Quantitative comparison: predicted mesh vs binary mito label (voxel IoU/Dice, surface distance)."""

from __future__ import annotations

import numpy as np
import scipy.ndimage as ndi
from scipy.spatial import cKDTree
import trimesh


def _boundary_points_z_y_x(label_bin: np.ndarray, max_points: int = 50000) -> np.ndarray:
    """Surface voxels of foreground (Z,Y,X indices)."""
    fg = label_bin.astype(bool)
    if not fg.any():
        return np.zeros((0, 3), dtype=np.float64)
    er = ndi.binary_erosion(fg)
    b = fg & ~er
    pts = np.argwhere(b)
    if len(pts) > max_points:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(pts), size=max_points, replace=False)
        pts = pts[idx]
    return pts.astype(np.float64)


def align_mesh_centroid_to_label(mesh: trimesh.Trimesh, label_bin: np.ndarray) -> trimesh.Trimesh:
    """Shift mesh so its vertex centroid matches foreground centroid of label."""
    m = mesh.copy()
    coords = np.argwhere(label_bin > 0)
    if len(coords) == 0:
        return m
    lc = coords.mean(axis=0)
    mc = m.vertices.mean(axis=0)
    m.vertices = m.vertices + (lc - mc)
    return m


def voxel_metrics_from_mesh(mesh: trimesh.Trimesh, label_bin: np.ndarray, dilate: int = 2) -> dict[str, float]:
    """
    Rasterize mesh vertices into label grid (with optional dilation), vs GT binary mask.
    """
    shape = label_bin.shape
    m = align_mesh_centroid_to_label(mesh, label_bin)
    occ = np.zeros(shape, dtype=np.uint8)
    for v in m.vertices:
        iz, iy, ix = int(np.round(v[0])), int(np.round(v[1])), int(np.round(v[2]))
        if 0 <= iz < shape[0] and 0 <= iy < shape[1] and 0 <= ix < shape[2]:
            occ[iz, iy, ix] = 1
    if dilate > 0:
        occ = ndi.binary_dilation(occ > 0, iterations=dilate).astype(np.uint8)
    pred = occ.astype(bool)
    gt = label_bin.astype(bool)
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    iou = float(inter / max(union, 1))
    dice = float(2 * inter / max(pred.sum() + gt.sum(), 1))
    return {"voxel_iou": iou, "voxel_dice": dice}


def chamfer_mean_mesh_to_gt_surface(mesh: trimesh.Trimesh, label_bin: np.ndarray, max_points: int = 50000) -> float:
    """Mean distance from mesh vertices to nearest GT boundary voxel (one-sided Chamfer term)."""
    b = _boundary_points_z_y_x(label_bin, max_points=max_points)
    if len(b) == 0:
        return float("nan")
    m = align_mesh_centroid_to_label(mesh, label_bin)
    tree = cKDTree(b)
    d, _ = tree.query(m.vertices, k=1)
    return float(np.mean(d))


def compute_mesh_label_metrics(mesh: trimesh.Trimesh, label_bin: np.ndarray) -> dict[str, float]:
    out = voxel_metrics_from_mesh(mesh, label_bin)
    out["chamfer_mean_pred_to_gt_vox"] = chamfer_mean_mesh_to_gt_surface(mesh, label_bin)
    return out
