"""
Load one OpenOrganelle crop: mito labels + fibsem EM aligned to the label grid.

Layout (Open Organelle default):
  {data_root}/hela2.zarr/jrc_hela-2.zarr/recon-1/em/fibsem-uint8/s0
  {data_root}/hela2.zarr/jrc_hela-2.zarr/recon-1/labels/groundtruth/crop{cid}/mito/s0

Consolidated layout (as in zarr.open(".../jrc_hela-2.zarr")):
  {zarr_root}/recon-1/...  (pass zarr_root= full path to the dataset .zarr directory)

Mito s0 voxel size (e.g. 2.62x2x2 nm) differs from fibsem s0 (e.g. 5.24x4x4 nm);
EM is read in its native grid and resampled to match label shape using scipy.ndimage.zoom.
"""

from __future__ import annotations

import json
import os
from typing import Tuple

import numpy as np
import zarr
from scipy.ndimage import zoom


def _resolve_top_bucket(dataset: str, store_bucket: str | None) -> str:
    if store_bucket is not None:
        return store_bucket
    return "hela2.zarr" if dataset.endswith("hela-2") else "hela3.zarr"


def dataset_zarr_root(
    data_root: str,
    dataset: str,
    *,
    store_bucket: str | None = None,
    zarr_root: str | None = None,
) -> str:
    """
    Path to the dataset Zarr group that contains `recon-1` (same object you pass to zarr.open(..., mode='r')).
    If `zarr_root` is set, it wins (absolute or relative path to .../jrc_hela-2.zarr).
    """
    if zarr_root is not None:
        root = os.path.abspath(os.path.expanduser(zarr_root))
        if not os.path.isdir(root):
            raise FileNotFoundError(f"zarr_root is not a directory: {root}")
        return root
    top = _resolve_top_bucket(dataset, store_bucket)
    return os.path.join(data_root, top, f"{dataset}.zarr")


def open_dataset_group(zarr_root: str):
    """Open dataset root like test_zarr: zarr.open(zarr_root, mode='r')."""
    return zarr.open(zarr_root, mode="r")


def label_s0_path(
    data_root: str,
    dataset: str,
    crop_id: int,
    organ: str = "mito",
    label_root: str = "labels/groundtruth",
    store_bucket: str | None = None,
    zarr_root: str | None = None,
) -> str:
    """
    organ: Zarr group name under crop, e.g. mito, nuc, er_mem_all (match OpenOrganelle folders).
    """
    grp = dataset_zarr_root(data_root, dataset, store_bucket=store_bucket, zarr_root=zarr_root)
    return os.path.join(grp, "recon-1", label_root, f"crop{crop_id}", organ, "s0")


def mito_s0_path(
    data_root: str,
    dataset: str,
    crop_id: int,
    label_root: str = "labels/groundtruth",
    store_bucket: str | None = None,
    zarr_root: str | None = None,
) -> str:
    return label_s0_path(
        data_root,
        dataset,
        crop_id,
        organ="mito",
        label_root=label_root,
        store_bucket=store_bucket,
        zarr_root=zarr_root,
    )


def raw_s0_path(
    data_root: str,
    dataset: str,
    recon_level: str = "recon-1",
    store_bucket: str | None = None,
    zarr_root: str | None = None,
) -> str:
    grp = dataset_zarr_root(data_root, dataset, store_bucket=store_bucket, zarr_root=zarr_root)
    return os.path.join(grp, recon_level, "em", "fibsem-uint8", "s0")


def load_fibsem_s0_scale_nm(
    data_root: str,
    dataset: str,
    recon_level: str = "recon-1",
    store_bucket: str | None = None,
    zarr_root: str | None = None,
) -> Tuple[float, float, float]:
    grp = dataset_zarr_root(data_root, dataset, store_bucket=store_bucket, zarr_root=zarr_root)
    path = os.path.join(grp, recon_level, "em", "fibsem-uint8", ".zattrs")
    with open(path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    for ds in meta["multiscales"][0]["datasets"]:
        if ds.get("path") != "s0":
            continue
        for tr in ds.get("coordinateTransformations", []):
            if tr.get("type") == "scale":
                s = tr["scale"]
                return float(s[0]), float(s[1]), float(s[2])
    raise ValueError(f"No s0 scale in {path}")


def _read_label_group_affine(label_s0_dir: str) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    """label_s0_path points to .../{organ}/s0; parent of s0 is .../{organ}."""
    organ_group = os.path.dirname(label_s0_dir)
    attrs_path = os.path.join(organ_group, ".zattrs")
    with open(attrs_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    vz = vy = vx = None
    tz = ty = tx = None
    for ds in meta["multiscales"][0]["datasets"]:
        if ds.get("path") != "s0":
            continue
        for tr in ds.get("coordinateTransformations", []):
            if tr.get("type") == "scale":
                s = tr["scale"]
                vz, vy, vx = float(s[0]), float(s[1]), float(s[2])
            if tr.get("type") == "translation":
                t = tr["translation"]
                tz, ty, tx = float(t[0]), float(t[1]), float(t[2])
        break
    if None in (vz, vy, vx, tz, ty, tx):
        raise ValueError(f"Missing s0 scale/translation in {attrs_path}")
    return (vz, vy, vx), (tz, ty, tx)


def _read_mito_s0_affine(mito_s0_dir: str) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    return _read_label_group_affine(mito_s0_dir)


def _label_indices_to_em_span(
    g0: float,
    g1: float,
    label_nm: float,
    em_nm: float,
) -> Tuple[int, int]:
    em0 = int(np.floor(g0 * label_nm / em_nm))
    em1 = int(np.ceil(g1 * label_nm / em_nm))
    return em0, max(em0 + 1, em1)


def read_raw_window_padded(
    raw_z,
    z0: int,
    y0: int,
    x0: int,
    dz: int,
    dy: int,
    dx: int,
    fill: int = 0,
) -> np.ndarray:
    Zm, Ym, Xm = (int(raw_z.shape[i]) for i in range(3))
    out = np.full((dz, dy, dx), fill, dtype=np.uint8)
    z1, y1, x1 = z0 + dz, y0 + dy, x0 + dx
    sz0, sy0, sx0 = max(0, z0), max(0, y0), max(0, x0)
    sz1, sy1, sx1 = min(Zm, z1), min(Ym, y1), min(Xm, x1)
    if sz0 >= sz1 or sy0 >= sy1 or sx0 >= sx1:
        return out
    src = np.asarray(raw_z[sz0:sz1, sy0:sy1, sx0:sx1])
    dz_off, dy_off, dx_off = sz0 - z0, sy0 - y0, sx0 - x0
    out[
        dz_off : dz_off + src.shape[0],
        dy_off : dy_off + src.shape[1],
        dx_off : dx_off + src.shape[2],
    ] = src.astype(np.uint8, copy=False)
    return out


def load_crop_em_label_aligned(
    data_root: str,
    dataset: str,
    crop_id: int,
    organ: str = "mito",
    label_as_binary: bool = True,
    store_bucket: str | None = None,
    zarr_root: str | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    EM resampled to the annotation s0 grid for `organ`.
    Returns (raw uint8, label as uint8 0/1 or int/uint instance IDs).
    """
    ls0 = label_s0_path(
        data_root, dataset, crop_id, organ=organ, store_bucket=store_bucket, zarr_root=zarr_root
    )
    if not os.path.isdir(ls0):
        raise FileNotFoundError(f"Missing path: {ls0}")

    label_src = np.asarray(zarr.open(ls0, mode="r"))
    dz, dy, dx = label_src.shape

    (vz, vy, vx), (tz, ty, tx) = _read_label_group_affine(ls0)
    z0g = tz / vz
    y0g = ty / vy
    x0g = tx / vx

    ez, ey, ex = load_fibsem_s0_scale_nm(
        data_root, dataset, store_bucket=store_bucket, zarr_root=zarr_root
    )

    raw_z0, raw_z1 = _label_indices_to_em_span(z0g, z0g + dz, vz, ez)
    raw_y0, raw_y1 = _label_indices_to_em_span(y0g, y0g + dy, vy, ey)
    raw_x0, raw_x1 = _label_indices_to_em_span(x0g, x0g + dx, vx, ex)
    raw_dz = raw_z1 - raw_z0
    raw_dy = raw_y1 - raw_y0
    raw_dx = raw_x1 - raw_x0

    raw_zr = zarr.open(
        raw_s0_path(data_root, dataset, store_bucket=store_bucket, zarr_root=zarr_root), mode="r"
    )
    raw_patch = read_raw_window_padded(raw_zr, raw_z0, raw_y0, raw_x0, raw_dz, raw_dy, raw_dx, fill=0)

    if raw_patch.shape != (dz, dy, dx):
        zf = dz / max(raw_dz, 1)
        yf = dy / max(raw_dy, 1)
        xf = dx / max(raw_dx, 1)
        raw_patch = zoom(raw_patch.astype(np.float32), (zf, yf, xf), order=1)
        raw_patch = np.clip(np.round(raw_patch), 0, 255).astype(np.uint8)

    if label_as_binary:
        label_out = (label_src > 0).astype(np.uint8, copy=False)
    else:
        label_out = np.asarray(label_src, copy=False)
        if label_out.dtype.kind in "iu" and label_out.max() <= 255 and label_out.min() >= 0:
            label_out = np.asarray(label_out, dtype=np.uint8)
        else:
            label_out = np.asarray(label_out, dtype=np.int32)

    if raw_patch.shape != label_out.shape:
        raise RuntimeError(f"raw {raw_patch.shape} != label {label_out.shape}")

    return raw_patch, label_out


def load_crop_em_mito_aligned(
    data_root: str,
    dataset: str,
    crop_id: int,
    label_as_binary: bool = True,
    store_bucket: str | None = None,
    zarr_root: str | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    return load_crop_em_label_aligned(
        data_root,
        dataset,
        crop_id,
        organ="mito",
        label_as_binary=label_as_binary,
        store_bucket=store_bucket,
        zarr_root=zarr_root,
    )
