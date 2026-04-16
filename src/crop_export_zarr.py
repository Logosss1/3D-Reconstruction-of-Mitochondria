"""Load per-crop Zarr from export_hela2_np_s0_mito_masked (raw + label on mito s0 grid)."""

from __future__ import annotations

import os

import numpy as np
import zarr


def crop_export_zarr_path(export_dir: str, dataset: str, crop_id: int) -> str:
    """Path to jrc_hela-2_crop{id}_mito.zarr under crop_exports_*_mito_bg."""
    return os.path.join(
        os.path.abspath(export_dir), f"{dataset}_crop{crop_id}_mito.zarr"
    )


def load_crop_export_volume(
    export_dir: str, dataset: str, crop_id: int, raw_key: str = "raw"
) -> tuple[np.ndarray, np.ndarray]:
    path = crop_export_zarr_path(export_dir, dataset, crop_id)
    if not os.path.isdir(path):
        raise FileNotFoundError(
            f"Missing export Zarr: {path}\n"
            f"Run: python export_hela2_np_s0_mito_masked.py --dataset {dataset} --all"
        )
    root = zarr.open(path, mode="r")
    if raw_key not in root:
        raise KeyError(f"{path} has no array {raw_key!r}; keys: {list(root.keys())}")
    raw = np.asarray(root[raw_key])
    label = (np.asarray(root["label"]) > 0).astype(np.uint8, copy=False)
    if raw.shape != label.shape:
        raise RuntimeError(f"raw {raw.shape} != label {label.shape} in {path}")
    return raw, label


def load_crop_export_label_only(
    export_dir: str, dataset: str, crop_id: int
) -> np.ndarray:
    """Binary mito mask from exported Zarr (for validation)."""
    path = crop_export_zarr_path(export_dir, dataset, crop_id)
    if not os.path.isdir(path):
        raise FileNotFoundError(f"Missing export Zarr: {path}")
    label = np.asarray(zarr.open(path, mode="r")["label"])
    return (label > 0).astype(np.uint8, copy=False)
