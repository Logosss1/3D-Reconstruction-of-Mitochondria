"""
Crop mitochondria-containing regions from jrc_hela-2 / jrc_hela-3
based on OpenOrganelle crop manifests.

Inputs (CSV on your Desktop, paths can be changed via CLI):
- train_crop_manifest.csv  : per-crop translation / shape / voxel_size
- np_s0_with_data.csv      : which (cell, crop) pairs are valid (has data)

Output Zarr structure (under data/hela_crops by default):
- data/hela_crops/jrc_hela-2/crop{N}.zarr
- data/hela_crops/jrc_hela-3/crop{N}.zarr
  each with datasets:
    - raw   : cropped EM volume
    - label : cropped foreground mask (from recon-1/labels/masks/foreground)
"""
from __future__ import annotations

import argparse
import csv
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import zarr


@dataclass
class CropSpec:
    crop_name: int
    dataset: str
    voxel_size: Tuple[float, float, float]  # (z, y, x) in nm
    translation: Tuple[float, float, float]  # (z, y, x) in nm
    shape: Tuple[int, int, int]  # (D, H, W) in voxels


def parse_bracket_vec(s: str) -> List[float]:
    """
    Parse strings like "[2.62;2.0;2.0]" into [2.62, 2.0, 2.0].
    """
    s = s.strip()
    if s.startswith("[") and s.endswith("]"):
        s = s[1:-1]
    parts = [p for p in s.split(";") if p]
    return [float(p) for p in parts]


def load_manifest_for_cell(
    manifest_path: Path,
    cell_name: str,
    valid_crops: List[int],
) -> Dict[int, CropSpec]:
    """
    For a given cell (e.g. jrc_hela-2), load crop specs from train_crop_manifest.csv,
    restricted to crop_name in valid_crops. Because each crop appears multiple times
    with different class_label but identical translation/shape/voxel_size, we only
    keep the first occurrence per crop_name.
    """
    specs: Dict[int, CropSpec] = {}
    valid_set = set(valid_crops)

    with manifest_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                ds = row["dataset"]
            except KeyError:
                # header mismatch
                continue
            if ds != cell_name:
                continue
            try:
                crop_id = int(row["crop_name"])
            except (KeyError, ValueError):
                continue
            if crop_id not in valid_set:
                continue
            if crop_id in specs:
                # already recorded one spec for this crop_name
                continue

            try:
                voxel_vec = parse_bracket_vec(row["voxel_size"])
                trans_vec = parse_bracket_vec(row["translation"])
                shape_vec = [int(v) for v in parse_bracket_vec(row["shape"])]
            except Exception as e:  # noqa: BLE001
                print(f"[WARN] Failed to parse row for crop {crop_id}: {e}")
                continue

            if len(voxel_vec) != 3 or len(trans_vec) != 3 or len(shape_vec) != 3:
                print(f"[WARN] Unexpected vector length for crop {crop_id}, skip.")
                continue

            specs[crop_id] = CropSpec(
                crop_name=crop_id,
                dataset=ds,
                voxel_size=(voxel_vec[0], voxel_vec[1], voxel_vec[2]),
                translation=(trans_vec[0], trans_vec[1], trans_vec[2]),
                shape=(shape_vec[0], shape_vec[1], shape_vec[2]),
            )

    return specs


def load_valid_crops(np_s0_path: Path, cell_name: str) -> List[int]:
    """
    From np_s0_with_data.csv, collect crop IDs for a given cell (jrc_hela-2 / -3).
    CSV format: cell,crop with crop like 'crop113'.
    """
    crops: List[int] = []
    with np_s0_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("cell") != cell_name:
                continue
            crop_str = row.get("crop", "")
            if crop_str.startswith("crop"):
                try:
                    cid = int(crop_str.replace("crop", ""))
                    crops.append(cid)
                except ValueError:
                    continue
    return sorted(set(crops))


def crop_from_zarr(
    zarr_root: Path,
    spec: CropSpec,
    out_dir: Path,
) -> None:
    """
    Use manifest info (translation / shape / voxel_size) to crop raw + foreground
    from a jrc_hela-2/3 zarr and write a small zarr dataset.
    """
    z = zarr.open(str(zarr_root), mode="r")

    # Paths follow OpenOrganelle convention used in your code.
    # For hela2 / hela3, foreground mask lives under:
    #   recon-1/labels/masks/foreground
    # Raw EM volume is usually under something like:
    #   recon-1/em/fibsem-uint8/s0
    # Adjust here if your dataset layout differs.
    try:
        raw_ds = z["recon-1/em/fibsem-uint8/s0"]
    except Exception:
        raise RuntimeError(f"Could not find raw volume in {zarr_root}")

    try:
        label_ds = z["recon-1/labels/masks/foreground"]
    except Exception:
        raise RuntimeError(f"Could not find foreground mask in {zarr_root}")

    # Compute voxel indices from translation / voxel_size.
    vz, vy, vx = spec.voxel_size
    tz, ty, tx = spec.translation
    D, H, W = spec.shape

    # Coordinates: approximate; manifest is in physical nm, volume voxel size nm.
    z0 = int(round(tz / vz))
    y0 = int(round(ty / vy))
    x0 = int(round(tx / vx))
    z1 = z0 + D
    y1 = y0 + H
    x1 = x0 + W

    # Clamp within volume bounds.
    Z, Y, X = raw_ds.shape
    z0 = max(0, min(z0, Z))
    y0 = max(0, min(y0, Y))
    x0 = max(0, min(x0, X))
    z1 = max(0, min(z1, Z))
    y1 = max(0, min(y1, Y))
    x1 = max(0, min(x1, X))

    if z1 <= z0 or y1 <= y0 or x1 <= x0:
        print(f"[WARN] Empty crop box for crop {spec.crop_name}, skip.")
        return

    print(
        f"  crop{spec.crop_name}: Z[{z0}:{z1}] Y[{y0}:{y1}] X[{x0}:{x1}] "
        f"from {zarr_root.name}"
    )

    raw_crop = raw_ds[z0:z1, y0:y1, x0:x1]
    label_crop = label_ds[z0:z1, y0:y1, x0:x1]

    out_dir.mkdir(parents=True, exist_ok=True)
    out_store = zarr.DirectoryStore(str(out_dir / f"crop{spec.crop_name}.zarr"))
    root = zarr.group(store=out_store, overwrite=True)
    root.create_dataset("raw", data=raw_crop, chunks=True)
    root.create_dataset("label", data=label_crop, chunks=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Crop mitochondria-containing patches from hela2/hela3 using manifest."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(r"c:\Users\yf\Desktop\train_crop_manifest.csv"),
        help="Path to train_crop_manifest.csv",
    )
    parser.add_argument(
        "--np-s0",
        type=Path,
        default=Path(r"c:\Users\yf\Desktop\np_s0_with_data.csv"),
        help="Path to np_s0_with_data.csv",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data"),
        help="Root directory containing hela2.zarr / hela3.zarr",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path("data/hela_crops"),
        help="Output root for cropped zarrs.",
    )
    args = parser.parse_args()

    cells = ["jrc_hela-2", "jrc_hela-3"]
    zarr_map = {
        "jrc_hela-2": args.data_root / "hela2.zarr" / "jrc_hela-2.zarr",
        "jrc_hela-3": args.data_root / "hela3.zarr" / "jrc_hela-3.zarr",
    }

    for cell in cells:
        print(f"=== Processing {cell} ===")
        valid_crops = load_valid_crops(args.np_s0, cell)
        if not valid_crops:
            print(f"  No valid crops listed for {cell}, skip.")
            continue
        print(f"  valid crops from np_s0_with_data: {valid_crops}")

        specs = load_manifest_for_cell(args.manifest, cell, valid_crops)
        if not specs:
            print(f"  No specs found in manifest for {cell}, skip.")
            continue

        zarr_root = zarr_map[cell]
        if not zarr_root.exists():
            print(f"  Zarr root not found: {zarr_root}, skip.")
            continue

        out_dir = args.out_root / cell
        for cid in sorted(specs.keys()):
            crop_from_zarr(zarr_root, specs[cid], out_dir)

    print("Done.")


if __name__ == "__main__":
    main()

