"""Export one OpenOrganelle crop to a small Zarr v2 store (raw + label [+ raw_masked]).

Output uses the Zarr **2** spec (.zgroup / .zarray) so Fiji/ImageJ N5-Zarr plugins can open it.
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
from typing import Iterable

import numpy as np
import zarr

from src.hela2_zarr_crop import load_crop_em_label_aligned


def _chunk_shape(shape: tuple[int, ...], max_chunk: int = 128) -> tuple[int, ...]:
    z, y, x = shape
    return (min(max_chunk, int(z)), min(max_chunk, int(y)), min(max_chunk, int(x)))


def write_crop_zarr(
    out_path: str,
    raw: np.ndarray,
    label: np.ndarray,
    raw_masked: np.ndarray | None = None,
) -> None:
    """Write a Zarr **v2** hierarchy (.zgroup / .zarray) for Fiji / ImageJ N5-Zarr plugins."""
    ch = _chunk_shape(raw.shape)
    if os.path.lexists(out_path):
        shutil.rmtree(out_path)

    major = int(zarr.__version__.partition(".")[0])
    if major >= 3:
        root = zarr.open_group(out_path, mode="w", zarr_format=2)
        root.create_array("raw", data=raw, chunks=ch)
        root.create_array("label", data=label, chunks=ch)
        if raw_masked is not None:
            root.create_array("raw_masked", data=raw_masked, chunks=ch)
    else:
        store = zarr.DirectoryStore(out_path)
        root = zarr.group(store=store, overwrite=True)
        root.create_dataset("raw", data=raw, chunks=ch, dtype=raw.dtype)
        root.create_dataset("label", data=label, chunks=ch, dtype=label.dtype)
        if raw_masked is not None:
            root.create_dataset("raw_masked", data=raw_masked, chunks=ch, dtype=raw_masked.dtype)


def _load_np_crop_filter(np_csv: str) -> set[tuple[str, int]]:
    out: set[tuple[str,int]] = set()
    with open(np_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            cell = row["cell"].strip()
            crop = row["crop"].strip()
            if not crop.lower().startswith("crop"):
                raise ValueError(f"Unexpected crop field (expected cropNNN): {crop!r}")
            cid = int(crop[4:])
            out.add((cell, cid))
    return out


def _iter_manifest(
    manifest_csv: str,
    dataset: str | None,
    class_label: str | None,
    np_filter: set[tuple[str, int]] | None,
) -> Iterable[tuple[int, str, str]]:
    with open(manifest_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            ds = row["dataset"].strip()
            if dataset and ds != dataset:
                continue
            cl = row["class_label"].strip()
            if class_label and cl != class_label:
                continue
            cid = int(row["crop_name"].strip())
            if np_filter is not None and (ds, cid) not in np_filter:
                continue
            yield cid, ds, cl


def export_one(
    data_root: str,
    dataset: str,
    crop_id: int,
    organ: str,
    out_path: str,
    *,
    store_bucket: str | None,
    zarr_root: str | None,
    label_as_binary: bool,
    mask_foreground: bool,
    instance_id: int | None,
) -> None:
    if instance_id is not None:
        label_as_binary = False
    raw, label = load_crop_em_label_aligned(
        data_root,
        dataset,
        crop_id,
        organ=organ,
        label_as_binary=label_as_binary,
        store_bucket=store_bucket,
        zarr_root=zarr_root,
    )
    raw_masked: np.ndarray | None = None
    if instance_id is not None:
        m = label == instance_id
        raw_masked = np.where(m, raw, 0).astype(np.uint8, copy=False)
    elif mask_foreground:
        raw_masked = (raw.astype(np.uint16) * (label > 0).astype(np.uint16)).astype(np.uint8)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    write_crop_zarr(out_path, raw, label, raw_masked)
    print(f"Wrote {out_path} raw={raw.shape} dtype={raw.dtype}")


def run() -> None:
    p = argparse.ArgumentParser(
        description=(
            "Crop EM + organ labels to a Zarr aligned on the organ s0 grid "
            "(see src/hela2_zarr_crop). Optional raw_masked removes background via "
            "organelle foreground or a specific instance id."
        )
    )
    p.add_argument("--data_root", type=str, default="data")
    p.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Bucket basename without .zarr, e.g. jrc_hela-2. Single-crop default: jrc_hela-2. Omit in batch mode to include all manifest datasets.",
    )
    p.add_argument("--crop_id", type=int, default=None, help="Single crop number (omit with --manifest).")
    p.add_argument(
        "--organ",
        type=str,
        default="mito",
        help="Label group folder under cropNNN, e.g. mito, er_mem_all, mito_ribo.",
    )
    p.add_argument(
        "--store_bucket",
        type=str,
        default=None,
        help="Top Zarr folder under data_root (e.g. hela2.zarr, cos7.zarr). Default: hela2/hela3 from dataset name.",
    )
    p.add_argument(
        "--zarr_root",
        type=str,
        default=None,
        help=(
            "Path to dataset root .zarr (contains recon-1), same as zarr.open(zarr_root). "
            "Use for consolidated layout e.g. .../jrc_hela-2/jrc_hela-2.zarr. Overrides hela2.zarr/... nesting."
        ),
    )
    p.add_argument("--out", type=str, default=None, help="Output DirectoryStore path for Zarr.")
    p.add_argument(
        "--keep_label_ids",
        action="store_true",
        help="Do not binarize labels to {0,1}; keep stored values as uint8.",
    )
    p.add_argument(
        "--mask_foreground",
        action="store_true",
        help="Also write raw_masked = raw * (label>0) (organelle foreground, not whole cell).",
    )
    p.add_argument(
        "--instance_id",
        type=int,
        default=None,
        help="If set, write raw_masked keeping voxels where label == instance_id (implies --keep_label_ids).",
    )
    p.add_argument("--manifest", type=str, default=None, help="train_crop_manifest.csv for batch export.")
    p.add_argument(
        "--class_label",
        type=str,
        default=None,
        help="With --manifest: filter rows where class_label matches (also used as organ folder name).",
    )
    p.add_argument(
        "--np_s0_csv",
        type=str,
        default=None,
        help="Optional np_s0_with_data.csv: only export (dataset, crop) pairs listed there.",
    )
    p.add_argument(
        "--out_dir",
        type=str,
        default="data/crop_exports",
        help="Batch mode: directory for <dataset>_crop<id>_<organ>.zarr",
    )
    args = p.parse_args()

    np_filter = _load_np_crop_filter(args.np_s0_csv) if args.np_s0_csv else None

    if args.manifest:
        if args.class_label is None:
            p.error("--manifest requires --class_label (organ name matching manifest column).")
        os.makedirs(args.out_dir, exist_ok=True)
        seen: set[tuple[str, int, str]] = set()
        for cid, ds, cl in _iter_manifest(args.manifest, args.dataset, args.class_label, np_filter):
            key = (ds, cid, cl)
            if key in seen:
                continue
            seen.add(key)
            out_name = f"{ds}_crop{cid}_{cl}.zarr"
            out_path = os.path.join(args.out_dir, out_name)
            export_one(
                args.data_root,
                ds,
                cid,
                cl,
                out_path,
                store_bucket=args.store_bucket,
                zarr_root=args.zarr_root,
                label_as_binary=not args.keep_label_ids,
                mask_foreground=args.mask_foreground,
                instance_id=args.instance_id,
            )
        return

    if args.crop_id is None:
        p.error("Provide --crop_id or --manifest.")
    if args.out is None:
        p.error("Single-crop mode requires --out")
    ds = args.dataset or "jrc_hela-2"
    export_one(
        args.data_root,
        ds,
        args.crop_id,
        args.organ,
        args.out,
        store_bucket=args.store_bucket,
        zarr_root=args.zarr_root,
        label_as_binary=not args.keep_label_ids,
        mask_foreground=args.mask_foreground,
        instance_id=args.instance_id,
    )


if __name__ == "__main__":
    run()
