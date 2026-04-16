"""Export Hela2 / Hela3 mito crops (np_s0_with_data.csv) to Zarr v2: raw, label, raw_masked.

Default data: data/hela2.zarr/jrc_hela-2.zarr or data/hela3.zarr/jrc_hela-3.zarr (auto if present).

Examples::

    python export_hela2_np_s0_mito_masked.py --dataset jrc_hela-2 --all
    python export_hela2_np_s0_mito_masked.py --dataset jrc_hela-3 --all

Consolidated store: --zarr_root "D:/path/to/jrc_hela-3.zarr"
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from crop_zarr_export import export_one
from src.hela2_zarr_crop import dataset_zarr_root


def crop_ids_from_np_s0(np_csv: Path, cell: str) -> list[int]:
    cell = cell.strip()
    ids: list[int] = []
    with np_csv.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["cell"].strip() != cell:
                continue
            crop = row["crop"].strip()
            if not crop.lower().startswith("crop"):
                raise ValueError(f"Bad crop field: {crop!r}")
            ids.append(int(crop[4:]))
    return ids


def _default_out_dir(data_root: str, dataset: str) -> str:
    if dataset.endswith("hela-2"):
        sub = "hela2"
    elif dataset.endswith("hela-3"):
        sub = "hela3"
    else:
        sub = dataset.replace("jrc_", "").replace("-", "_")
    return os.path.join(data_root, f"crop_exports_{sub}_mito_bg")


def _auto_zarr_root(data_root: str, dataset: str) -> str | None:
    """data/{hela2|hela3}.zarr/{dataset}.zarr if that directory exists."""
    if dataset.endswith("hela-2"):
        top = "hela2.zarr"
    elif dataset.endswith("hela-3"):
        top = "hela3.zarr"
    else:
        return None
    cand = os.path.join(os.path.abspath(data_root), top, f"{dataset}.zarr")
    return cand if os.path.isdir(cand) else None


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Hela2/Hela3 mito crops from np_s0 -> Zarr v2 (raw, label, raw_masked)."
    )
    ap.add_argument("--data_root", type=str, default="data")
    ap.add_argument(
        "--dataset",
        type=str,
        default="jrc_hela-2",
        help="Matches np_s0 'cell' column, e.g. jrc_hela-3.",
    )
    ap.add_argument(
        "--np_s0_csv",
        type=str,
        default=os.path.join("data", "manifests", "np_s0_with_data.csv"),
    )
    ap.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Default: data/crop_exports_hela2_mito_bg or ..._hela3_... from --dataset.",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=3,
        help="First N ids in CSV order. Ignored with --crop_ids or --all.",
    )
    ap.add_argument(
        "--crop_ids",
        type=str,
        default=None,
        help="Comma-separated crop ids for this dataset in np_s0.",
    )
    ap.add_argument("--all", action="store_true", help="Export all np_s0 rows for this dataset.")
    ap.add_argument("--store_bucket", type=str, default=None)
    ap.add_argument(
        "--zarr_root",
        type=str,
        default=None,
        help="Dataset root .zarr (has recon-1). Default: auto from data/hela{N}.zarr.",
    )
    args = ap.parse_args()

    dataset = args.dataset.strip()
    data_root = os.path.abspath(args.data_root)
    out_dir = args.out_dir if args.out_dir is not None else _default_out_dir(data_root, dataset)
    zarr_root = args.zarr_root or _auto_zarr_root(data_root, dataset)

    store_root = dataset_zarr_root(
        data_root, dataset, store_bucket=args.store_bucket, zarr_root=zarr_root
    )
    if not os.path.isdir(store_root):
        ap.error(
            f"Dataset Zarr root missing: {store_root}\n"
            f"Download Open Organelle jrc_hela-3 into data/hela3.zarr/jrc_hela-3.zarr "
            f"or pass --zarr_root /path/to/jrc_hela-3.zarr"
        )

    allowed = crop_ids_from_np_s0(Path(args.np_s0_csv), dataset)
    if not allowed:
        ap.error(f"No rows for cell={dataset!r} in {args.np_s0_csv}")
    allowed_set = set(allowed)

    if args.crop_ids is not None:
        ids = [int(x.strip()) for x in args.crop_ids.split(",") if x.strip()]
        bad = [i for i in ids if i not in allowed_set]
        if bad:
            ap.error(f"crop ids not in np_s0 for {dataset}: {bad}; allowed: {allowed}")
    elif args.all:
        ids = list(allowed)
    else:
        ids = allowed[: args.limit]

    os.makedirs(out_dir, exist_ok=True)
    print(f"dataset={dataset} zarr_root={zarr_root!r}")
    print(f"Exporting {len(ids)} crop(s): {ids}")
    for cid in ids:
        out = os.path.join(out_dir, f"{dataset}_crop{cid}_mito.zarr")
        export_one(
            data_root,
            dataset,
            cid,
            "mito",
            out,
            store_bucket=args.store_bucket,
            zarr_root=zarr_root,
            label_as_binary=True,
            mask_foreground=True,
            instance_id=None,
        )


if __name__ == "__main__":
    main()
