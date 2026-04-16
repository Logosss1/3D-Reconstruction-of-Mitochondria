"""
Export crop-export Zarr (raw + label) to 3D TIFF stacks for Fiji/ImageJ.

Example:
  python scripts/export_crop_zarr_to_tiff.py --crop_export_dir data/crop_exports_hela2_mito_bg --dataset jrc_hela-2 --crop_id 9 --out_dir result/tiff_export
"""
from __future__ import annotations

import argparse
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)


def main() -> None:
    ap = argparse.ArgumentParser(description="Zarr crop -> uint8 TIFF stacks (Fiji).")
    ap.add_argument("--crop_export_dir", type=str, required=True)
    ap.add_argument("--dataset", type=str, default="jrc_hela-2")
    ap.add_argument("--crop_id", type=int, required=True)
    ap.add_argument("--out_dir", type=str, default="result/tiff_export")
    args = ap.parse_args()

    try:
        import tifffile
    except ImportError:
        raise SystemExit("Install tifffile: pip install tifffile")

    from src.crop_export_zarr import load_crop_export_volume

    raw, lab = load_crop_export_volume(args.crop_export_dir, args.dataset, args.crop_id)
    os.makedirs(args.out_dir, exist_ok=True)
    tag = f"{args.dataset.replace('jrc_', '').replace('-', '')}_crop{args.crop_id}"
    raw_path = os.path.join(args.out_dir, f"{tag}_raw_uint8.tif")
    lab_path = os.path.join(args.out_dir, f"{tag}_label_uint8.tif")

    tifffile.imwrite(raw_path, raw.astype("uint8"), imagej=True)
    tifffile.imwrite(lab_path, ((lab > 0).astype("uint8") * 255), imagej=True)
    print(f"Wrote {raw_path} shape={raw.shape}")
    print(f"Wrote {lab_path} shape={lab.shape}")


if __name__ == "__main__":
    main()
