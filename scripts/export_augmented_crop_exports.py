"""
Export augmented versions of crop-export Zarrs (raw + label) into new directories.

This implements the "strong" augmentation strategy: treat augmented volumes as extra volumes
in the mixed training pool (original + augmented are sampled across steps).

Modes:
  - contrast: mean-centered contrast only
  - stretch: percentile stretch only
  - geo: rot90+flip only (raw+label)
  - gamma: gamma only
  - noise: additive noise only
  - combo: geo + (contrast + noise + gamma + stretch)
  - all: generate the 6 modes above (writes 6 output folders)

Example (Hela3, all modes):
  python scripts/export_augmented_crop_exports.py --dataset jrc_hela-3 --in_dir data/crop_exports_hela3_mito_bg --mode all

Example (Hela2, one mode):
  python scripts/export_augmented_crop_exports.py --dataset jrc_hela-2 --in_dir data/crop_exports_hela2_mito_bg --mode geo --out_dir data/crop_exports_hela2_mito_bg_aug_geo
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import zarr

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)


def _copy_array(root_in, root_out, name: str, data: np.ndarray) -> None:
    arr_in = root_in[name]
    chunks = getattr(arr_in, "chunks", None) or (128, 128, 128)
    compressor = getattr(arr_in, "compressor", None)
    dtype = arr_in.dtype
    root_out.create_dataset(
        name,
        shape=data.shape,
        chunks=chunks,
        compressor=compressor,
        dtype=dtype,
        overwrite=True,
    )[:] = data.astype(dtype, copy=False)


def main() -> None:
    ap = argparse.ArgumentParser(description="Export augmented crop-export Zarr volumes.")
    ap.add_argument("--dataset", type=str, required=True, help="e.g. jrc_hela-2 or jrc_hela-3")
    ap.add_argument("--in_dir", type=str, required=True, help="original crop_exports_*_mito_bg")
    ap.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=("contrast", "stretch", "geo", "gamma", "noise", "combo", "all"),
        help="augmentation mode",
    )
    ap.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="output dir (required unless --mode all; otherwise defaults to <in_dir>_aug_<mode>)",
    )
    ap.add_argument("--seed", type=int, default=123, help="base RNG seed (per-crop derived)")
    ap.add_argument("--contrast_scale", type=float, default=0.12)
    ap.add_argument("--noise_std", type=float, default=4.0)
    ap.add_argument("--gamma_span", type=float, default=0.12)
    ap.add_argument("--force", action="store_true", help="overwrite existing outputs")
    args = ap.parse_args()

    from src.crop_export_zarr import crop_export_zarr_path, load_crop_export_volume
    from src.np_s0_manifest import np_s0_crop_ids_for_dataset
    from src.train_augment import augment_intensity_uint8, augment_spatial_3d

    manifest = os.path.join(_REPO, "data", "manifests", "np_s0_with_data.csv")
    crop_ids = np_s0_crop_ids_for_dataset(manifest, args.dataset)
    if not crop_ids:
        raise SystemExit(f"No crop ids for dataset={args.dataset!r} in {manifest}")

    modes = ["contrast", "stretch", "geo", "gamma", "noise", "combo"]
    run_modes = modes if args.mode == "all" else [args.mode]

    def out_dir_for_mode(m: str) -> str:
        if args.out_dir and args.mode != "all":
            return str(args.out_dir)
        return os.path.abspath(f"{args.in_dir}_aug_{m}")

    def flags_for_mode(m: str) -> tuple[bool, bool, bool, bool, bool]:
        # (geo, contrast, noise, gamma, stretch)
        if m == "contrast":
            return (False, True, False, False, False)
        if m == "stretch":
            return (False, False, False, False, True)
        if m == "geo":
            return (True, False, False, False, False)
        if m == "gamma":
            return (False, False, False, True, False)
        if m == "noise":
            return (False, False, True, False, False)
        if m == "combo":
            return (True, True, True, True, True)
        raise ValueError(m)

    for mode in run_modes:
        out_dir = out_dir_for_mode(mode)
        os.makedirs(out_dir, exist_ok=True)
        do_geo, do_con, do_noise, do_gamma, do_pct = flags_for_mode(mode)
        print(f"=== export mode={mode} -> {out_dir} ===")

        for cid in crop_ids:
            in_path = crop_export_zarr_path(args.in_dir, args.dataset, cid)
            out_path = crop_export_zarr_path(out_dir, args.dataset, cid)
            if os.path.exists(out_path):
                if args.force:
                    import shutil

                    shutil.rmtree(out_path)
                else:
                    print(f"Skip exists: {out_path}")
                    continue

            raw, lab = load_crop_export_volume(args.in_dir, args.dataset, cid, raw_key="raw")
            rng = np.random.default_rng(int(args.seed) + int(cid) * 1009 + hash(mode) % 10_000)
            lab_f = lab.astype(np.float32, copy=False)

            if do_geo:
                raw, lab_f = augment_spatial_3d(raw, lab_f, rng)

            raw_aug = augment_intensity_uint8(
                raw,
                rng,
                contrast_scale=float(args.contrast_scale) if do_con else 0.0,
                noise_std=float(args.noise_std) if do_noise else 0.0,
                gamma_span=float(args.gamma_span) if do_gamma else 0.0,
                percentile_stretch=bool(do_pct),
            )
            lab_aug = (lab_f > 0.5).astype(np.uint8)
            raw_masked = (raw_aug * (lab_aug > 0)).astype(np.uint8)

            root_in = zarr.open(in_path, mode="r")
            root_out = zarr.open(out_path, mode="w")
            _copy_array(root_in, root_out, "raw", raw_aug)
            _copy_array(root_in, root_out, "label", lab_aug)
            if "raw_masked" in root_in:
                _copy_array(root_in, root_out, "raw_masked", raw_masked)
            print(f"Wrote {mode} crop{cid} -> {out_path}")


if __name__ == "__main__":
    main()

