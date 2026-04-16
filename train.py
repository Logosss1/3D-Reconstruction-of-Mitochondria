from __future__ import annotations

import argparse
import csv
import os
import random
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import zarr

from src.model import ConvONet
from src.volume_utils import crop_volume, random_crop_bounds


def _append_val_log_csv(path: str, row: dict) -> None:
    fieldnames = list(row.keys())
    new_file = not os.path.isfile(path)
    d = os.path.dirname(os.path.abspath(path))
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if new_file:
            w.writeheader()
        w.writerow(row)


def _append_train_log_csv(path: str, row: dict) -> None:
    fieldnames = list(row.keys())
    new_file = not os.path.isfile(path)
    d = os.path.dirname(os.path.abspath(path))
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if new_file:
            w.writeheader()
        w.writerow(row)


def dice_loss(pred, target):
    pred = torch.sigmoid(pred)
    smooth = 1e-5
    intersect = (pred * target).sum()
    return 1 - (2.0 * intersect + smooth) / (pred.sum() + target.sum() + smooth)


def train():
    parser = argparse.ArgumentParser(description="Train ConvONet on Zarr raw/label volumes.")
    parser.add_argument(
        "--zarr",
        type=str,
        default=None,
        help="Zarr group with datasets 'raw' and 'label'. If unset, uses hela2.zarr (--data_root, --crop_id).",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="data",
        help="Contains hela2.zarr/ for OpenOrganelle layout.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="jrc_hela-2",
        help="Bucket name jrc_hela-2.zarr under hela2.zarr.",
    )
    parser.add_argument(
        "--crop_id",
        type=int,
        default=94,
        help="Single training crop (ignored if --crop_ids is set).",
    )
    parser.add_argument(
        "--crop_ids",
        type=str,
        default=None,
        help=(
            "Comma-separated crop ids for multi-volume training: each step randomly picks one crop "
            "(same schedule as single-crop). Example: 1,3,6,9,23,113,155 (np_s0 hela2 mito)."
        ),
    )
    parser.add_argument(
        "--zarr_root",
        type=str,
        default=None,
        help="Optional path to jrc_hela-2.zarr root (contains recon-1); same as export script.",
    )
    parser.add_argument(
        "--crop_export_dir",
        type=str,
        default=None,
        help=(
            "Directory with {dataset}_crop{id}_mito.zarr (raw, label from export_hela2_np_s0_mito_masked). "
            "When set, skips Open Organelle hela2.zarr paths."
        ),
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default=None,
        help=(
            "Multi-domain crop-export training: comma-separated datasets (e.g. jrc_hela-2,jrc_hela-3). "
            "Each step samples a random volume from the union of np_s0 manifest crops. "
            "Requires --crop_export_dirs or default folders under --data_root. Incompatible with --zarr."
        ),
    )
    parser.add_argument(
        "--crop_export_dirs",
        type=str,
        default=None,
        help="Comma-separated export folders, same order as --datasets (mixed training only).",
    )
    parser.add_argument(
        "--encoder_spatial",
        type=int,
        default=128,
        help="Random 3D crop size for encoder (e.g. 128; full volume may be 400^3).",
    )
    parser.add_argument(
        "--native_required",
        action="store_true",
        help=(
            "Strict native training flag. Training already uses native-voxel crops (no resampling). "
            "This flag documents/locks that behavior in scripts."
        ),
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=300,
    )
    parser.add_argument(
        "--num_points",
        type=int,
        default=10000,
        help="Query points per gradient step (split by --fg_query_fraction). Larger => heavier decoder/GPU.",
    )
    parser.add_argument(
        "--steps_per_epoch",
        type=int,
        default=1,
        help=(
            "Gradient updates per logged epoch (each uses a new random subcrop). "
            "Default 1 matches old behavior. Use 4–8 to keep the GPU busier (batch=1 ConvONet, "
            "so multiple steps per epoch amortize CPU/Zarr work)."
        ),
    )
    parser.add_argument(
        "--fg_query_fraction",
        type=float,
        default=0.5,
        help=(
            "Fraction of query points sampled on foreground (label>0); rest uniform in crop. "
            "Try 0.35–0.4 if logits saturate high (Hela2/Hela3)."
        ),
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=5e-4,
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.0,
        help="Adam weight decay (L2); try 1e-5 for slightly smoother val curves.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="checkpoints/model_final.pth",
    )
    parser.add_argument(
        "--train_log_csv",
        type=str,
        default=None,
        help="Append epoch, loss, volume tag each --train_log_every step (convergence / advisor review).",
    )
    parser.add_argument(
        "--train_log_every",
        type=int,
        default=1,
        help="Log every N epochs when --train_log_csv is set (default 1).",
    )
    parser.add_argument(
        "--val_every",
        type=int,
        default=0,
        help="Every N epochs run a fixed-seed val forward (BCE+Dice on held-out subcrop); 0=off.",
    )
    parser.add_argument(
        "--val_log_csv",
        type=str,
        default=None,
        help="Append val_loss / val_bce / val_dice per val step (default: <checkpoint_dir>/val_metrics.csv).",
    )
    parser.add_argument(
        "--val_num_points",
        type=int,
        default=4096,
        help="Query points for lightweight val (smaller => faster).",
    )
    parser.add_argument(
        "--val_seed",
        type=int,
        default=42,
        help="RNG seed for fixed val subcrop + query points (reproducible across epochs).",
    )
    parser.add_argument(
        "--aug_contrast",
        action="store_true",
        help="Random intensity scaling around mean (train only; advisor-style augmentation).",
    )
    parser.add_argument(
        "--aug_contrast_scale",
        type=float,
        default=0.1,
        help="With --aug_contrast, scale factor in [1-s, 1+s].",
    )
    parser.add_argument(
        "--aug_noise_std",
        type=float,
        default=0.0,
        help="Gaussian noise std on raw 0–255 (train only). Try 2–8.",
    )
    parser.add_argument(
        "--aug_prob",
        type=float,
        default=0.5,
        help="Probability to apply augmentation each step (if any aug enabled).",
    )
    parser.add_argument(
        "--aug_geometric",
        action="store_true",
        help="Train only: random 90° rotations + flips on raw+label subcrop (same transform).",
    )
    parser.add_argument(
        "--aug_gamma",
        type=float,
        default=0.0,
        help="Train only: if >0, random gamma in [1-span, 1+span] on normalized intensities (e.g. 0.12).",
    )
    parser.add_argument(
        "--aug_percentile_stretch",
        action="store_true",
        help="Train only: random percentile-based intensity stretch (histogram expansion).",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if args.native_required:
        print("Native required (train): using native voxel crops only (no resize/downsample).")

    use_single_zarr_file = args.zarr is not None
    crop_export_dir = os.path.abspath(args.crop_export_dir) if args.crop_export_dir else None

    datasets_list: list[str] = []
    if args.datasets:
        datasets_list = [s.strip() for s in args.datasets.split(",") if s.strip()]
    use_mixed = not use_single_zarr_file and len(datasets_list) >= 2

    if use_single_zarr_file and use_mixed:
        raise SystemExit("Cannot combine --zarr with multi --datasets.")

    _repo_root = os.path.dirname(os.path.abspath(__file__))
    _manifest_csv = os.path.join(_repo_root, "data", "manifests", "np_s0_with_data.csv")

    def _export_subdir_for_dataset(dataset: str) -> str:
        if dataset.endswith("hela-2"):
            return "crop_exports_hela2_mito_bg"
        if dataset.endswith("hela-3"):
            return "crop_exports_hela3_mito_bg"
        raise SystemExit(
            f"Mixed training: unsupported dataset {dataset!r} for auto export path "
            "(expected jrc_hela-2 / jrc_hela-3, or pass --crop_export_dirs)."
        )

    use_crop_exports = False
    use_hela2 = False
    crop_ids: list[int] = []
    pool: list[tuple[str, str, int]] = []
    vol_cache_mixed: dict[tuple[str, str, int], tuple[np.ndarray, np.ndarray]] = {}
    dirs_mixed: list[str] = []

    if use_single_zarr_file:
        crop_ids = [args.crop_id]
        print(f"Zarr: {args.zarr}")
        zroot = zarr.open(args.zarr, mode="r")
        raw_vol = np.asarray(zroot["raw"])
        label_vol = (np.asarray(zroot["label"]) > 0).astype(np.uint8)
        depth, height, width = raw_vol.shape
        print(f"Volume shape (Z,Y,X): {depth} x {height} x {width}")
    elif use_mixed:
        from src.crop_export_zarr import load_crop_export_volume
        from src.np_s0_manifest import np_s0_crop_ids_for_dataset

        if args.crop_export_dirs:
            dirs_mixed = [
                os.path.abspath(p.strip()) for p in args.crop_export_dirs.split(",") if p.strip()
            ]
            if len(dirs_mixed) != len(datasets_list):
                raise SystemExit(
                    f"--crop_export_dirs ({len(dirs_mixed)} paths) must match "
                    f"--datasets ({len(datasets_list)} entries)."
                )
        else:
            root = os.path.abspath(args.data_root)
            dirs_mixed = [os.path.join(root, _export_subdir_for_dataset(ds)) for ds in datasets_list]

        for d in dirs_mixed:
            if not os.path.isdir(d):
                raise SystemExit(f"Mixed training: missing crop export directory: {d}")

        for ds, ed in zip(datasets_list, dirs_mixed):
            if not os.path.isfile(_manifest_csv):
                raise SystemExit(f"Missing {_manifest_csv}")
            cids = np_s0_crop_ids_for_dataset(_manifest_csv, ds)
            if not cids:
                raise SystemExit(f"No np_s0 rows for dataset={ds!r} in {_manifest_csv}")
            for cid in cids:
                pool.append((ds, ed, cid))

        def get_vol_mixed(ds: str, ed: str, cid: int) -> tuple[np.ndarray, np.ndarray]:
            key = (ed, ds, cid)
            if key not in vol_cache_mixed:
                print(f"Loading {ds} crop {cid} ... ({os.path.basename(ed)})")
                vol_cache_mixed[key] = load_crop_export_volume(ed, ds, cid)
            return vol_cache_mixed[key]

        ds0, ed0, c0 = pool[0]
        raw_vol, label_vol = get_vol_mixed(ds0, ed0, c0)
        depth, height, width = raw_vol.shape
        print(
            f"Mixed-domain training: {len(datasets_list)} datasets, {len(pool)} volumes in pool; "
            f"first {ds0} crop{c0} shape (Z,Y,X): {depth} x {height} x {width}"
        )
    elif crop_export_dir is not None:
        use_crop_exports = True
        from src.crop_export_zarr import load_crop_export_volume

        if args.crop_ids:
            crop_ids = [int(x.strip()) for x in args.crop_ids.split(",") if x.strip()]
        else:
            crop_ids = [args.crop_id]
        if len(crop_ids) > 1:
            print(
                f"Multi-crop training (crop exports): {crop_ids} "
                f"(one random crop per step, volumes cached on first load)\n"
                f"  dir: {crop_export_dir}"
            )
        else:
            print(f"Source: crop export {crop_ids[0]} under {crop_export_dir}")

        vol_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}

        def get_volumes(cid: int) -> tuple[np.ndarray, np.ndarray]:
            if cid not in vol_cache:
                print(f"Loading crop {cid} ...")
                vol_cache[cid] = load_crop_export_volume(crop_export_dir, args.dataset, cid)
            return vol_cache[cid]

        raw_vol, label_vol = get_volumes(crop_ids[0])
        depth, height, width = raw_vol.shape
        print(f"First crop volume shape (Z,Y,X): {depth} x {height} x {width}")
    else:
        use_hela2 = True
        from src.hela2_zarr_crop import load_crop_em_mito_aligned

        if args.crop_ids:
            crop_ids = [int(x.strip()) for x in args.crop_ids.split(",") if x.strip()]
        else:
            crop_ids = [args.crop_id]
        if len(crop_ids) > 1:
            print(
                f"Multi-crop training: {crop_ids} (one random crop per step, volumes cached on first load)"
            )
        else:
            print(
                f"Source: dataset={args.dataset} crop={crop_ids[0]} "
                f"(mito label + fibsem-uint8/s0)"
            )

        vol_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}

        def get_volumes(cid: int) -> tuple[np.ndarray, np.ndarray]:
            if cid not in vol_cache:
                print(f"Loading crop {cid} ...")
                vol_cache[cid] = load_crop_em_mito_aligned(
                    args.data_root,
                    args.dataset,
                    cid,
                    zarr_root=args.zarr_root,
                )
            return vol_cache[cid]

        raw_vol, label_vol = get_volumes(crop_ids[0])
        depth, height, width = raw_vol.shape
        print(f"First crop volume shape (Z,Y,X): {depth} x {height} x {width}")

    print(
        f"Encoder crop: {args.encoder_spatial}^3 "
        f"(volume can be larger, e.g. 400^3)"
    )

    ps = min(args.encoder_spatial, depth, height, width)
    if ps < args.encoder_spatial:
        print(f"Note: encoder_spatial capped to {ps} to fit volume.")

    model = ConvONet().to(device)
    wd = float(getattr(args, "weight_decay", 0.0) or 0.0)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=wd)
    scaler: Any = None
    if wd > 0:
        print(f"Adam weight_decay={wd}")

    out_dir = os.path.dirname(os.path.abspath(args.out))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    do_val = args.val_every > 0
    val_batch = None
    val_log_path = args.val_log_csv
    if do_val:
        if not val_log_path:
            val_log_path = os.path.join(out_dir or ".", "val_metrics.csv")
        from src.train_val_batch import build_fixed_val_batch

        if use_mixed:
            ds_v, ed_v, cid_v = pool[0]
            v_raw, v_lab = get_vol_mixed(ds_v, ed_v, cid_v)
        elif (use_hela2 or use_crop_exports) and len(crop_ids) > 1:
            v_raw, v_lab = get_volumes(crop_ids[0])
        else:
            v_raw, v_lab = raw_vol, label_vol

        val_batch = build_fixed_val_batch(
            v_raw,
            v_lab,
            args.encoder_spatial,
            args.val_num_points,
            args.fg_query_fraction,
            args.val_seed,
            device,
        )
        if val_batch is None:
            print("warning: could not build fixed val batch; disabling --val_every.")
            do_val = False
        else:
            print(
                f"Lightweight val: every {args.val_every} epoch(s) -> {val_log_path} "
                f"(num_points={args.val_num_points}, seed={args.val_seed})"
            )

    aug_rng = np.random.default_rng()
    gamma_span = float(getattr(args, "aug_gamma", 0.0) or 0.0)
    aug_enabled = bool(
        args.aug_contrast
        or args.aug_noise_std > 0
        or getattr(args, "aug_geometric", False)
        or gamma_span > 0
        or getattr(args, "aug_percentile_stretch", False)
    )
    if aug_enabled:
        parts = []
        if args.aug_contrast:
            parts.append("contrast")
        if args.aug_noise_std > 0:
            parts.append("noise")
        if getattr(args, "aug_geometric", False):
            parts.append("rot90+flip")
        if gamma_span > 0:
            parts.append(f"gamma(span={gamma_span})")
        if getattr(args, "aug_percentile_stretch", False):
            parts.append("pct_stretch")
        print(f"Augmentations enabled: {', '.join(parts)} (prob={args.aug_prob})")

    if use_mixed:
        pass
    elif (use_hela2 or use_crop_exports) and len(crop_ids) > 1:
        pass
    else:
        pos_indices_full = np.argwhere(label_vol > 0)
        print(f"Foreground voxels (full volume): {len(pos_indices_full)}")

    spe = max(1, int(args.steps_per_epoch))
    if spe > 1:
        print(
            f"steps_per_epoch={spe}: each logged epoch runs {spe} gradient steps "
            f"(total optimizer steps ~ {args.epochs * spe})."
        )

    for epoch in range(1, args.epochs + 1):
        epoch_losses: list[float] = []
        last_z0 = last_y0 = last_x0 = 0
        tag = ""
        last_ds_pick = ""
        last_cid_pick = 0
        last_cid = 0

        for _step in range(spe):
            optimizer.zero_grad()

            if use_mixed:
                ds_pick, ed_pick, cid_pick = random.choice(pool)
                raw_vol, label_vol = get_vol_mixed(ds_pick, ed_pick, cid_pick)
                depth, height, width = raw_vol.shape
                ps_step = min(args.encoder_spatial, depth, height, width)
            elif (use_hela2 or use_crop_exports) and len(crop_ids) > 1:
                cid = random.choice(crop_ids)
                raw_vol, label_vol = get_volumes(cid)
                depth, height, width = raw_vol.shape
                ps_step = min(args.encoder_spatial, depth, height, width)
            else:
                ps_step = ps

            z0, y0, x0 = random_crop_bounds(depth, height, width, ps_step, ps_step, ps_step)
            raw_np = crop_volume(raw_vol, z0, y0, x0, ps_step, ps_step, ps_step)
            label_np = crop_volume(label_vol, z0, y0, x0, ps_step, ps_step, ps_step)

            if label_np.max() == 0:
                for _ in range(32):
                    z0, y0, x0 = random_crop_bounds(depth, height, width, ps_step, ps_step, ps_step)
                    label_np = crop_volume(label_vol, z0, y0, x0, ps_step, ps_step, ps_step)
                    if label_np.max() > 0:
                        raw_np = crop_volume(raw_vol, z0, y0, x0, ps_step, ps_step, ps_step)
                        break
                else:
                    continue

            ap = float(np.clip(args.aug_prob, 0.0, 1.0))
            if aug_enabled and aug_rng.random() < ap:
                from src.train_augment import augment_intensity_uint8, augment_spatial_3d

                if getattr(args, "aug_geometric", False):
                    raw_np, label_np = augment_spatial_3d(raw_np, label_np, aug_rng)
                cscale = float(args.aug_contrast_scale) if args.aug_contrast else 0.0
                raw_np = augment_intensity_uint8(
                    raw_np,
                    aug_rng,
                    contrast_scale=cscale,
                    noise_std=float(args.aug_noise_std),
                    gamma_span=gamma_span,
                    percentile_stretch=bool(getattr(args, "aug_percentile_stretch", False)),
                )

            raw = torch.from_numpy(raw_np).float().unsqueeze(0).unsqueeze(0).to(device) / 255.0
            label = torch.from_numpy(label_np).float().unsqueeze(0).unsqueeze(0).to(device)

            d, h, w = label.shape[-3:]
            coord_scale = torch.tensor([max(d, 1), max(h, 1), max(w, 1)], device=device, dtype=torch.float32)

            pos_indices = torch.from_numpy(np.argwhere(label_np > 0)).float().to(device)
            num_points = args.num_points
            fg_frac = float(min(0.95, max(0.05, args.fg_query_fraction)))
            n_fg = max(1, int(round(num_points * fg_frac)))
            n_bg = max(0, num_points - n_fg)
            if n_fg > len(pos_indices):
                n_fg = len(pos_indices)
                n_bg = num_points - n_fg
            idx = torch.randint(0, len(pos_indices), (n_fg,), device=device)
            p_pos = pos_indices[idx] / coord_scale
            p_rand = torch.rand(n_bg, 3, device=device)
            points = torch.cat([p_pos, p_rand], dim=0).unsqueeze(0)

            iz = torch.clamp((points[0, :, 0] * d).long(), 0, d - 1)
            ih = torch.clamp((points[0, :, 1] * h).long(), 0, h - 1)
            iw = torch.clamp((points[0, :, 2] * w).long(), 0, w - 1)
            target = label[0, 0, iz, ih, iw].unsqueeze(0)

            out = model(raw, points)
            loss = nn.BCEWithLogitsLoss()(out, target) + dice_loss(out, target)

            loss.backward()
            optimizer.step()

            li = float(loss.item())
            epoch_losses.append(li)
            last_z0, last_y0, last_x0 = z0, y0, x0
            if use_mixed:
                last_ds_pick, last_cid_pick = ds_pick, cid_pick
                tag = f"{ds_pick} crop{cid_pick} | "
            elif (use_hela2 or use_crop_exports) and len(crop_ids) > 1:
                last_cid = cid
                tag = f"crop{cid} | "

        if not epoch_losses:
            continue

        last_train_loss = epoch_losses[-1]
        mean_train_loss = sum(epoch_losses) / len(epoch_losses)

        if args.train_log_csv and epoch % max(1, args.train_log_every) == 0:
            vol_key = ""
            if use_mixed:
                vol_key = f"{last_ds_pick}:{last_cid_pick}"
            elif (use_hela2 or use_crop_exports) and len(crop_ids) > 1:
                vol_key = str(last_cid)
            _append_train_log_csv(
                args.train_log_csv,
                {
                    "epoch": epoch,
                    "loss": f"{mean_train_loss:.8f}",
                    "lr": args.lr,
                    "volume": vol_key,
                    "subcrop_z0": last_z0,
                    "subcrop_y0": last_y0,
                    "subcrop_x0": last_x0,
                },
            )

        if do_val and epoch % args.val_every == 0:
            model.eval()
            with torch.no_grad():
                v_raw, v_pts, v_tgt = val_batch
                v_out = model(v_raw, v_pts)
                v_bce = nn.BCEWithLogitsLoss()(v_out, v_tgt)
                v_dice = dice_loss(v_out, v_tgt)
                v_loss = v_bce + v_dice
            model.train()
            _append_val_log_csv(
                val_log_path,
                {
                    "epoch": epoch,
                    "val_loss": f"{float(v_loss):.8f}",
                    "val_bce": f"{float(v_bce):.8f}",
                    "val_dice_term": f"{float(v_dice):.8f}",
                    "train_loss_last": f"{mean_train_loss:.8f}",
                },
            )

        if epoch % 20 == 0:
            extra = ""
            if do_val and epoch % args.val_every == 0:
                extra = f" | val_loss={float(v_loss):.4f}"
            loss_show = mean_train_loss
            print(
                f"Epoch {epoch}/{args.epochs} | Loss: {loss_show:.4f}{extra} | {tag}subcrop @({last_z0},{last_y0},{last_x0})"
            )

    checkpoint = {
        "epoch": int(args.epochs),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "args": vars(args),
        "scaler": scaler,
    }
    torch.save(checkpoint, args.out)
    print(f"Done. Saved full checkpoint to {args.out}")


if __name__ == "__main__":
    train()
