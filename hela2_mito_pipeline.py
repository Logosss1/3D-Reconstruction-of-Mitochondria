"""
Hela2 / Hela3 mito training / inference (np_s0 crops from data/manifests/np_s0_with_data.csv).

- joint:       train one model on all crops for ``--dataset`` (default jrc_hela-2 ->
  ``checkpoints/model_hela2_all_mito.pth``; jrc_hela-3 -> ``model_hela3_all_mito.pth``).
- infer-all:   batch ``generate.py`` + ``joint_infer_montage.png``.
- per-crop:    train -> generate PNG/OBJ -> validate per crop
  (hela2: ``checkpoints/hela2_mito_per_crop``, ``result/hela2_mito_per_crop``;
  hela3 defaults: ``hela3_mito_*``).

One crop at a time (thesis runs)::

    python hela2_mito_pipeline.py per-crop --only-crop 9 --epochs 300

All np_s0 crops in one command (still processed sequentially)::

    python hela2_mito_pipeline.py per-crop --epochs 300

Skip validation::

    python hela2_mito_pipeline.py per-crop --no-validate

Use a conda/venv that has torch; pass interpreter explicitly::

    python hela2_mito_pipeline.py --python D:/miniconda3/envs/pt/python.exe per-crop --only-crop 9

Data: prefer sliced Zarrs under ``data/crop_exports_hela2_mito_bg`` (from
``export_hela2_np_s0_mito_masked.py``) when that folder exists; otherwise use
Open Organelle ``hela2.zarr`` and optional ``--zarr_root``. Override with
``--crop_export_dir`` or ``--open-organelle-data``.

Joint model batch infer + montage::

    python hela2_mito_pipeline.py joint --epochs 300
    python hela2_mito_pipeline.py joint --datasets jrc_hela-2,jrc_hela-3 --epochs 300
    python hela2_mito_pipeline.py --python <py> infer-all --zarr_root <path/to/jrc_hela-2.zarr>
    python hela2_mito_pipeline.py infer-all --checkpoint checkpoints/my_joint.pth --validate

One-shot Hela2+Hela3 mixed joint (needs both ``data/crop_exports_*_mito_bg``) writes
``checkpoints/model_hela2_hela3_mixed.pth`` by default when ``--out`` is still the Hela2 joint default.

Validation / advisor reporting::

    infer-all ... --validate  -> per-crop ``metrics_*_infer.csv`` **and**
    one summary row ``metrics_*_infer_summary.csv`` (mean ± std IoU/Dice/Chamfer over crops).
    Joint/mixed = **one model**; per-crop metrics show which FOVs are hard; the summary row is the
    headline number for that model on that dataset. Training convergence uses **loss** (optional
    ``--train_log_csv``), not IoU — run ``--validate`` after inference for task metrics.
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone

_REPO = os.path.dirname(os.path.abspath(__file__))
_TRAIN = os.path.join(_REPO, "train.py")
_GEN = os.path.join(_REPO, "generate.py")
_VAL = os.path.join(_REPO, "validate_crop.py")

_NP_S0_CSV = os.path.join(_REPO, "data", "manifests", "np_s0_with_data.csv")

# Default checkpoint / folder names (used with _apply_dataset_path_defaults for jrc_hela-3)
_DEF_JOINT_OUT_HELA2 = os.path.join("checkpoints", "model_hela2_all_mito.pth")
_DEF_JOINT_OUT_HELA3 = os.path.join("checkpoints", "model_hela3_all_mito.pth")
_DEF_PC_CKPT_DIR_HELA2 = os.path.join("checkpoints", "hela2_mito_per_crop")
_DEF_PC_CKPT_DIR_HELA3 = os.path.join("checkpoints", "hela3_mito_per_crop")
_DEF_PC_RESULT_HELA2 = os.path.join("result", "hela2_mito_per_crop")
_DEF_PC_RESULT_HELA3 = os.path.join("result", "hela3_mito_per_crop")
_DEF_INF_OUT_HELA2 = os.path.join("result", "hela2_mito_joint_infer")
_DEF_INF_OUT_HELA3 = os.path.join("result", "hela3_mito_joint_infer")
_DEF_JOINT_OUT_MIXED = os.path.join("checkpoints", "model_hela2_hela3_mixed.pth")


def _np_s0_crop_ids_for_dataset(dataset: str) -> list[int]:
    """All crop ids for `cell` column == dataset in np_s0_with_data.csv."""
    if not os.path.isfile(_NP_S0_CSV):
        raise SystemExit(f"Missing {_NP_S0_CSV}")
    from src.np_s0_manifest import np_s0_crop_ids_for_dataset

    try:
        return np_s0_crop_ids_for_dataset(_NP_S0_CSV, dataset)
    except ValueError as e:
        raise SystemExit(str(e))


def resolve_python(exe: str | None) -> str:
    """Resolve --python to an executable path."""
    if not exe:
        return sys.executable
    exe = os.path.expanduser(exe.strip())
    if os.path.isfile(exe):
        return os.path.abspath(exe)
    found = shutil.which(exe)
    if found:
        return found
    raise SystemExit(f"--python not found: {exe!r}")


def _run(py: str, argv: list[str]) -> None:
    cmd = [py] + argv
    print("+", " ".join(cmd))
    subprocess.check_call(cmd, cwd=_REPO)


def _parse_ids(s: str | None, dataset: str) -> list[int]:
    if s:
        return [int(x.strip()) for x in s.split(",") if x.strip()]
    ids = _np_s0_crop_ids_for_dataset(dataset)
    if not ids:
        raise SystemExit(
            f"No rows for dataset={dataset!r} in {_NP_S0_CSV}; pass --crop_ids explicitly."
        )
    return ids


def _apply_joint_mixed_out(args) -> None:
    """Default joint --out to mixed checkpoint name when training two domains."""
    if args.cmd != "joint" or not getattr(args, "datasets", None):
        return
    parts = [s.strip() for s in args.datasets.split(",") if s.strip()]
    if len(parts) < 2:
        return

    def n(p: str) -> str:
        return os.path.normpath(p)

    if n(args.out) in (n(_DEF_JOINT_OUT_HELA2), n(_DEF_JOINT_OUT_HELA3)):
        args.out = _DEF_JOINT_OUT_MIXED


def _apply_dataset_path_defaults(args) -> None:
    """When training jrc_hela-3, use hela3 checkpoint/result paths if user left hela2 defaults."""
    if getattr(args, "datasets", None):
        parts = [s.strip() for s in args.datasets.split(",") if s.strip()]
        if len(parts) >= 2:
            return
    if getattr(args, "dataset", None) != "jrc_hela-3":
        return

    def n(p: str) -> str:
        return os.path.normpath(p)

    if args.cmd == "joint" and n(args.out) == n(_DEF_JOINT_OUT_HELA2):
        args.out = _DEF_JOINT_OUT_HELA3
    if args.cmd == "per-crop":
        if n(args.checkpoint_dir) == n(_DEF_PC_CKPT_DIR_HELA2):
            args.checkpoint_dir = _DEF_PC_CKPT_DIR_HELA3
        if n(args.result_dir) == n(_DEF_PC_RESULT_HELA2):
            args.result_dir = _DEF_PC_RESULT_HELA3
    if args.cmd == "infer-all":
        if n(args.checkpoint) == n(_DEF_JOINT_OUT_HELA2):
            args.checkpoint = _DEF_JOINT_OUT_HELA3
        if n(args.output_dir) == n(_DEF_INF_OUT_HELA2):
            args.output_dir = _DEF_INF_OUT_HELA3


def _default_crop_export_dir(data_root: str, dataset: str) -> str | None:
    if dataset.endswith("hela-2"):
        sub = "crop_exports_hela2_mito_bg"
    elif dataset.endswith("hela-3"):
        sub = "crop_exports_hela3_mito_bg"
    else:
        return None
    p = os.path.join(_REPO, data_root, sub)
    return p if os.path.isdir(p) else None


def _resolve_crop_export_for_args(args) -> str | None:
    if getattr(args, "open_organelle_data", False):
        return None
    manual = getattr(args, "crop_export_dir", None)
    if manual:
        p = manual
        return os.path.abspath(os.path.join(_REPO, p)) if not os.path.isabs(p) else p
    return _default_crop_export_dir(args.data_root, args.dataset)


def _data_source_cli(crop_export: str | None, zarr_root: str | None) -> list[str]:
    out: list[str] = []
    if crop_export:
        out += ["--crop_export_dir", crop_export]
    elif zarr_root:
        out += ["--zarr_root", zarr_root]
    return out


def _mc_generate_cli(args) -> list[str]:
    """generate.py marching-cubes level options (per-crop / infer-all)."""
    out = [
        "--mc_level_mode",
        args.mc_level_mode,
        "--mc_fixed_level",
        str(args.mc_fixed_level),
        "--mc_percentile",
        str(args.mc_percentile),
        "--mc_adaptive_floor",
        str(args.mc_adaptive_floor),
        "--mesh_strip_boundary_voxels",
        str(args.mesh_strip_boundary_voxels),
        "--mesh_strip_mc_shell_cells",
        str(args.mesh_strip_mc_shell_cells),
        "--mc_prob_smooth_sigma",
        str(args.mc_prob_smooth_sigma),
        "--preview_vertex_stride",
        str(args.preview_vertex_stride),
        "--preview_point_size",
        str(args.preview_point_size),
        "--preview_alpha",
        str(args.preview_alpha),
    ]
    if getattr(args, "save_mc_prob_tiff", False):
        out += ["--save_mc_prob_tiff"]
    if getattr(args, "save_mc_prob_tiff_binary", False):
        out += ["--save_mc_prob_tiff_binary"]
    if getattr(args, "raw_subcrop", None):
        out += ["--raw_subcrop", getattr(args, "raw_subcrop", "")]
    if getattr(args, "native_required", False):
        out += [
            "--native_required",
            "--native_tile",
            str(getattr(args, "native_tile", 160)),
            "--native_overlap",
            str(getattr(args, "native_overlap", 24)),
            "--query_chunk",
            str(getattr(args, "query_chunk", 250000)),
        ]
    if getattr(args, "binary_cleanup", False):
        out += ["--binary_cleanup"]
        out += [
            "--binary_cleanup_threshold",
            str(getattr(args, "binary_cleanup_threshold", 0.5)),
            "--binary_cleanup_min_voxels",
            str(getattr(args, "binary_cleanup_min_voxels", 300)),
            "--binary_cleanup_close_iters",
            str(getattr(args, "binary_cleanup_close_iters", 1)),
            "--binary_cleanup_open_iters",
            str(getattr(args, "binary_cleanup_open_iters", 0)),
            "--binary_cleanup_keep_largest_k",
            str(getattr(args, "binary_cleanup_keep_largest_k", 0)),
            "--binary_cleanup_smooth_sigma",
            str(getattr(args, "binary_cleanup_smooth_sigma", 0.0)),
        ]
        if getattr(args, "binary_cleanup_roi", None):
            out += ["--binary_cleanup_roi", str(args.binary_cleanup_roi)]
    return out


def _append_metrics_csv(
    csv_path: str,
    dataset: str,
    crop_id: int,
    json_path: str,
    *,
    checkpoint: str | None = None,
    train_mode: str = "",
) -> None:
    import json

    with open(json_path, encoding="utf-8") as f:
        m = json.load(f)
    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "train_mode": train_mode,
        "checkpoint": os.path.basename(checkpoint) if checkpoint else "",
        "dataset": dataset,
        "crop_id": crop_id,
        "voxel_iou": m.get("voxel_iou"),
        "voxel_dice": m.get("voxel_dice"),
        "chamfer_mean_pred_to_gt_vox": m.get("chamfer_mean_pred_to_gt_vox"),
    }
    new_file = not os.path.isfile(csv_path)
    fieldnames = list(row.keys())
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if new_file:
            w.writeheader()
        w.writerow(row)
    print(f"Appended row to {csv_path}")


def _train_log_cli(args) -> list[str]:
    out: list[str] = []
    if getattr(args, "train_log_csv", None):
        out += ["--train_log_csv", args.train_log_csv, "--train_log_every", str(args.train_log_every)]
    return out


def _val_cli(args, val_log_csv: str | None = None) -> list[str]:
    ve = int(getattr(args, "val_every", 0) or 0)
    if ve <= 0:
        return []
    vlog = val_log_csv if val_log_csv is not None else getattr(args, "val_log_csv", None)
    out = [
        "--val_every",
        str(ve),
        "--val_num_points",
        str(getattr(args, "val_num_points", 4096)),
        "--val_seed",
        str(getattr(args, "val_seed", 42)),
    ]
    if vlog:
        out += ["--val_log_csv", vlog]
    return out


def _aug_cli(args) -> list[str]:
    out: list[str] = []
    if getattr(args, "aug_contrast", False):
        out.append("--aug_contrast")
        out += ["--aug_contrast_scale", str(getattr(args, "aug_contrast_scale", 0.1))]
    ns = float(getattr(args, "aug_noise_std", 0.0) or 0.0)
    if ns > 0:
        out += ["--aug_noise_std", str(ns)]
    if getattr(args, "aug_geometric", False):
        out.append("--aug_geometric")
    ag = float(getattr(args, "aug_gamma", 0.0) or 0.0)
    if ag > 0:
        out += ["--aug_gamma", str(ag)]
    if getattr(args, "aug_percentile_stretch", False):
        out.append("--aug_percentile_stretch")
    if (
        getattr(args, "aug_contrast", False)
        or ns > 0
        or getattr(args, "aug_geometric", False)
        or ag > 0
        or getattr(args, "aug_percentile_stretch", False)
    ):
        out += ["--aug_prob", str(getattr(args, "aug_prob", 0.5))]
    return out


def _weight_decay_cli(args) -> list[str]:
    wd = float(getattr(args, "weight_decay", 0.0) or 0.0)
    if wd > 0:
        return ["--weight_decay", str(wd)]
    return []


def _per_crop_train_log_path(base: str, cid: int) -> str:
    if base.lower().endswith(".csv"):
        return base[:-4] + f"_crop{cid}.csv"
    return f"{base}_crop{cid}.csv"


def _infer_validate_train_mode(checkpoint_abs: str) -> str:
    """Tag for summary CSV from checkpoint filename (mixed / hela2 / hela3 joint)."""
    base = os.path.basename(checkpoint_abs).lower()
    if "mixed" in base:
        return "mixed_joint_infer"
    if "hela3_all" in base:
        return "joint_infer_hela3"
    if "hela2_all" in base:
        return "joint_infer_hela2"
    return "joint_infer"


def main() -> None:
    ap = argparse.ArgumentParser(description="Hela2/Hela3 mito: joint, per-crop train, batch infer.")
    ap.add_argument(
        "--python",
        dest="python_exe",
        metavar="EXE",
        default=None,
        help="Python interpreter for train.py / generate.py / validate_crop.py (default: current).",
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_joint = sub.add_parser("joint", help="Train one model on all crop_ids (comma list or np_s0 default).")
    p_joint.add_argument("--data_root", type=str, default="data")
    p_joint.add_argument("--dataset", type=str, default="jrc_hela-2")
    p_joint.add_argument(
        "--datasets",
        type=str,
        default=None,
        help=(
            "Multi-domain joint: comma-separated, e.g. jrc_hela-2,jrc_hela-3 — one model, "
            "random (dataset,crop) per step from np_s0 manifest; needs both crop_export folders under data/. "
            "Default --out becomes checkpoints/model_hela2_hela3_mixed.pth."
        ),
    )
    p_joint.add_argument(
        "--crop_export_dirs",
        type=str,
        default=None,
        help=(
            "Mixed joint only: comma-separated export dirs, same length as --datasets entries. "
            "Use this to mix original + augmented export folders."
        ),
    )
    p_joint.add_argument(
        "--crop_ids",
        type=str,
        default=None,
        help="Default: all crops for --dataset from data/manifests/np_s0_with_data.csv",
    )
    p_joint.add_argument("--zarr_root", type=str, default=None)
    p_joint.add_argument(
        "--crop_export_dir",
        type=str,
        default=None,
        help="Sliced crop Zarr dir (see export_hela2_np_s0_mito_masked). Default: use data/crop_exports_*_mito_bg if it exists.",
    )
    p_joint.add_argument(
        "--open-organelle-data",
        action="store_true",
        help="Use full hela2.zarr recon-1 layout; do not use crop_exports.",
    )
    p_joint.add_argument("--epochs", type=int, default=300)
    p_joint.add_argument("--encoder_spatial", type=int, default=128)
    p_joint.add_argument("--num_points", type=int, default=24000)
    p_joint.add_argument(
        "--native_required",
        action="store_true",
        help="train.py strict native flag (no resize; native voxel crops).",
    )
    p_joint.add_argument(
        "--steps_per_epoch",
        type=int,
        default=4,
        help="train.py: gradient steps per logged epoch (amortizes CPU/Zarr; raises GPU load).",
    )
    p_joint.add_argument(
        "--fg_query_fraction",
        type=float,
        default=0.5,
        help="Fraction of query points on foreground (label>0); rest uniform in crop. Try ~0.38 if logits saturate.",
    )
    p_joint.add_argument("--lr", type=float, default=5e-4)
    p_joint.add_argument(
        "--weight_decay",
        type=float,
        default=1e-5,
        help="train.py Adam L2 (0 to disable).",
    )
    p_joint.add_argument("--out", type=str, default="checkpoints/model_hela2_all_mito.pth")
    p_joint.add_argument(
        "--train_log_csv",
        type=str,
        default=None,
        help="Passed to train.py: append loss curve CSV (convergence; not validation IoU).",
    )
    p_joint.add_argument(
        "--train_log_every",
        type=int,
        default=1,
        help="With --train_log_csv, log every N epochs (train.py).",
    )
    p_joint.add_argument(
        "--val_every",
        type=int,
        default=0,
        help="train.py: lightweight val loss every N epochs (fixed subcrop; CSV).",
    )
    p_joint.add_argument(
        "--val_log_csv",
        type=str,
        default=None,
        help="Default: <checkpoint_dir>/val_metrics.csv next to --out.",
    )
    p_joint.add_argument("--val_num_points", type=int, default=4096)
    p_joint.add_argument("--val_seed", type=int, default=42)
    p_joint.add_argument(
        "--aug_contrast",
        action="store_true",
        help="train.py: random intensity scaling (train only).",
    )
    p_joint.add_argument("--aug_contrast_scale", type=float, default=0.1)
    p_joint.add_argument(
        "--aug_noise_std",
        type=float,
        default=0.0,
        help="train.py: additive noise on raw 0–255 (try 2–8).",
    )
    p_joint.add_argument("--aug_prob", type=float, default=0.5)
    p_joint.add_argument(
        "--aug_geometric",
        action="store_true",
        help="train.py: random 90° rotations + flips (raw+label).",
    )
    p_joint.add_argument(
        "--aug_gamma",
        type=float,
        default=0.0,
        help="train.py: gamma span; 0 disables. Scripts often pass 0.12 with contrast/noise.",
    )
    p_joint.add_argument(
        "--aug_percentile_stretch",
        action="store_true",
        help="train.py: random percentile intensity stretch.",
    )

    p_pc = sub.add_parser("per-crop", help="Train a separate checkpoint for each crop_id.")
    p_pc.add_argument("--data_root", type=str, default="data")
    p_pc.add_argument("--dataset", type=str, default="jrc_hela-2")
    p_pc.add_argument("--crop_ids", type=str, default=None)
    p_pc.add_argument(
        "--only-crop",
        type=int,
        default=None,
        help="Process a single crop id (overrides --crop_ids).",
    )
    p_pc.add_argument("--zarr_root", type=str, default=None)
    p_pc.add_argument("--crop_export_dir", type=str, default=None)
    p_pc.add_argument(
        "--open-organelle-data",
        action="store_true",
        help="Use full hela2.zarr recon-1 layout; do not use crop_exports.",
    )
    p_pc.add_argument("--epochs", type=int, default=300)
    p_pc.add_argument("--encoder_spatial", type=int, default=128)
    p_pc.add_argument("--num_points", type=int, default=24000)
    p_pc.add_argument(
        "--native_required",
        action="store_true",
        help="train.py strict native flag (no resize; native voxel crops).",
    )
    p_pc.add_argument(
        "--steps_per_epoch",
        type=int,
        default=4,
        help="train.py: gradient steps per logged epoch (amortizes CPU/Zarr; raises GPU load).",
    )
    p_pc.add_argument(
        "--fg_query_fraction",
        type=float,
        default=0.5,
        help="Passed to train.py: foreground query fraction (see joint).",
    )
    p_pc.add_argument(
        "--checkpoint_dir",
        type=str,
        default="checkpoints/hela2_mito_per_crop",
        help="Directory for model_crop{id}.pth (not the project root).",
    )
    p_pc.add_argument(
        "--no-infer",
        action="store_true",
        help="Only train each crop; skip generate (PNG/OBJ). Default is to infer after each train.",
    )
    p_pc.add_argument("--infer_max_spatial", type=int, default=160)
    p_pc.add_argument("--mc_resolution", type=int, default=160)
    p_pc.add_argument(
        "--mc_level_mode",
        type=str,
        choices=("fixed", "percentile", "adaptive"),
        default="fixed",
        help="Marching-cubes iso-level (passed to generate.py).",
    )
    p_pc.add_argument("--mc_fixed_level", type=float, default=0.5)
    p_pc.add_argument("--mc_percentile", type=float, default=90.0)
    p_pc.add_argument("--mc_adaptive_floor", type=float, default=0.2)
    p_pc.add_argument(
        "--mesh_strip_mc_shell_cells",
        type=float,
        default=3.0,
        help="Strip outer N MC-grid cells (generate.py); main fix for boundary walls.",
    )
    p_pc.add_argument(
        "--mesh_strip_boundary_voxels",
        type=float,
        default=0.0,
        help="Optional extra voxel-space strip after MC scale; 0=off.",
    )
    p_pc.add_argument(
        "--mc_prob_smooth_sigma",
        type=float,
        default=0.0,
        help="Gaussian smooth occupancy grid before marching cubes (generate.py); 0=off.",
    )
    p_pc.add_argument("--preview_vertex_stride", type=int, default=2)
    p_pc.add_argument("--preview_point_size", type=float, default=5.0)
    p_pc.add_argument("--preview_alpha", type=float, default=0.8)
    p_pc.add_argument(
        "--save_mc_prob_tiff",
        action="store_true",
        help="generate.py: save MC P(occ) as float32 TIFF per crop (Fiji).",
    )
    p_pc.add_argument(
        "--save_mc_prob_tiff_binary",
        action="store_true",
        help="generate.py: also save uint8 binary mask TIFF (P>=threshold).",
    )
    p_pc.add_argument(
        "--raw_subcrop",
        type=str,
        default=None,
        help="generate.py: z0,y0,x0,dz,dy,dx subvolume before inference.",
    )
    p_pc.add_argument("--native_tile", type=int, default=160)
    p_pc.add_argument("--native_overlap", type=int, default=24)
    p_pc.add_argument("--query_chunk", type=int, default=250000)
    p_pc.add_argument(
        "--result_dir",
        type=str,
        default="result/hela2_mito_per_crop",
        help="Folder for preview PNGs and meshes (default under result/, not root).",
    )
    p_pc.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip mesh-vs-label metrics after generate (default: run validation).",
    )
    p_pc.add_argument(
        "--train_log_csv",
        type=str,
        default=None,
        help="Base path for train.py logs; becomes *_crop{id}.csv per crop.",
    )
    p_pc.add_argument("--train_log_every", type=int, default=1)
    p_pc.add_argument("--val_every", type=int, default=0)
    p_pc.add_argument(
        "--val_log_csv",
        type=str,
        default=None,
        help="Per-crop default: <checkpoint_dir>/val_metrics_crop{id}.csv if unset.",
    )
    p_pc.add_argument("--val_num_points", type=int, default=4096)
    p_pc.add_argument("--val_seed", type=int, default=42)
    p_pc.add_argument("--aug_contrast", action="store_true")
    p_pc.add_argument("--aug_contrast_scale", type=float, default=0.1)
    p_pc.add_argument("--aug_noise_std", type=float, default=0.0)
    p_pc.add_argument("--aug_prob", type=float, default=0.5)
    p_pc.add_argument("--aug_geometric", action="store_true")
    p_pc.add_argument("--aug_gamma", type=float, default=0.0)
    p_pc.add_argument("--aug_percentile_stretch", action="store_true")
    p_pc.add_argument("--lr", type=float, default=5e-4)
    p_pc.add_argument("--weight_decay", type=float, default=1e-5)

    p_inf = sub.add_parser(
        "infer-all",
        help=(
            "Joint model: batch generate.py on each crop (same weights), then optional montage PNG for the paper."
        ),
    )
    p_inf.add_argument("--data_root", type=str, default="data")
    p_inf.add_argument("--dataset", type=str, default="jrc_hela-2")
    p_inf.add_argument("--crop_ids", type=str, default=None)
    p_inf.add_argument("--zarr_root", type=str, default=None)
    p_inf.add_argument("--crop_export_dir", type=str, default=None)
    p_inf.add_argument(
        "--open-organelle-data",
        action="store_true",
        help="Infer from full hela2.zarr; do not use crop_exports.",
    )
    p_inf.add_argument(
        "--checkpoint",
        type=str,
        default=os.path.join("checkpoints", "model_hela2_all_mito.pth"),
        help="Weights from `joint` training (default path).",
    )
    p_inf.add_argument(
        "--output_dir",
        type=str,
        default="result/hela2_mito_joint_infer",
        help="Per-crop PNG/OBJ + montage (default under result/).",
    )
    p_inf.add_argument("--infer_max_spatial", type=int, default=160)
    p_inf.add_argument("--mc_resolution", type=int, default=160)
    p_inf.add_argument(
        "--mc_level_mode",
        type=str,
        choices=("fixed", "percentile", "adaptive"),
        default="percentile",
        help=(
            "Marching-cubes iso-level (passed to generate.py). "
            "Default percentile: robust when P(occ) is calibrated below 0.5 everywhere; "
            "use fixed + --mc_fixed_level 0.5 if your field spans 0.5 clearly."
        ),
    )
    p_inf.add_argument("--mc_fixed_level", type=float, default=0.5)
    p_inf.add_argument("--mc_percentile", type=float, default=90.0)
    p_inf.add_argument("--mc_adaptive_floor", type=float, default=0.2)
    p_inf.add_argument("--mesh_strip_mc_shell_cells", type=float, default=3.0)
    p_inf.add_argument("--mesh_strip_boundary_voxels", type=float, default=0.0)
    p_inf.add_argument(
        "--mc_prob_smooth_sigma",
        type=float,
        default=0.75,
        help="Gaussian smooth occupancy grid before marching cubes (generate.py); 0=off.",
    )
    p_inf.add_argument("--preview_vertex_stride", type=int, default=2)
    p_inf.add_argument("--preview_point_size", type=float, default=5.0)
    p_inf.add_argument("--preview_alpha", type=float, default=0.8)
    p_inf.add_argument(
        "--save_mc_prob_tiff",
        action="store_true",
        help="generate.py: save MC P(occ) grid as float32 TIFF per crop (open in Fiji).",
    )
    p_inf.add_argument(
        "--save_mc_prob_tiff_binary",
        action="store_true",
        help="generate.py: also save uint8 binary mask TIFF (P>=mc_binary_threshold in generate).",
    )
    p_inf.add_argument(
        "--raw_subcrop",
        type=str,
        default=None,
        help=(
            "generate.py: one subvolume z0,y0,x0,dz,dy,dx applied to every crop in this run. "
            "Use with --crop_ids for a single id if shapes differ per crop."
        ),
    )
    p_inf.add_argument("--native_required", action="store_true")
    p_inf.add_argument("--native_tile", type=int, default=160)
    p_inf.add_argument("--native_overlap", type=int, default=24)
    p_inf.add_argument("--query_chunk", type=int, default=250000)
    p_inf.add_argument(
        "--binary_cleanup",
        action="store_true",
        help="Native-only: binarize + cleanup (remove_small_objects) before marching cubes; outputs OBJ/PNG only.",
    )
    p_inf.add_argument("--binary_cleanup_threshold", type=float, default=0.5)
    p_inf.add_argument("--binary_cleanup_min_voxels", type=int, default=300)
    p_inf.add_argument("--binary_cleanup_close_iters", type=int, default=1)
    p_inf.add_argument("--binary_cleanup_open_iters", type=int, default=0)
    p_inf.add_argument("--binary_cleanup_keep_largest_k", type=int, default=0)
    p_inf.add_argument("--binary_cleanup_smooth_sigma", type=float, default=0.0)
    p_inf.add_argument("--binary_cleanup_roi", type=str, default=None)
    p_inf.add_argument(
        "--no-montage",
        action="store_true",
        help="Skip joint_infer_montage.png grid after all crops.",
    )
    p_inf.add_argument(
        "--validate",
        action="store_true",
        help="After each crop, mesh vs label metrics -> crop{id}_metrics_joint.json + metrics_joint_infer.csv",
    )
    p_inf.add_argument(
        "--validate_train_mode",
        type=str,
        default=None,
        help=(
            "Label written to metrics_*_summary.csv (default: inferred from checkpoint name, "
            "e.g. mixed_joint_infer for model_hela2_hela3_mixed.pth)."
        ),
    )

    args = ap.parse_args()
    _apply_joint_mixed_out(args)
    _apply_dataset_path_defaults(args)
    py = resolve_python(args.python_exe)
    print(f"Interpreter: {py}", flush=True)

    crop_export = _resolve_crop_export_for_args(args)
    if crop_export:
        print(f"Using crop export Zarrs: {crop_export}", flush=True)
    elif args.cmd in ("joint", "per-crop", "infer-all") and getattr(
        args, "zarr_root", None
    ):
        print(f"Using Open Organelle zarr_root: {args.zarr_root}", flush=True)

    if args.cmd == "joint":
        mix_parts = (
            [s.strip() for s in args.datasets.split(",") if s.strip()]
            if getattr(args, "datasets", None)
            else []
        )
        if len(mix_parts) >= 2:
            if getattr(args, "open_organelle_data", False):
                raise SystemExit(
                    "joint --datasets … requires sliced crop-export Zarrs under data/, not --open-organelle-data."
                )
            if getattr(args, "crop_export_dirs", None):
                dirs_csv = str(args.crop_export_dirs)
            else:
                dir_list: list[str] = []
                for ds in mix_parts:
                    dd = _default_crop_export_dir(args.data_root, ds)
                    if not dd:
                        raise SystemExit(
                            f"No crop export folder for {ds!r} under data_root={args.data_root!r} "
                            "(expected crop_exports_hela2_mito_bg / crop_exports_hela3_mito_bg)."
                        )
                    dir_list.append(dd)
                dirs_csv = ",".join(dir_list)
            cmd = [
                _TRAIN,
                "--data_root",
                args.data_root,
                "--datasets",
                args.datasets,
                "--crop_export_dirs",
                dirs_csv,
                "--epochs",
                str(args.epochs),
                "--encoder_spatial",
                str(args.encoder_spatial),
                "--num_points",
                str(args.num_points),
                "--steps_per_epoch",
                str(getattr(args, "steps_per_epoch", 1)),
                "--fg_query_fraction",
                str(args.fg_query_fraction),
                "--lr",
                str(args.lr),
            ]
            cmd += _weight_decay_cli(args)
            cmd += [
                "--out",
                args.out,
            ]
            cmd += _train_log_cli(args)
            cmd += _val_cli(args)
            cmd += _aug_cli(args)
            if getattr(args, "native_required", False):
                cmd += ["--native_required"]
            print(f"Mixed joint: export dirs {dirs_csv}", flush=True)
            _run(py, cmd)
            return

        ids = _parse_ids(args.crop_ids, args.dataset)
        cmd = [
            _TRAIN,
            "--data_root",
            args.data_root,
            "--dataset",
            args.dataset,
            "--crop_ids",
            ",".join(map(str, ids)),
            "--epochs",
            str(args.epochs),
            "--encoder_spatial",
            str(args.encoder_spatial),
            "--num_points",
            str(args.num_points),
            "--steps_per_epoch",
            str(getattr(args, "steps_per_epoch", 1)),
            "--fg_query_fraction",
            str(args.fg_query_fraction),
            "--lr",
            str(args.lr),
        ]
        cmd += _weight_decay_cli(args)
        cmd += [
            "--out",
            args.out,
        ]
        cmd += _train_log_cli(args)
        cmd += _val_cli(args)
        cmd += _aug_cli(args)
        if getattr(args, "native_required", False):
            cmd += ["--native_required"]
        cmd += _data_source_cli(crop_export, args.zarr_root)
        _run(py, cmd)
        return

    if args.cmd == "per-crop":
        if getattr(args, "only_crop", None) is not None:
            ids = [args.only_crop]
        else:
            ids = _parse_ids(args.crop_ids, args.dataset)
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        do_infer = not args.no_infer
        do_validate = do_infer and not args.no_validate
        if do_infer:
            os.makedirs(args.result_dir, exist_ok=True)
        metrics_csv = os.path.join(args.result_dir, "metrics_per_crop.csv")
        for cid in ids:
            ckpt = os.path.join(args.checkpoint_dir, f"model_crop{cid}.pth")
            cmd = [
                _TRAIN,
                "--data_root",
                args.data_root,
                "--dataset",
                args.dataset,
                "--crop_id",
                str(cid),
                "--epochs",
                str(args.epochs),
                "--encoder_spatial",
                str(args.encoder_spatial),
                "--num_points",
                str(args.num_points),
                "--steps_per_epoch",
                str(getattr(args, "steps_per_epoch", 1)),
                "--fg_query_fraction",
                str(args.fg_query_fraction),
                "--lr",
                str(args.lr),
            ]
            cmd += _weight_decay_cli(args)
            cmd += [
                "--out",
                ckpt,
            ]
            if getattr(args, "train_log_csv", None):
                tlog = _per_crop_train_log_path(args.train_log_csv, cid)
                cmd += ["--train_log_csv", tlog, "--train_log_every", str(args.train_log_every)]
            if getattr(args, "val_every", 0) and args.val_every > 0:
                vlog = args.val_log_csv
                if not vlog:
                    vlog = os.path.join(
                        os.path.abspath(os.path.join(_REPO, args.checkpoint_dir)),
                        f"val_metrics_crop{cid}.csv",
                    )
                elif not os.path.isabs(vlog):
                    vlog = os.path.abspath(os.path.join(_REPO, vlog))
                cmd += _val_cli(args, val_log_csv=vlog)
            cmd += _aug_cli(args)
            if getattr(args, "native_required", False):
                cmd += ["--native_required"]
            cmd += _data_source_cli(crop_export, args.zarr_root)
            _run(py, cmd)
            if do_infer:
                prefix = f"crop{cid}_"
                gcmd = [
                    _GEN,
                    "--data_root",
                    args.data_root,
                    "--dataset",
                    args.dataset,
                    "--crop_id",
                    str(cid),
                    "--checkpoint",
                    ckpt,
                    "--infer_max_spatial",
                    str(args.infer_max_spatial),
                    "--mc_resolution",
                    str(args.mc_resolution),
                    "--output_dir",
                    args.result_dir,
                    "--name_prefix",
                    prefix,
                ]
                gcmd += _data_source_cli(crop_export, args.zarr_root)
                gcmd += _mc_generate_cli(args)
                _run(py, gcmd)
                if do_validate:
                    obj_path = os.path.join(args.result_dir, f"{prefix}final_mitochondria.obj")
                    jpath = os.path.join(args.result_dir, f"crop{cid}_metrics.json")
                    vcmd = [
                        _VAL,
                        "--data_root",
                        args.data_root,
                        "--dataset",
                        args.dataset,
                        "--crop_id",
                        str(cid),
                        "--mesh",
                        obj_path,
                        "--out_json",
                        jpath,
                    ]
                    vcmd += _data_source_cli(crop_export, args.zarr_root)
                    _run(py, vcmd)
                    _append_metrics_csv(
                        metrics_csv,
                        args.dataset,
                        cid,
                        jpath,
                        checkpoint=ckpt,
                        train_mode="per_crop",
                    )
        if do_validate and os.path.isfile(metrics_csv):
            from src.metrics_summary import append_validation_summary

            append_validation_summary(
                metrics_csv,
                os.path.join(args.result_dir, "metrics_per_crop_summary.csv"),
                checkpoint=os.path.abspath(args.checkpoint_dir),
                dataset=args.dataset,
                train_mode="per_crop",
                note=(
                    "Separate checkpoint per crop. Summary = mean±std over crops in this run; "
                    "see metrics_per_crop.csv for each crop."
                ),
            )
        return

    if args.cmd == "infer-all":
        ckpt = os.path.abspath(os.path.join(_REPO, args.checkpoint))
        if not os.path.isfile(ckpt):
            raise SystemExit(
                f"Checkpoint missing: {ckpt}\n"
                "Train a joint model first, e.g.\n"
                "  python hela2_mito_pipeline.py --python <py> joint --epochs 300\n"
                "(jrc_hela-3 defaults to checkpoints/model_hela3_all_mito.pth)"
            )

        ids = _parse_ids(args.crop_ids, args.dataset)
        os.makedirs(args.output_dir, exist_ok=True)
        joint_csv = os.path.join(args.output_dir, "metrics_joint_infer.csv") if args.validate else None

        for cid in ids:
            prefix = f"crop{cid}_"
            gcmd = [
                _GEN,
                "--data_root",
                args.data_root,
                "--dataset",
                args.dataset,
                "--crop_id",
                str(cid),
                "--checkpoint",
                args.checkpoint,
                "--infer_max_spatial",
                str(args.infer_max_spatial),
                "--mc_resolution",
                str(args.mc_resolution),
                "--output_dir",
                args.output_dir,
                "--name_prefix",
                prefix,
            ]
            gcmd += _data_source_cli(crop_export, args.zarr_root)
            gcmd += _mc_generate_cli(args)
            _run(py, gcmd)

            if args.validate:
                obj_path = os.path.join(args.output_dir, f"{prefix}final_mitochondria.obj")
                jpath = os.path.join(args.output_dir, f"crop{cid}_metrics_joint.json")
                vcmd = [
                    _VAL,
                    "--data_root",
                    args.data_root,
                    "--dataset",
                    args.dataset,
                    "--crop_id",
                    str(cid),
                    "--mesh",
                    obj_path,
                    "--out_json",
                    jpath,
                ]
                vcmd += _data_source_cli(crop_export, args.zarr_root)
                _run(py, vcmd)
                if joint_csv:
                    _append_metrics_csv(
                        joint_csv,
                        args.dataset,
                        cid,
                        jpath,
                        checkpoint=args.checkpoint,
                        train_mode=args.validate_train_mode
                        or _infer_validate_train_mode(ckpt),
                    )

        if args.validate and joint_csv and os.path.isfile(joint_csv):
            from src.metrics_summary import append_validation_summary

            summary_path = os.path.join(
                args.output_dir, "metrics_joint_infer_summary.csv"
            )
            tag = args.validate_train_mode or _infer_validate_train_mode(ckpt)
            append_validation_summary(
                joint_csv,
                summary_path,
                checkpoint=ckpt,
                dataset=args.dataset,
                train_mode=tag,
                note=(
                    "Summary over all crops in this infer-all run. "
                    "Per-crop detail: metrics_joint_infer.csv. "
                    "One joint/mixed model — per-crop metrics diagnose FOV quality; "
                    "means are the headline numbers for the model on this dataset."
                ),
            )

        if not args.no_montage:
            from src.infer_montage import write_joint_montage

            out_abs = os.path.abspath(os.path.join(_REPO, args.output_dir))
            montage_path = write_joint_montage(out_abs, ids)
            if montage_path:
                print(f"Saved montage: {montage_path}", flush=True)

        return


if __name__ == "__main__":
    main()
