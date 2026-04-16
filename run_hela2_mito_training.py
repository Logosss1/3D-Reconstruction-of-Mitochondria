"""
Train ConvONet using jrc_hela-2 under data/hela2.zarr (crop mito + fibsem EM).

No intermediate sample.zarr; see src/hela2_zarr_crop.load_crop_em_mito_aligned.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

_REPO_ROOT = os.path.abspath(os.path.dirname(__file__))


def main() -> None:
    p = argparse.ArgumentParser(description="Hela2 mito: train from hela2.zarr crop + EM.")
    p.add_argument("--data_root", type=str, default=os.path.join(_REPO_ROOT, "data"))
    p.add_argument("--dataset", type=str, default="jrc_hela-2")
    p.add_argument("--crop_id", type=int, default=94)
    p.add_argument("--encoder_spatial", type=int, default=128)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--num_points", type=int, default=10000)
    p.add_argument("--checkpoint", type=str, default="checkpoints/model_final.pth")
    args = p.parse_args()

    py = sys.executable
    train_py = os.path.join(_REPO_ROOT, "train.py")
    cmd = [
        py,
        train_py,
        "--data_root",
        args.data_root,
        "--dataset",
        args.dataset,
        "--crop_id",
        str(args.crop_id),
        "--encoder_spatial",
        str(args.encoder_spatial),
        "--epochs",
        str(args.epochs),
        "--num_points",
        str(args.num_points),
        "--out",
        args.checkpoint,
    ]
    print("=== train (hela2.zarr) ===")
    subprocess.check_call(cmd, cwd=_REPO_ROOT)
    print(f"Done. Checkpoint: {args.checkpoint}")


if __name__ == "__main__":
    main()
