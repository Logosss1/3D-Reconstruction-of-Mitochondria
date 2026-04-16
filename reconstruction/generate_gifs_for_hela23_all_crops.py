import argparse
import os
from pathlib import Path
from typing import Iterable, List

import matplotlib

matplotlib.use("Agg")
import numpy as np
import trimesh
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation


def iter_obj_files(root: Path, mode: str) -> List[Path]:
    if mode == "pred":
        patterns = ["*_pred.obj"]
    elif mode == "gt":
        patterns = ["*_gt.obj"]
    elif mode == "all":
        patterns = ["*_pred.obj", "*_gt.obj"]
    else:
        raise ValueError("mode must be pred|gt|all")

    files: List[Path] = []
    for pat in patterns:
        files.extend(list(root.glob(pat)))
    # deterministic
    files = sorted(set(files))
    return files


def create_rotation_gif(
    mesh_path: Path,
    out_gif: Path,
    frames: int = 120,
    interval_ms: int = 50,
    max_points: int = 20000,
    elev: float = 20.0,
    azim_step: float = 3.0,
) -> None:
    mesh = trimesh.load(str(mesh_path), force="mesh")
    if mesh is None or mesh.vertices is None or len(mesh.vertices) == 0:
        print(f"[WARN] empty mesh: {mesh_path.name}, skip")
        return

    v = mesh.vertices
    step = max(1, len(v) // max_points)
    v = v[::step]

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(v[:, 0], v[:, 1], v[:, 2], s=2, c=v[:, 2], cmap="magma", alpha=0.7)
    ax.set_axis_off()

    def update(frame: int):
        ax.view_init(elev=elev, azim=frame * azim_step)
        return (ax,)

    ani = FuncAnimation(fig, update, frames=frames, interval=interval_ms)
    out_gif.parent.mkdir(parents=True, exist_ok=True)
    ani.save(str(out_gif), writer="pillow")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate rotation GIFs for hela23 crop recon results.")
    parser.add_argument("--root", type=Path, default=Path("result/hela23_all_crops"))
    parser.add_argument("--mode", type=str, default="all", choices=["pred", "gt", "all"])
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--interval", type=int, default=50)
    parser.add_argument("--max-points", type=int, default=20000)
    args = parser.parse_args()

    root: Path = args.root
    if not root.exists():
        raise FileNotFoundError(str(root))

    obj_files = iter_obj_files(root, args.mode)
    if not obj_files:
        print(f"[WARN] no obj files found under {root} for mode={args.mode}")
        return

    print(f"Found {len(obj_files)} obj files under {root} (mode={args.mode})")
    for i, p in enumerate(obj_files, 1):
        out_gif = p.with_suffix(".gif")
        if out_gif.exists():
            print(f"[SKIP] {i}/{len(obj_files)} {out_gif.name} exists")
            continue
        print(f"[RUN]  {i}/{len(obj_files)} {p.name} -> {out_gif.name}")
        create_rotation_gif(
            p,
            out_gif,
            frames=args.frames,
            interval_ms=args.interval,
            max_points=args.max_points,
        )

    print("✅ GIF generation done.")


if __name__ == "__main__":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    main()

