"""
Clean OBJ mesh by removing tiny connected components.

This is a visualization/cleanup helper: it does NOT re-run the neural net.
Instead, it splits the mesh into connected components and keeps only components
with enough faces (or top-K by face count).
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import trimesh


def _maybe_render_preview_png(mesh: trimesh.Trimesh, out_png: str, *, title: str) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print("matplotlib not installed; skipping preview png.")
        return

    v = mesh.vertices
    if len(v) == 0:
        return

    # Plot a subsampled vertex set for speed.
    stride = max(1, int(len(v) / 20000))
    vs = v[::stride]
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(vs[:, 0], vs[:, 1], vs[:, 2], s=5.0, c=vs[:, 2], cmap="magma", alpha=0.8)
    ax.set_xlabel("X (vox)")
    ax.set_ylabel("Y (vox)")
    ax.set_zlabel("Z (vox)")
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close(fig)
    print(f"Saved preview: {out_png}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Remove small connected components from an OBJ mesh.")
    ap.add_argument("--in_obj", type=str, required=True)
    ap.add_argument("--out_obj", type=str, required=True)
    ap.add_argument("--min_faces", type=int, default=500, help="Drop components with faces < min_faces.")
    ap.add_argument(
        "--keep_largest_k",
        type=int,
        default=0,
        help="Keep only largest K components by face count (0 keeps all above min_faces).",
    )
    ap.add_argument("--preview_png", type=str, default=None, help="Optional preview PNG path.")
    args = ap.parse_args()

    if not os.path.isfile(args.in_obj):
        raise SystemExit(f"--in_obj not found: {args.in_obj}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out_obj)) or ".", exist_ok=True)

    mesh = trimesh.load(args.in_obj, force="mesh")
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))

    if len(mesh.faces) == 0:
        print("Mesh has 0 faces; nothing to do.")
        return

    parts = mesh.split(only_watertight=False)
    parts = [p for p in parts if len(p.faces) > 0]
    if not parts:
        print("No connected components found; copying mesh.")
        mesh.export(args.out_obj)
        return

    parts_sorted = sorted(parts, key=lambda m: len(m.faces), reverse=True)
    face_counts = [len(p.faces) for p in parts_sorted]
    print(f"Found {len(parts_sorted)} components; face_counts(top10)={face_counts[:10]}")

    selected: list[trimesh.Trimesh] = []
    if args.keep_largest_k and args.keep_largest_k > 0:
        selected = parts_sorted[: int(args.keep_largest_k)]
    else:
        selected = [p for p in parts_sorted if len(p.faces) >= int(args.min_faces)]

    if not selected:
        # If threshold is too high, keep the largest one to avoid empty output.
        selected = [parts_sorted[0]]
        print(
            f"No component passed min_faces={args.min_faces}; keeping the largest component instead."
        )

    cleaned = trimesh.util.concatenate(selected)
    cleaned.export(args.out_obj)
    print(f"Saved cleaned OBJ: {args.out_obj} (faces={len(cleaned.faces)})")

    if args.preview_png:
        _maybe_render_preview_png(cleaned, args.preview_png, title="Cleaned mesh preview")


if __name__ == "__main__":
    main()

