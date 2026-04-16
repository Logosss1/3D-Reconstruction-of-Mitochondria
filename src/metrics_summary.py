"""Roll up per-crop validation CSV into one summary row (mean ± std) for joint/mixed reporting."""

from __future__ import annotations

import csv
import os
from datetime import datetime, timezone
from statistics import mean, stdev


def _col_float(rows: list[dict], key: str) -> list[float]:
    out: list[float] = []
    for r in rows:
        v = r.get(key)
        if v is None or v == "":
            continue
        try:
            out.append(float(v))
        except (TypeError, ValueError):
            continue
    return out


def append_validation_summary(
    per_crop_csv: str,
    summary_csv: str,
    *,
    checkpoint: str,
    dataset: str,
    train_mode: str,
    note: str = "",
) -> None:
    """
    Read all numeric rows from per_crop_csv and append one summary row to summary_csv.
    train_mode: e.g. 'joint_infer', 'per_crop', 'mixed_infer' — for thesis / advisor tables.
    """
    if not os.path.isfile(per_crop_csv):
        print(f"[metrics summary] skip: missing {per_crop_csv}")
        return
    with open(per_crop_csv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        print(f"[metrics summary] skip: empty {per_crop_csv}")
        return

    niou = _col_float(rows, "voxel_iou")
    ndi = _col_float(rows, "voxel_dice")
    nch = _col_float(rows, "chamfer_mean_pred_to_gt_vox")

    def mu_sd(vals: list[float]) -> tuple[str, str]:
        if not vals:
            return "", ""
        if len(vals) == 1:
            return f"{vals[0]:.6f}", "0.000000"
        return f"{mean(vals):.6f}", f"{stdev(vals):.6f}"

    m_iou, s_iou = mu_sd(niou)
    m_di, s_di = mu_sd(ndi)
    m_ch, s_ch = mu_sd(nch)

    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "train_mode": train_mode,
        "dataset": dataset,
        "checkpoint": os.path.basename(checkpoint),
        "checkpoint_path": os.path.abspath(checkpoint),
        "n_crops": len(rows),
        "voxel_iou_mean": m_iou,
        "voxel_iou_std": s_iou,
        "voxel_dice_mean": m_di,
        "voxel_dice_std": s_di,
        "chamfer_mean_mean": m_ch,
        "chamfer_mean_std": s_ch,
        "per_crop_csv": os.path.abspath(per_crop_csv),
        "note": note,
    }

    new_file = not os.path.isfile(summary_csv)
    fieldnames = list(row.keys())
    os.makedirs(os.path.dirname(os.path.abspath(summary_csv)) or ".", exist_ok=True)
    with open(summary_csv, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if new_file:
            w.writeheader()
        w.writerow(row)
    print(
        f"[metrics summary] {train_mode} | {dataset} | n={len(rows)} "
        f"IoU_mean={m_iou} -> {summary_csv}",
        flush=True,
    )
