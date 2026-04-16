import csv
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import zarr

import rtx50_compat  # noqa: F401
from src.model import ConvONet


def dice_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred = torch.sigmoid(pred)
    smooth = 1e-5
    inter = (pred * target).sum()
    return 1 - (2.0 * inter + smooth) / (pred.sum() + target.sum() + smooth)


def compute_metrics(logits: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    probs = torch.sigmoid(logits)
    pred = (probs >= 0.5).float()
    tp = ((pred == 1) & (target == 1)).sum().item()
    tn = ((pred == 0) & (target == 0)).sum().item()
    fp = ((pred == 1) & (target == 0)).sum().item()
    fn = ((pred == 0) & (target == 1)).sum().item()
    total = tp + tn + fp + fn
    acc = (tp + tn) / total if total > 0 else 0.0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return {"acc": float(acc), "iou": float(iou), "f1": float(f1)}


def parse_np_with_data(np_csv: Path) -> Dict[str, List[int]]:
    out: Dict[str, List[int]] = {"jrc_hela-2": [], "jrc_hela-3": []}
    with np_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cell = row.get("cell", "")
            if cell not in out:
                continue
            c = row.get("crop", "")
            if c.startswith("crop"):
                try:
                    out[cell].append(int(c.replace("crop", "")))
                except ValueError:
                    pass
    for k in out:
        out[k] = sorted(set(out[k]))
    return out


class Hela23PatchBank:
    """
    Build patch samplers from:
      data/hela2.zarr/jrc_hela-2.zarr/recon-1/labels/groundtruth/cropX/{all,mito}/s0
      data/hela3.zarr/jrc_hela-3.zarr/recon-1/labels/groundtruth/cropX/{all,mito}/s0

    We use 'all/s0' as condition volume and 'mito/s0' as target occupancy.
    """

    def __init__(
        self,
        data_root: Path,
        crop_ids: Dict[str, List[int]],
        patch_size: int = 96,
        seed: int = 42,
    ) -> None:
        self.data_root = data_root
        self.patch_size = patch_size
        self.rng = random.Random(seed)
        self.items: List[Tuple[str, int, zarr.Array, zarr.Array]] = []

        zarr_paths = {
            "jrc_hela-2": data_root / "hela2.zarr" / "jrc_hela-2.zarr",
            "jrc_hela-3": data_root / "hela3.zarr" / "jrc_hela-3.zarr",
        }

        for cell, ids in crop_ids.items():
            zroot = zarr.open(str(zarr_paths[cell]), mode="r")
            for cid in ids:
                base = f"recon-1/labels/groundtruth/crop{cid}"
                all_key = f"{base}/all/s0"
                mito_key = f"{base}/mito/s0"
                try:
                    all_arr = zroot[all_key]
                    mito_arr = zroot[mito_key]
                except Exception:
                    continue
                if all_arr.shape != mito_arr.shape:
                    continue
                d, h, w = all_arr.shape
                if min(d, h, w) < patch_size:
                    # skip too small
                    continue
                self.items.append((cell, cid, all_arr, mito_arr))

        if not self.items:
            raise RuntimeError("No valid hela2/3 crops found for patch sampling.")

    def __len__(self) -> int:
        return len(self.items)

    def sample_patch(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        _, _, all_arr, mito_arr = self.items[index]
        d, h, w = all_arr.shape
        ps = self.patch_size

        # Try a few times to find patch with some mito positives.
        for _ in range(8):
            z0 = self.rng.randint(0, d - ps)
            y0 = self.rng.randint(0, h - ps)
            x0 = self.rng.randint(0, w - ps)
            all_patch = np.asarray(all_arr[z0:z0 + ps, y0:y0 + ps, x0:x0 + ps], dtype=np.float32)
            mito_patch = np.asarray(mito_arr[z0:z0 + ps, y0:y0 + ps, x0:x0 + ps], dtype=np.uint8)
            if np.any(mito_patch > 0):
                return all_patch, mito_patch

        # fallback: last sampled patch
        return all_patch, mito_patch


def sample_points_from_patch(
    label_patch: np.ndarray,
    num_points: int,
    pos_ratio: float,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    d, h, w = label_patch.shape
    n_pos = int(num_points * pos_ratio)
    n_neg = num_points - n_pos

    pos_idx = np.argwhere(label_patch > 0)
    if len(pos_idx) == 0:
        n_pos = 0
        n_neg = num_points

    if n_pos > 0:
        picked = pos_idx[np.random.randint(0, len(pos_idx), size=n_pos)]
        p_pos = np.stack(
            [
                picked[:, 0] / max(d - 1, 1),
                picked[:, 1] / max(h - 1, 1),
                picked[:, 2] / max(w - 1, 1),
            ],
            axis=1,
        ).astype(np.float32)
    else:
        p_pos = np.empty((0, 3), dtype=np.float32)

    p_rand = np.random.rand(n_neg, 3).astype(np.float32)
    pts = np.concatenate([p_pos, p_rand], axis=0)
    np.random.shuffle(pts)

    iz = np.clip((pts[:, 0] * d).astype(np.int64), 0, d - 1)
    ih = np.clip((pts[:, 1] * h).astype(np.int64), 0, h - 1)
    iw = np.clip((pts[:, 2] * w).astype(np.int64), 0, w - 1)
    tgt = (label_patch[iz, ih, iw] > 0).astype(np.float32)

    points = torch.from_numpy(pts).unsqueeze(0).to(device)
    target = torch.from_numpy(tgt).unsqueeze(0).to(device)
    return points, target


def split_indices(n: int, val_ratio: float = 0.25) -> Tuple[List[int], List[int]]:
    idx = list(range(n))
    if n <= 1:
        return idx, []
    n_val = max(1, int(round(n * val_ratio)))
    return idx[:-n_val], idx[-n_val:]


def train() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Hela2/3 mito scientific training on {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    data_root = Path("data")
    np_csv = Path(r"C:\Users\yf\Desktop\np_s0_with_data.csv")
    crop_ids = parse_np_with_data(np_csv)
    print("hela2 crops:", crop_ids["jrc_hela-2"])
    print("hela3 crops:", crop_ids["jrc_hela-3"])

    bank = Hela23PatchBank(data_root=data_root, crop_ids=crop_ids, patch_size=96, seed=42)
    tr_idx, va_idx = split_indices(len(bank), val_ratio=0.25)
    if not va_idx:
        va_idx = tr_idx
    print(f"usable crops: {len(bank)} | train={len(tr_idx)} val={len(va_idx)}")

    model = ConvONet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=5e-4)
    bce = nn.BCEWithLogitsLoss()

    max_epochs = 100
    train_steps = 40
    val_steps = max(8, len(va_idx) * 2)
    train_points = 12000
    val_points = 24000
    pos_ratio = 0.5
    patience = 20
    min_delta = 1e-4

    out_dir = Path("outputs/scientific_hela23")
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path("checkpoints")
    ckpt_dir.mkdir(exist_ok=True)
    best_ckpt = ckpt_dir / "model_hela23_mito_best.pth"
    last_ckpt = ckpt_dir / "model_hela23_mito_last.pth"

    history: Dict[str, List[float]] = {
        "train_loss": [],
        "train_acc": [],
        "train_iou": [],
        "val_loss": [],
        "val_acc": [],
        "val_iou": [],
        "val_f1": [],
    }

    best_val_iou = -1.0
    best_epoch = 0
    bad = 0

    for ep in range(1, max_epochs + 1):
        model.train()
        tr_losses, tr_accs, tr_ious = [], [], []

        for _ in range(train_steps):
            bi = random.choice(tr_idx)
            all_patch, mito_patch = bank.sample_patch(bi)

            # condition input
            x = torch.from_numpy(all_patch).unsqueeze(0).unsqueeze(0).to(device) / 255.0
            pts, tgt = sample_points_from_patch(mito_patch, train_points, pos_ratio, device)

            logits = model(x, pts)
            loss = bce(logits, tgt) + dice_loss(logits, tgt)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            m = compute_metrics(logits.detach(), tgt)
            tr_losses.append(float(loss.item()))
            tr_accs.append(m["acc"])
            tr_ious.append(m["iou"])

        tr_loss = float(np.mean(tr_losses))
        tr_acc = float(np.mean(tr_accs))
        tr_iou = float(np.mean(tr_ious))

        model.eval()
        va_losses, va_accs, va_ious, va_f1s = [], [], [], []
        with torch.no_grad():
            for _ in range(val_steps):
                bi = random.choice(va_idx)
                all_patch, mito_patch = bank.sample_patch(bi)
                x = torch.from_numpy(all_patch).unsqueeze(0).unsqueeze(0).to(device) / 255.0
                pts, tgt = sample_points_from_patch(mito_patch, val_points, pos_ratio, device)
                logits = model(x, pts)
                loss = bce(logits, tgt) + dice_loss(logits, tgt)
                m = compute_metrics(logits, tgt)
                va_losses.append(float(loss.item()))
                va_accs.append(m["acc"])
                va_ious.append(m["iou"])
                va_f1s.append(m["f1"])

        va_loss = float(np.mean(va_losses))
        va_acc = float(np.mean(va_accs))
        va_iou = float(np.mean(va_ious))
        va_f1 = float(np.mean(va_f1s))

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["train_iou"].append(tr_iou)
        history["val_loss"].append(va_loss)
        history["val_acc"].append(va_acc)
        history["val_iou"].append(va_iou)
        history["val_f1"].append(va_f1)

        print(
            f"Epoch {ep:03d}/{max_epochs} | "
            f"train_loss={tr_loss:.4f} acc={tr_acc:.3f} iou={tr_iou:.3f} | "
            f"val_loss={va_loss:.4f} acc={va_acc:.3f} iou={va_iou:.3f} f1={va_f1:.3f}"
        )

        if va_iou > best_val_iou + min_delta:
            best_val_iou = va_iou
            best_epoch = ep
            bad = 0
            torch.save(model.state_dict(), best_ckpt)
        else:
            bad += 1

        if bad >= patience:
            print(f"🛑 Early stopping at epoch {ep} (best={best_epoch}, best val IoU={best_val_iou:.4f})")
            break

    torch.save(model.state_dict(), last_ckpt)

    summary = {
        "best_epoch": best_epoch,
        "best_val_iou": best_val_iou,
        "final_epoch": len(history["train_loss"]),
        "final_val_acc": history["val_acc"][-1],
        "final_val_iou": history["val_iou"][-1],
        "final_val_f1": history["val_f1"][-1],
    }

    with (out_dir / "metrics_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    with (out_dir / "history.json").open("w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False)

    e = np.arange(1, len(history["train_loss"]) + 1)
    fig, axs = plt.subplots(1, 3, figsize=(16, 4))
    axs[0].plot(e, history["train_loss"], label="train")
    axs[0].plot(e, history["val_loss"], label="val")
    axs[0].set_title("Loss")
    axs[0].legend()
    axs[1].plot(e, history["train_acc"], label="train")
    axs[1].plot(e, history["val_acc"], label="val")
    axs[1].set_title("Accuracy")
    axs[1].legend()
    axs[2].plot(e, history["train_iou"], label="train")
    axs[2].plot(e, history["val_iou"], label="val")
    axs[2].set_title("IoU")
    axs[2].legend()
    plt.tight_layout()
    curve_path = out_dir / "training_curves.png"
    plt.savefig(curve_path, dpi=220)
    plt.close(fig)

    print("✅ Hela2/3 mito training complete.")
    print(f"Best checkpoint: {best_ckpt}")
    print(f"Metrics: {out_dir / 'metrics_summary.json'}")
    print(f"Curves: {curve_path}")


if __name__ == "__main__":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    train()

