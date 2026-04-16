import glob
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
from scipy import ndimage as ndi

import rtx50_compat  # noqa: F401
from src.model import ConvONet
from src.post_process import reconstruction_pipeline


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


class ExportCropBankMixed:
    def __init__(self, patch_size: int = 128, seed: int = 42):
        self.patch_size = patch_size
        self.rng = random.Random(seed)
        self.items: List[Tuple[str, zarr.Array, zarr.Array, zarr.Array]] = []

        paths = sorted(
            glob.glob("data/crop_exports_hela2_mito_bg/*.zarr")
            + glob.glob("data/crop_exports_hela3_mito_bg/*.zarr")
        )
        for p in paths:
            z = zarr.open(p, mode="r")
            if "raw" not in z or "label" not in z:
                continue
            raw_arr = z["raw"]
            raw_masked_arr = z["raw_masked"] if "raw_masked" in z else z["raw"]
            lab_arr = z["label"]
            if raw_arr.shape != lab_arr.shape or raw_masked_arr.shape != lab_arr.shape:
                continue
            if min(raw_arr.shape) < patch_size:
                continue
            self.items.append((os.path.basename(p), raw_arr, raw_masked_arr, lab_arr))

        if not self.items:
            raise RuntimeError("No usable crop exports found.")

    def __len__(self) -> int:
        return len(self.items)

    def get_name(self, idx: int) -> str:
        return self.items[idx][0]

    def sample_patch(self, idx: int, masked_prob: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
        _, raw_arr, raw_masked_arr, lab_arr = self.items[idx]
        d, h, w = raw_arr.shape
        ps = self.patch_size
        raw_patch = None
        lab_patch = None
        for _ in range(12):
            z0 = self.rng.randint(0, d - ps)
            y0 = self.rng.randint(0, h - ps)
            x0 = self.rng.randint(0, w - ps)
            use_masked = self.rng.random() < masked_prob
            src = raw_masked_arr if use_masked else raw_arr
            raw_patch = np.asarray(src[z0:z0 + ps, y0:y0 + ps, x0:x0 + ps], dtype=np.float32)
            lab_patch = np.asarray(lab_arr[z0:z0 + ps, y0:y0 + ps, x0:x0 + ps], dtype=np.uint8)
            if np.any(lab_patch > 0):
                break
        return raw_patch, lab_patch

    def get_full_crop(self, idx: int) -> Tuple[str, np.ndarray]:
        name, raw_arr, _, _ = self.items[idx]
        return name, np.asarray(raw_arr[:], dtype=np.float32)


def sample_points_hard(
    label_patch: np.ndarray,
    n_points: int,
    pos_ratio: float,
    hard_neg_ratio: float,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    d, h, w = label_patch.shape
    pos_mask = label_patch > 0
    neg_mask = ~pos_mask

    n_pos = int(n_points * pos_ratio)
    n_neg = n_points - n_pos
    n_hard_neg = int(n_neg * hard_neg_ratio)
    n_rand_neg = n_neg - n_hard_neg

    pos = np.argwhere(pos_mask)
    if len(pos) == 0:
        n_pos = 0
        n_neg = n_points
        n_hard_neg = 0
        n_rand_neg = n_points

    # hard negative = near positive boundary
    if n_hard_neg > 0:
        dil = ndi.binary_dilation(pos_mask, iterations=2)
        boundary_neg = np.argwhere(dil & neg_mask)
        if len(boundary_neg) == 0:
            n_hard_neg = 0
            n_rand_neg = n_neg
    else:
        boundary_neg = np.empty((0, 3), dtype=np.int64)

    pts_vox = []
    if n_pos > 0:
        p = pos[np.random.randint(0, len(pos), size=n_pos)]
        pts_vox.append(p)
    if n_hard_neg > 0:
        hn = boundary_neg[np.random.randint(0, len(boundary_neg), size=n_hard_neg)]
        pts_vox.append(hn)
    if n_rand_neg > 0:
        rz = np.random.randint(0, d, size=n_rand_neg)
        ry = np.random.randint(0, h, size=n_rand_neg)
        rx = np.random.randint(0, w, size=n_rand_neg)
        rn = np.stack([rz, ry, rx], axis=1)
        pts_vox.append(rn)

    vox = np.concatenate(pts_vox, axis=0).astype(np.int64)
    np.random.shuffle(vox)
    tgt = (label_patch[vox[:, 0], vox[:, 1], vox[:, 2]] > 0).astype(np.float32)

    pts = np.stack(
        [
            vox[:, 0] / max(d - 1, 1),
            vox[:, 1] / max(h - 1, 1),
            vox[:, 2] / max(w - 1, 1),
        ],
        axis=1,
    ).astype(np.float32)
    return torch.from_numpy(pts).unsqueeze(0).to(device), torch.from_numpy(tgt).unsqueeze(0).to(device)


def split_indices(n: int, val_ratio: float = 0.25) -> Tuple[List[int], List[int]]:
    idx = list(range(n))
    random.shuffle(idx)
    nv = max(2, int(round(n * val_ratio)))
    nv = min(n - 1, nv)
    return idx[:-nv], idx[-nv:]


def save_recon(model: ConvONet, bank: ExportCropBankMixed, idx: int, device: torch.device, out_dir: Path) -> None:
    model.eval()
    name, full_raw = bank.get_full_crop(idx)
    # Avoid OOM on very large native crops (e.g., 800^3 / 1000x500x1000):
    # reconstruct on a center subvolume with capped size.
    max_dim = 256
    d, h, w = full_raw.shape
    if max(d, h, w) > max_dim:
        z0 = max(0, d // 2 - max_dim // 2)
        y0 = max(0, h // 2 - max_dim // 2)
        x0 = max(0, w // 2 - max_dim // 2)
        z1 = min(d, z0 + max_dim)
        y1 = min(h, y0 + max_dim)
        x1 = min(w, x0 + max_dim)
        recon_raw = full_raw[z0:z1, y0:y1, x0:x1]
        recon_resolution = 128
    else:
        recon_raw = full_raw
        recon_resolution = 160

    x = torch.from_numpy(recon_raw).unsqueeze(0).unsqueeze(0).to(device) / 255.0
    _, mesh = reconstruction_pipeline(
        model, x, recon_raw, resolution=recon_resolution, threshold=0.2
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = name.replace(".zarr", "")
    mesh.export(str(out_dir / f"{stem}_recon.obj"))

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")
    v = mesh.vertices[::2] if len(mesh.vertices) > 2 else mesh.vertices
    ax.scatter(v[:, 0], v[:, 1], v[:, 2], s=4, c=v[:, 2], cmap="magma", alpha=0.8)
    ax.set_title(f"Hela23 exports recon ({name})")
    plt.tight_layout()
    plt.savefig(out_dir / f"{stem}_preview.png", dpi=200)
    plt.savefig(out_dir / f"{stem}_paper.png", dpi=300)
    plt.close(fig)


def train() -> None:
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 v2 training on {device}")
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    bank = ExportCropBankMixed(patch_size=128, seed=42)
    tr_idx, va_idx = split_indices(len(bank), 0.25)
    print("usable crops:", len(bank), "| train:", len(tr_idx), "| val:", len(va_idx))
    print("train:", [bank.get_name(i) for i in tr_idx])
    print("val:", [bank.get_name(i) for i in va_idx])

    model = ConvONet().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=140, eta_min=1e-5)
    bce = nn.BCEWithLogitsLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    max_epochs = 140
    train_steps = 80
    val_steps = max(12, len(va_idx) * 5)
    train_points = 28000
    val_points = 42000
    pos_ratio = 0.45
    hard_neg_ratio = 0.6
    patience = 30
    min_delta = 1e-4

    out_dir = Path("result/hela23_exports_v2")
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path("checkpoints")
    ckpt_dir.mkdir(exist_ok=True)
    best_ckpt = ckpt_dir / "model_hela23_exports_v2_best.pth"
    last_ckpt = ckpt_dir / "model_hela23_exports_v2_last.pth"

    hist: Dict[str, List[float]] = {k: [] for k in ["train_loss", "train_acc", "train_iou", "val_loss", "val_acc", "val_iou", "val_f1"]}

    best_iou = -1.0
    best_epoch = 0
    best_val_idx = va_idx[0]
    bad = 0

    for ep in range(1, max_epochs + 1):
        model.train()
        tl, ta, ti = [], [], []
        for _ in range(train_steps):
            bi = random.choice(tr_idx)
            raw_patch, lab_patch = bank.sample_patch(bi, masked_prob=0.5)
            x = torch.from_numpy(raw_patch).unsqueeze(0).unsqueeze(0).to(device) / 255.0
            pts, tgt = sample_points_hard(lab_patch, train_points, pos_ratio, hard_neg_ratio, device)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                logits = model(x, pts)
                loss = bce(logits, tgt) + dice_loss(logits, tgt)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            m = compute_metrics(logits.detach(), tgt)
            tl.append(float(loss.item()))
            ta.append(m["acc"])
            ti.append(m["iou"])

        model.eval()
        vl, va, vi, vf = [], [], [], []
        with torch.no_grad():
            for _ in range(val_steps):
                bi = random.choice(va_idx)
                raw_patch, lab_patch = bank.sample_patch(bi, masked_prob=0.5)
                x = torch.from_numpy(raw_patch).unsqueeze(0).unsqueeze(0).to(device) / 255.0
                pts, tgt = sample_points_hard(lab_patch, val_points, pos_ratio, hard_neg_ratio, device)
                with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                    logits = model(x, pts)
                    loss = bce(logits, tgt) + dice_loss(logits, tgt)
                m = compute_metrics(logits, tgt)
                vl.append(float(loss.item()))
                va.append(m["acc"])
                vi.append(m["iou"])
                vf.append(m["f1"])

        tr_loss, tr_acc, tr_iou = float(np.mean(tl)), float(np.mean(ta)), float(np.mean(ti))
        va_loss, va_acc, va_iou, va_f1 = float(np.mean(vl)), float(np.mean(va)), float(np.mean(vi)), float(np.mean(vf))

        hist["train_loss"].append(tr_loss)
        hist["train_acc"].append(tr_acc)
        hist["train_iou"].append(tr_iou)
        hist["val_loss"].append(va_loss)
        hist["val_acc"].append(va_acc)
        hist["val_iou"].append(va_iou)
        hist["val_f1"].append(va_f1)

        print(
            f"Epoch {ep:03d}/{max_epochs} | train_loss={tr_loss:.4f} acc={tr_acc:.3f} iou={tr_iou:.3f} | "
            f"val_loss={va_loss:.4f} acc={va_acc:.3f} iou={va_iou:.3f} f1={va_f1:.3f} | lr={optimizer.param_groups[0]['lr']:.2e}"
        )
        scheduler.step()

        if va_iou > best_iou + min_delta:
            best_iou = va_iou
            best_epoch = ep
            bad = 0
            best_val_idx = random.choice(va_idx)
            torch.save(model.state_dict(), best_ckpt)
        else:
            bad += 1

        if bad >= patience:
            print(f"🛑 Early stopping at epoch {ep} (best={best_epoch}, best val IoU={best_iou:.4f})")
            break

    torch.save(model.state_dict(), last_ckpt)
    summary = {
        "best_epoch": best_epoch,
        "best_val_iou": best_iou,
        "final_epoch": len(hist["train_loss"]),
        "final_val_acc": hist["val_acc"][-1],
        "final_val_iou": hist["val_iou"][-1],
        "final_val_f1": hist["val_f1"][-1],
        "best_val_crop_for_recon": bank.get_name(best_val_idx),
    }
    (out_dir / "metrics_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "history.json").write_text(json.dumps(hist, ensure_ascii=False), encoding="utf-8")

    e = np.arange(1, len(hist["train_loss"]) + 1)
    fig, axs = plt.subplots(1, 3, figsize=(16, 4))
    axs[0].plot(e, hist["train_loss"], label="train")
    axs[0].plot(e, hist["val_loss"], label="val")
    axs[0].set_title("Loss")
    axs[0].legend()
    axs[1].plot(e, hist["train_acc"], label="train")
    axs[1].plot(e, hist["val_acc"], label="val")
    axs[1].set_title("Accuracy")
    axs[1].legend()
    axs[2].plot(e, hist["train_iou"], label="train")
    axs[2].plot(e, hist["val_iou"], label="val")
    axs[2].set_title("IoU")
    axs[2].legend()
    plt.tight_layout()
    plt.savefig(out_dir / "training_curves.png", dpi=240)
    plt.close(fig)

    model.load_state_dict(torch.load(best_ckpt, map_location=device))
    save_recon(model, bank, best_val_idx, device, out_dir)

    print("✅ v2 done.")
    print("Best:", best_ckpt)
    print("Summary:", out_dir / "metrics_summary.json")
    print("Curves:", out_dir / "training_curves.png")


if __name__ == "__main__":
    train()

