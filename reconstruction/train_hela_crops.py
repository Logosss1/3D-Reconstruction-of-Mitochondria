import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import zarr
from matplotlib import pyplot as plt

import rtx50_compat  # noqa: F401  # ensure RTX 5090 compatibility hooks are active
from src.model import ConvONet


def dice_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred = torch.sigmoid(pred)
    smooth = 1e-5
    intersect = (pred * target).sum()
    return 1 - (2.0 * intersect + smooth) / (pred.sum() + target.sum() + smooth)


class HelaCropsDataset:
    """
    Simple in-memory dataset wrapper for cropped Zarr volumes:
    - root: data/hela_crops/jrc_hela-2 or jrc_hela-3
    - each crop: cropN.zarr with datasets 'raw' and 'label'
    """

    def __init__(self, root: Path, crop_names: List[str]) -> None:
        self.root = root
        self.crop_names = crop_names

        self.volumes: List[torch.Tensor] = []
        self.labels: List[torch.Tensor] = []
        self.pos_indices: List[torch.Tensor] = []
        self.shapes: List[Tuple[int, int, int]] = []

        for cname in crop_names:
            zpath = root / cname
            z = zarr.open(str(zpath), mode="r")
            raw_np = np.array(z["raw"], dtype=np.float32)
            lab_np = np.array(z["label"], dtype=np.uint8)

            # Normalize raw to [0,1]
            raw_t = torch.from_numpy(raw_np).unsqueeze(0).unsqueeze(0) / 255.0
            lab_t = torch.from_numpy(lab_np).unsqueeze(0).unsqueeze(0).float()

            self.volumes.append(raw_t)
            self.labels.append(lab_t)
            self.shapes.append(raw_np.shape)

            pos = np.argwhere(lab_np > 0)
            if pos.size == 0:
                pos_t = torch.zeros((0, 3), dtype=torch.float32)
            else:
                pos_t = torch.from_numpy(pos.astype(np.float32))
            self.pos_indices.append(pos_t)

    def __len__(self) -> int:
        return len(self.crop_names)

    def get_item(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Tuple[int, int, int]]:
        return (
            self.volumes[idx],  # [1,1,D,H,W]
            self.labels[idx],   # [1,1,D,H,W]
            self.pos_indices[idx],  # [K,3] in (z,y,x)
            self.shapes[idx],
        )


def make_train_val_splits(all_crops: List[str], val_ratio: float = 0.2) -> Tuple[List[str], List[str]]:
    if not all_crops:
        return [], []
    # deterministic split: last few crops as val
    n = len(all_crops)
    n_val = max(1, int(round(n * val_ratio))) if n > 1 else 0
    if n_val == 0:
        return all_crops, []
    train = all_crops[:-n_val]
    val = all_crops[-n_val:]
    return train, val


def sample_points_for_crop(
    pos_indices: torch.Tensor,
    shape: Tuple[int, int, int],
    num_points: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Sample mixture of positive and random points for one crop.
    Returns:
        points: [1,N,3] in [0,1]^3 with order (z,y,x)
        indices: (iz,ih,iw) integer indices on CPU for label lookup
    """
    D, H, W = shape
    half = num_points // 2

    if pos_indices.numel() > 0:
        n_pos = min(half, pos_indices.shape[0])
        sel = torch.randint(0, pos_indices.shape[0], (n_pos,), device=pos_indices.device)
        p_pos_vox = pos_indices[sel]  # [n_pos,3] (z,y,x)
        # Normalize by per-dimension size
        p_pos = torch.stack(
            [
                p_pos_vox[:, 0] / max(D - 1, 1),
                p_pos_vox[:, 1] / max(H - 1, 1),
                p_pos_vox[:, 2] / max(W - 1, 1),
            ],
            dim=1,
        )
    else:
        n_pos = 0
        p_pos = torch.empty((0, 3), device=device)

    n_rand = num_points - n_pos
    p_rand = torch.rand((n_rand, 3), device=device)

    pts = torch.cat([p_pos.to(device), p_rand], dim=0)  # [N,3]

    # Compute integer indices for target sampling
    iz = torch.clamp((pts[:, 0] * D).long(), 0, D - 1)
    ih = torch.clamp((pts[:, 1] * H).long(), 0, H - 1)
    iw = torch.clamp((pts[:, 2] * W).long(), 0, W - 1)

    pts = pts.unsqueeze(0)  # [1,N,3]
    return pts, (iz.cpu(), ih.cpu(), iw.cpu())


def compute_batch_metrics(
    logits: torch.Tensor,
    target: torch.Tensor,
) -> Dict[str, float]:
    """
    logits: [1,N]
    target: [1,N]
    """
    with torch.no_grad():
        probs = torch.sigmoid(logits)
        pred = (probs >= 0.5).float()
        tp = torch.sum((pred == 1.0) & (target == 1.0)).item()
        tn = torch.sum((pred == 0.0) & (target == 0.0)).item()
        fp = torch.sum((pred == 1.0) & (target == 0.0)).item()
        fn = torch.sum((pred == 0.0) & (target == 1.0)).item()

        total = tp + tn + fp + fn
        acc = (tp + tn) / total if total > 0 else 0.0

        inter = tp
        union = tp + fp + fn
        iou = inter / union if union > 0 else 0.0

    return {"acc": acc, "iou": iou}


def evaluate(
    model: ConvONet,
    dataset: HelaCropsDataset,
    crop_indices: List[int],
    device: torch.device,
    num_points: int = 20000,
) -> Dict[str, float]:
    model.eval()
    all_losses: List[float] = []
    all_acc: List[float] = []
    all_iou: List[float] = []
    bce = nn.BCEWithLogitsLoss()

    with torch.no_grad():
        for ci in crop_indices:
            raw, label, pos_idx, shape = dataset.get_item(ci)
            raw = raw.to(device)
            label_cpu = label  # keep on CPU for indexing

            pts, (iz, ih, iw) = sample_points_for_crop(
                pos_idx, shape, num_points=num_points, device=device
            )
            # Targets from CPU label
            d, h, w = shape
            tgt = label_cpu[0, 0, iz, ih, iw].unsqueeze(0).to(device)  # [1,N]

            logits = model(raw, pts)  # [1,N]
            l = bce(logits, tgt) + dice_loss(logits, tgt)
            all_losses.append(float(l.item()))

            metrics = compute_batch_metrics(logits, tgt)
            all_acc.append(metrics["acc"])
            all_iou.append(metrics["iou"])

    if not all_losses:
        return {"loss": 0.0, "acc": 0.0, "iou": 0.0}
    return {
        "loss": float(np.mean(all_losses)),
        "acc": float(np.mean(all_acc)),
        "iou": float(np.mean(all_iou)),
    }


def train_hela_crops() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Hela crops training on device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    project_root = Path(__file__).resolve().parent
    data_root = project_root / "data" / "hela_crops"

    # Discover crops
    hela2_root = data_root / "jrc_hela-2"
    hela3_root = data_root / "jrc_hela-3"

    hela2_crops = sorted([p.name for p in hela2_root.glob("crop*.zarr")]) if hela2_root.exists() else []
    hela3_crops = sorted([p.name for p in hela3_root.glob("crop*.zarr")]) if hela3_root.exists() else []

    if not hela2_crops and not hela3_crops:
        print("❌ No crops found under data/hela_crops. Run crop_hela_from_manifest.py first.")
        return

    print(f"Found hela2 crops: {hela2_crops}")
    print(f"Found hela3 crops: {hela3_crops}")

    # For now, train on hela2 crops only; hela3 can be used as extra val/test if desired.
    train_crops, val_crops = make_train_val_splits(hela2_crops, val_ratio=0.25)
    print(f"Train crops: {train_crops}")
    print(f"Val crops  : {val_crops}")

    train_dataset = HelaCropsDataset(hela2_root, train_crops)
    val_dataset = HelaCropsDataset(hela2_root, val_crops) if val_crops else train_dataset

    model = ConvONet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=5e-4)
    bce = nn.BCEWithLogitsLoss()

    max_epochs = 60
    steps_per_epoch = 100
    num_points_train = 10000
    num_points_val = 20000

    history: Dict[str, List[float]] = {
        "train_loss": [],
        "train_acc": [],
        "train_iou": [],
        "val_loss": [],
        "val_acc": [],
        "val_iou": [],
    }

    best_val_iou = -1.0
    best_epoch = -1
    checkpoints_dir = project_root / "checkpoints"
    checkpoints_dir.mkdir(exist_ok=True)
    best_ckpt_path = checkpoints_dir / "model_hela_crops_best.pth"

    for epoch in range(1, max_epochs + 1):
        model.train()
        epoch_losses: List[float] = []
        epoch_accs: List[float] = []
        epoch_ious: List[float] = []

        for _ in range(steps_per_epoch):
            # Randomly pick a crop from training set
            ci = np.random.randint(0, len(train_crops))
            raw, label, pos_idx, shape = train_dataset.get_item(ci)
            raw = raw.to(device)

            pts, (iz, ih, iw) = sample_points_for_crop(
                pos_idx, shape, num_points=num_points_train, device=device
            )

            d, h, w = shape
            tgt = label[0, 0, iz, ih, iw].unsqueeze(0).to(device)

            logits = model(raw, pts)
            loss = bce(logits, tgt) + dice_loss(logits, tgt)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            metrics = compute_batch_metrics(logits, tgt)
            epoch_losses.append(float(loss.item()))
            epoch_accs.append(metrics["acc"])
            epoch_ious.append(metrics["iou"])

        train_loss = float(np.mean(epoch_losses))
        train_acc = float(np.mean(epoch_accs))
        train_iou = float(np.mean(epoch_ious))

        # Validation
        val_metrics = evaluate(
            model,
            val_dataset,
            list(range(len(val_dataset))),
            device=device,
            num_points=num_points_val,
        )

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["train_iou"].append(train_iou)
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics["acc"])
        history["val_iou"].append(val_metrics["iou"])

        print(
            f"Epoch {epoch:03d}/{max_epochs} | "
            f"train_loss={train_loss:.4f} acc={train_acc:.3f} IoU={train_iou:.3f} | "
            f"val_loss={val_metrics['loss']:.4f} acc={val_metrics['acc']:.3f} IoU={val_metrics['iou']:.3f}"
        )

        # Track best model by validation IoU
        if val_metrics["iou"] > best_val_iou:
            best_val_iou = val_metrics["iou"]
            best_epoch = epoch
            torch.save(model.state_dict(), best_ckpt_path)

    print(f"✅ Training finished. Best val IoU={best_val_iou:.3f} at epoch {best_epoch}.")
    print(f"Best checkpoint: {best_ckpt_path}")

    # Plot learning curves
    epochs = np.arange(1, max_epochs + 1)
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))

    axs[0].plot(epochs, history["train_loss"], label="train")
    axs[0].plot(epochs, history["val_loss"], label="val")
    axs[0].set_title("Loss")
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Loss")
    axs[0].legend()

    axs[1].plot(epochs, history["train_acc"], label="train")
    axs[1].plot(epochs, history["val_acc"], label="val")
    axs[1].set_title("Accuracy")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Acc")
    axs[1].legend()

    axs[2].plot(epochs, history["train_iou"], label="train")
    axs[2].plot(epochs, history["val_iou"], label="val")
    axs[2].set_title("IoU")
    axs[2].set_xlabel("Epoch")
    axs[2].set_ylabel("IoU")
    axs[2].legend()

    plt.tight_layout()
    curves_path = project_root / "training_curves_hela_crops.png"
    plt.savefig(curves_path, dpi=200)
    plt.close(fig)
    print(f"📈 Training curves saved to: {curves_path}")


if __name__ == "__main__":
    train_hela_crops()

