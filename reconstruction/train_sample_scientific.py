import json
import os
from pathlib import Path
from typing import Dict, Tuple

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
    intersect = (pred * target).sum()
    return 1 - (2.0 * intersect + smooth) / (pred.sum() + target.sum() + smooth)


def sample_points(
    label_np: np.ndarray,
    num_points: int,
    pos_ratio: float,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
        points: [1, N, 3] normalized in [0,1]
        target: [1, N] float binary
    """
    d, h, w = label_np.shape
    n_pos = int(num_points * pos_ratio)
    n_neg = num_points - n_pos

    pos_indices = np.argwhere(label_np > 0)
    if len(pos_indices) == 0:
        n_pos = 0
        n_neg = num_points

    # Positive points
    if n_pos > 0:
        chosen = pos_indices[np.random.randint(0, len(pos_indices), size=n_pos)]
        pz = chosen[:, 0] / max(d - 1, 1)
        py = chosen[:, 1] / max(h - 1, 1)
        px = chosen[:, 2] / max(w - 1, 1)
        p_pos = np.stack([pz, py, px], axis=1).astype(np.float32)
    else:
        p_pos = np.empty((0, 3), dtype=np.float32)

    # Random points
    p_neg = np.random.rand(n_neg, 3).astype(np.float32)

    pts = np.concatenate([p_pos, p_neg], axis=0)
    np.random.shuffle(pts)

    iz = np.clip((pts[:, 0] * d).astype(np.int64), 0, d - 1)
    ih = np.clip((pts[:, 1] * h).astype(np.int64), 0, h - 1)
    iw = np.clip((pts[:, 2] * w).astype(np.int64), 0, w - 1)
    tgt = label_np[iz, ih, iw].astype(np.float32)

    points = torch.from_numpy(pts).unsqueeze(0).to(device)
    target = torch.from_numpy(tgt).unsqueeze(0).to(device)
    return points, target


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
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "acc": float(acc),
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Scientific training on device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Data
    z = zarr.open("data/sample_hela2/sample.zarr", mode="r")
    raw_np = np.array(z["raw"], dtype=np.float32)
    label_np = (np.array(z["label"]) > 0).astype(np.float32)

    raw = torch.from_numpy(raw_np).unsqueeze(0).unsqueeze(0).to(device) / 255.0

    model = ConvONet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=5e-4)
    bce = nn.BCEWithLogitsLoss()

    # "科研级"训练配置
    max_epochs = 220
    train_steps_per_epoch = 20
    train_points = 12000
    val_points = 40000
    pos_ratio = 0.5
    early_stop_patience = 30
    min_delta = 1e-4

    run_dir = Path("outputs/scientific_sample")
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path("checkpoints")
    ckpt_dir.mkdir(exist_ok=True)
    best_ckpt = ckpt_dir / "model_sample_scientific_best.pth"
    last_ckpt = ckpt_dir / "model_sample_scientific_last.pth"

    history = {
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
    patience_counter = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        epoch_losses = []
        epoch_accs = []
        epoch_ious = []

        for _ in range(train_steps_per_epoch):
            points, target = sample_points(label_np, train_points, pos_ratio, device)
            logits = model(raw, points)
            loss = bce(logits, target) + dice_loss(logits, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            metrics = compute_metrics(logits.detach(), target)
            epoch_losses.append(float(loss.item()))
            epoch_accs.append(metrics["acc"])
            epoch_ious.append(metrics["iou"])

        train_loss = float(np.mean(epoch_losses))
        train_acc = float(np.mean(epoch_accs))
        train_iou = float(np.mean(epoch_ious))

        # Validation
        model.eval()
        with torch.no_grad():
            v_points, v_target = sample_points(label_np, val_points, pos_ratio, device)
            v_logits = model(raw, v_points)
            v_loss = bce(v_logits, v_target) + dice_loss(v_logits, v_target)
            v_metrics = compute_metrics(v_logits, v_target)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["train_iou"].append(train_iou)
        history["val_loss"].append(float(v_loss.item()))
        history["val_acc"].append(v_metrics["acc"])
        history["val_iou"].append(v_metrics["iou"])
        history["val_f1"].append(v_metrics["f1"])

        print(
            f"Epoch {epoch:03d}/{max_epochs} | "
            f"train_loss={train_loss:.4f} acc={train_acc:.3f} iou={train_iou:.3f} | "
            f"val_loss={v_loss.item():.4f} acc={v_metrics['acc']:.3f} iou={v_metrics['iou']:.3f} f1={v_metrics['f1']:.3f}"
        )

        improved = v_metrics["iou"] > (best_val_iou + min_delta)
        if improved:
            best_val_iou = v_metrics["iou"]
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), best_ckpt)
        else:
            patience_counter += 1

        if patience_counter >= early_stop_patience:
            print(
                f"🛑 Early stopping at epoch {epoch} (best epoch={best_epoch}, best val IoU={best_val_iou:.4f})"
            )
            break

    torch.save(model.state_dict(), last_ckpt)

    # Save metrics
    summary = {
        "best_epoch": best_epoch,
        "best_val_iou": best_val_iou,
        "final_epoch": len(history["train_loss"]),
        "final_val_acc": history["val_acc"][-1],
        "final_val_iou": history["val_iou"][-1],
        "final_val_f1": history["val_f1"][-1],
    }
    with (run_dir / "metrics_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    with (run_dir / "history.json").open("w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False)

    # Plot curves
    epochs = np.arange(1, len(history["train_loss"]) + 1)
    fig, axs = plt.subplots(1, 3, figsize=(16, 4))

    axs[0].plot(epochs, history["train_loss"], label="train")
    axs[0].plot(epochs, history["val_loss"], label="val")
    axs[0].set_title("Loss")
    axs[0].set_xlabel("Epoch")
    axs[0].legend()

    axs[1].plot(epochs, history["train_acc"], label="train")
    axs[1].plot(epochs, history["val_acc"], label="val")
    axs[1].set_title("Accuracy")
    axs[1].set_xlabel("Epoch")
    axs[1].legend()

    axs[2].plot(epochs, history["train_iou"], label="train")
    axs[2].plot(epochs, history["val_iou"], label="val")
    axs[2].set_title("IoU")
    axs[2].set_xlabel("Epoch")
    axs[2].legend()

    plt.tight_layout()
    curve_path = run_dir / "training_curves.png"
    plt.savefig(curve_path, dpi=220)
    plt.close(fig)

    print("✅ Training complete.")
    print(f"Best checkpoint: {best_ckpt}")
    print(f"Last checkpoint: {last_ckpt}")
    print(f"Metrics summary: {run_dir / 'metrics_summary.json'}")
    print(f"Curves: {curve_path}")


if __name__ == "__main__":
    # avoids OMP duplicate library crash on some Windows conda setups
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    main()

