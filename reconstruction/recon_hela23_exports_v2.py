import json
from pathlib import Path

import numpy as np
import torch

from train_hela23_exports_converge_v2 import ExportCropBankMixed, save_recon
from src.model import ConvONet


def main() -> None:
    summary_path = Path("result/hela23_exports_v2/metrics_summary.json")
    ckpt_path = Path("checkpoints/model_hela23_exports_v2_best.pth")
    out_dir = Path("result/hela23_exports_v2")
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary: {summary_path}")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    target_name = summary.get("best_val_crop_for_recon")
    if not target_name:
        raise RuntimeError("best_val_crop_for_recon missing in summary")

    bank = ExportCropBankMixed(patch_size=128, seed=42)
    idx = None
    for i in range(len(bank)):
        if bank.get_name(i) == target_name:
            idx = i
            break
    if idx is None:
        raise RuntimeError(f"Target crop not found: {target_name}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ConvONet().to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    save_recon(model, bank, idx, device, out_dir)
    print("✅ Reconstructed:", target_name)
    print("📁 Saved under:", out_dir)


if __name__ == "__main__":
    main()

