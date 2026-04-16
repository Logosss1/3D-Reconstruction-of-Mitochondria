"""Crop ids per dataset from data/manifests/np_s0_with_data.csv."""

from __future__ import annotations

import csv
import os


def np_s0_crop_ids_for_dataset(manifest_csv: str, dataset: str) -> list[int]:
    if not os.path.isfile(manifest_csv):
        raise FileNotFoundError(manifest_csv)
    ids: list[int] = []
    with open(manifest_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["cell"].strip() != dataset.strip():
                continue
            c = row["crop"].strip()
            if not c.lower().startswith("crop"):
                raise ValueError(f"Bad crop field {c!r} in {manifest_csv}")
            ids.append(int(c[4:]))
    return ids
