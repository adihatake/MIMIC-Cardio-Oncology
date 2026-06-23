"""
dataset.py

PyTorch Dataset and DataLoader factory for the tokenized EHR cycle dataset.

Loads the six tensor files from a tokenization_outputs/<name>/ directory and
uses splits.json (written by split_dataset.py) to partition samples into
train / val / test without any patient overlap.

Usage (standalone check):
    python model_src/dataset.py tokenization_outputs/ver1

Programmatic:
    from model_src.dataset import get_dataloaders
    train_dl, val_dl, test_dl = get_dataloaders("tokenization_outputs/ver1", batch_size=32)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parent.parent


class CycleDataset(Dataset):
    """One item = one (patient, cycle) sample identified by a pre-computed row index."""

    def __init__(self, indices: list[int], data_dir: Path) -> None:
        self.indices = indices
        self.concept_ids  = torch.load(data_dir / "concept_ids.pt",  weights_only=True)
        self.type_ids     = torch.load(data_dir / "type_ids.pt",     weights_only=True)
        self.visit_ids    = torch.load(data_dir / "visit_ids.pt",    weights_only=True)
        self.position_ids = torch.load(data_dir / "position_ids.pt", weights_only=True)
        self.age_ids      = torch.load(data_dir / "age_ids.pt",      weights_only=True)
        self.labels       = torch.load(data_dir / "labels.pt",       weights_only=True)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        idx = self.indices[i]
        return {
            "concept_ids":  self.concept_ids[idx],
            "type_ids":     self.type_ids[idx],
            "visit_ids":    self.visit_ids[idx],
            "position_ids": self.position_ids[idx],
            "age_ids":      self.age_ids[idx],
            "label":        self.labels[idx],
        }


def get_dataloaders(
    data_dir: str | Path,
    batch_size: int = 32,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Returns (train_loader, val_loader, test_loader).
    Raises FileNotFoundError if splits.json is missing — run split_dataset.py first.
    """
    data_dir = Path(data_dir)
    splits_path = data_dir / "splits.json"
    if not splits_path.exists():
        raise FileNotFoundError(
            f"splits.json not found in {data_dir}. "
            "Run: python tokenization_src/split_dataset.py {data_dir}"
        )

    with open(splits_path) as f:
        splits = json.load(f)

    row_indices = splits["row_indices"]

    loaders: dict[str, DataLoader] = {}
    for split in ("train", "val", "test"):
        ds = CycleDataset(row_indices[split], data_dir)
        loaders[split] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    return loaders["train"], loaders["val"], loaders["test"]


if __name__ == "__main__":
    data_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO_ROOT / "tokenization_outputs" / "ver1"
    train_dl, val_dl, test_dl = get_dataloaders(data_dir, batch_size=4)
    batch = next(iter(train_dl))
    print(f"Train batches : {len(train_dl)}")
    print(f"Val   batches : {len(val_dl)}")
    print(f"Test  batches : {len(test_dl)}")
    print(f"Batch keys    : {list(batch.keys())}")
    print(f"concept_ids   : {batch['concept_ids'].shape}")
    print(f"age_ids       : {batch['age_ids'].shape}")
    print(f"label         : {batch['label']}")
