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

import numpy as np
import pandas as pd
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
        # Optional files produced by the updated tokenizer — absent for old tokenizations.
        dates_path     = data_dir / "dates.pt"
        age_years_path = data_dir / "age_years.pt"
        self.dates     = torch.load(dates_path,     weights_only=True) if dates_path.exists()     else None
        self.age_years = torch.load(age_years_path, weights_only=True) if age_years_path.exists() else None

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        idx = self.indices[i]
        item = {
            "concept_ids":  self.concept_ids[idx],
            "type_ids":     self.type_ids[idx],
            "visit_ids":    self.visit_ids[idx],
            "position_ids": self.position_ids[idx],
            "age_ids":      self.age_ids[idx],
            "label":        self.labels[idx],
        }
        if self.dates is not None:
            item["dates"] = self.dates[idx]
        if self.age_years is not None:
            item["age_years"] = self.age_years[idx]
        return item


def _stratified_split(
    subjects: np.ndarray,
    labels: np.ndarray,
    frac: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    selected: list[int] = []
    remaining: list[int] = []
    for label_val in np.unique(labels):
        stratum_idx = np.where(labels == label_val)[0]
        rng.shuffle(stratum_idx)
        k = max(1, round(frac * len(stratum_idx)))
        selected.extend(stratum_idx[:k].tolist())
        remaining.extend(stratum_idx[k:].tolist())
    return np.array(selected), np.array(remaining)


def _compute_row_indices(
    data_dir: Path,
    seed: int,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
) -> dict[str, list[int]]:
    """Compute patient-level stratified split row indices from samples.parquet."""
    samples_path = data_dir / "samples.parquet"
    if not samples_path.exists():
        raise FileNotFoundError(
            f"samples.parquet not found at {data_dir}. "
            "Run tokenize_cycle_sequences.py first."
        )
    samples = pd.read_parquet(samples_path)

    patient_df = (
        samples.groupby("subject_id")["binary_label"]
        .max()
        .reset_index()
        .rename(columns={"binary_label": "patient_label"})
        .sort_values("subject_id")
        .reset_index(drop=True)
    )
    subjects = patient_df["subject_id"].values
    labels   = patient_df["patient_label"].values
    rng      = np.random.default_rng(seed)

    test_frac = 1.0 - train_frac - val_frac
    test_idx, trainval_idx = _stratified_split(subjects, labels, test_frac, rng)

    val_size       = val_frac / (train_frac + val_frac)
    trainval_labels = labels[trainval_idx]
    val_rel_idx, train_rel_idx = _stratified_split(
        np.arange(len(trainval_idx)), trainval_labels, val_size, rng
    )
    val_idx   = trainval_idx[val_rel_idx]
    train_idx = trainval_idx[train_rel_idx]

    split_subjects = {
        "train": subjects[train_idx].tolist(),
        "val":   subjects[val_idx].tolist(),
        "test":  subjects[test_idx].tolist(),
    }
    return {
        split: samples.index[samples["subject_id"].isin(set(sids))].tolist()
        for split, sids in split_subjects.items()
    }


def get_dataloaders(
    data_dir: str | Path,
    batch_size: int = 32,
    num_workers: int = 0,
    seed: int | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Returns (train_loader, val_loader, test_loader).

    If `seed` is given, the patient split is computed in-memory from
    samples.parquet using that seed (different seeds → different splits).
    Otherwise falls back to splits.json — run split_dataset.py first.
    """
    data_dir = Path(data_dir)

    if seed is not None:
        row_indices = _compute_row_indices(data_dir, seed)
    else:
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
