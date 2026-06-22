"""
split_dataset.py  —  ver1

Stratified patient-level train / val / test split for the tokenized
cycle dataset.

Splitting is done at the patient level (not sample level) so that all
cycles belonging to one patient land in the same partition — preventing
data leakage across splits.

Stratification is based on patient-level label: a patient is treated as
positive if any of their cycle samples has binary_label = 1.

Default proportions: 70 % train / 15 % val / 15 % test.

Outputs (tokenization_outputs/ver1/):
    splits.json           subject_id lists and row indices for each split
    splits_summary.csv    per-split label counts and positive rates
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

# ── configuration ─────────────────────────────────────────────────────────────
INPUT_DIR  = Path(__file__).resolve().parent.parent / "tokenization_outputs" / "ver1"
TRAIN_FRAC = 0.70
VAL_FRAC   = 0.15
TEST_FRAC  = 0.15
SEED       = 42
# ─────────────────────────────────────────────────────────────────────────────

assert abs(TRAIN_FRAC + VAL_FRAC + TEST_FRAC - 1.0) < 1e-9, "Fractions must sum to 1"


def _stratified_split(
    subjects: np.ndarray,
    labels:   np.ndarray,
    frac:     float,
    rng:      np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Draw `frac` of subjects from each label stratum separately, return
    (selected_indices, remaining_indices) into `subjects`.
    """
    selected: list[int] = []
    remaining: list[int] = []

    for label_val in np.unique(labels):
        stratum_idx = np.where(labels == label_val)[0]
        rng.shuffle(stratum_idx)
        k = max(1, round(frac * len(stratum_idx)))
        selected.extend(stratum_idx[:k].tolist())
        remaining.extend(stratum_idx[k:].tolist())

    return np.array(selected), np.array(remaining)


def main(input_name: str | None = None) -> None:
    global INPUT_DIR
    if input_name is not None:
        INPUT_DIR = Path(__file__).resolve().parent.parent / "tokenization_outputs" / input_name

    samples_path = INPUT_DIR / "samples.parquet"
    if not samples_path.exists():
        raise FileNotFoundError(
            f"samples.parquet not found at {INPUT_DIR}. "
            "Run tokenize_cycle_sequences.py first."
        )

    samples = pd.read_parquet(samples_path)
    print(f"Loaded {len(samples):,} samples from {len(samples['subject_id'].unique()):,} patients")

    # Patient-level label: positive if any cycle is positive
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
    rng      = np.random.default_rng(SEED)

    # Test split from full pool
    test_size = TEST_FRAC / 1.0
    test_idx, trainval_idx = _stratified_split(subjects, labels, test_size, rng)

    # Val split from the train+val pool
    val_size = VAL_FRAC / (TRAIN_FRAC + VAL_FRAC)
    trainval_labels = labels[trainval_idx]
    val_rel_idx, train_rel_idx = _stratified_split(
        np.arange(len(trainval_idx)), trainval_labels, val_size, rng
    )
    val_idx   = trainval_idx[val_rel_idx]
    train_idx = trainval_idx[train_rel_idx]

    split_subjects: dict[str, list[int]] = {
        "train": subjects[train_idx].tolist(),
        "val":   subjects[val_idx].tolist(),
        "test":  subjects[test_idx].tolist(),
    }

    # Row indices in samples DataFrame (for direct tensor indexing)
    split_row_indices: dict[str, list[int]] = {}
    for split, sids in split_subjects.items():
        sid_set = set(sids)
        split_row_indices[split] = samples.index[
            samples["subject_id"].isin(sid_set)
        ].tolist()

    # Verify no subject appears in more than one split
    all_assigned = sum(len(v) for v in split_subjects.values())
    unique_assigned = len(set().union(*[set(v) for v in split_subjects.values()]))
    assert all_assigned == unique_assigned == len(subjects), \
        "Subject assignment is not a clean partition — check split logic"

    # Save
    with open(INPUT_DIR / "splits.json", "w") as f:
        json.dump({
            "seed":        SEED,
            "train_frac":  TRAIN_FRAC,
            "val_frac":    VAL_FRAC,
            "test_frac":   TEST_FRAC,
            "subject_ids": {k: [int(x) for x in v] for k, v in split_subjects.items()},
            "row_indices": split_row_indices,
        }, f, indent=2)
    print(f"Saved splits to {INPUT_DIR / 'splits.json'}")

    # Summary table
    rows = []
    for split in ("train", "val", "test"):
        sids   = set(split_subjects[split])
        subset = samples[samples["subject_id"].isin(sids)]
        rows.append({
            "split":        split,
            "n_patients":   len(sids),
            "n_samples":    len(subset),
            "n_positive":   int(subset["binary_label"].sum()),
            "n_negative":   int((subset["binary_label"] == 0).sum()),
            "pct_positive": f"{subset['binary_label'].mean():.1%}",
        })
    summary = pd.DataFrame(rows)
    summary.to_csv(INPUT_DIR / "splits_summary.csv", index=False)

    print()
    print(summary.to_string(index=False))
    print()


if __name__ == "__main__":
    main()
