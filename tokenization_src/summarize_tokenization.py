"""
summarize_tokenization.py  —  ver1

Prints summary statistics for a tokenized cycle dataset.

Reads from tokenization_outputs/ver1/ (or a directory passed as argv[1]).

Statistics reported:
    Cohort          patients, samples, cycles per patient
    Labels          positive / negative counts, rate
    Sequence lengths  mean, median, std, min, max, % truncated
    Vocabulary      total size, breakdown by event type
    Splits          train / val / test counts (if splits.json exists)
    Age             distribution of decade buckets
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# ── configuration ─────────────────────────────────────────────────────────────
DEFAULT_DIR = Path(__file__).resolve().parent.parent / "tokenization_outputs" / "ver1"
# ─────────────────────────────────────────────────────────────────────────────


def _bar(value: float, width: int = 30, char: str = "█") -> str:
    filled = round(value * width)
    return char * filled + "░" * (width - filled)


def main(input_dir: Path = DEFAULT_DIR) -> None:
    # ── load artefacts ────────────────────────────────────────────────────────
    if not input_dir.exists():
        print(f"Directory not found: {input_dir}")
        print("Run tokenize_cycle_sequences.py first.")
        sys.exit(1)

    samples = pd.read_parquet(input_dir / "samples.parquet")

    labels      = torch.load(input_dir / "labels.pt",      weights_only=True)
    mask        = torch.load(input_dir / "attention_mask.pt", weights_only=True)
    concept_ids = torch.load(input_dir / "concept_ids.pt", weights_only=True)
    age_ids     = torch.load(input_dir / "age_ids.pt",     weights_only=True)

    with open(input_dir / "vocab.json") as f:
        vocab = json.load(f)
    with open(input_dir / "metadata.json") as f:
        meta = json.load(f)

    seq_lens    = mask.sum(dim=1).numpy()
    max_seq_len = meta["max_seq_len"]

    W = 60
    print("=" * W)
    print(f"  TOKENIZED DATASET SUMMARY  —  {meta.get('version', '?')}")
    print("=" * W)

    # ── cohort ────────────────────────────────────────────────────────────────
    cycles_per_patient = samples.groupby("subject_id")["cycle_number"]
    print("\n── Cohort ──────────────────────────────────────────────────")
    print(f"  Unique patients          {samples['subject_id'].nunique():>10,}")
    print(f"  Total samples (cycles)   {len(samples):>10,}")
    print(f"  Max cycles / patient     {cycles_per_patient.max().max():>10}")
    print(f"  Mean cycles / patient    {cycles_per_patient.count().mean():>10.1f}")
    print(f"  Median cycles / patient  {cycles_per_patient.count().median():>10.0f}")

    # ── labels ────────────────────────────────────────────────────────────────
    n_pos  = int(labels.sum())
    n_neg  = int((labels == 0).sum())
    n_tot  = len(labels)
    p_rate = n_pos / n_tot if n_tot else 0.0

    print("\n── Labels ──────────────────────────────────────────────────")
    print(f"  Positive (label = 1)     {n_pos:>10,}   {p_rate:>6.1%}  {_bar(p_rate)}")
    print(f"  Negative (label = 0)     {n_neg:>10,}   {1-p_rate:>6.1%}  {_bar(1-p_rate)}")

    # Patient-level label balance
    pat_labels = samples.groupby("subject_id")["binary_label"].max()
    n_pos_pat  = int((pat_labels == 1).sum())
    n_neg_pat  = int((pat_labels == 0).sum())
    n_pat      = len(pat_labels)
    print(f"\n  Patient-level positive   {n_pos_pat:>10,} / {n_pat:,}  ({n_pos_pat/n_pat:.1%})")
    print(f"  Patient-level negative   {n_neg_pat:>10,} / {n_pat:,}  ({n_neg_pat/n_pat:.1%})")

    # ── sequence lengths ──────────────────────────────────────────────────────
    n_trunc = int((seq_lens == max_seq_len).sum())
    print("\n── Sequence lengths (tokens per sample) ────────────────────")
    print(f"  Mean                     {seq_lens.mean():>10.1f}")
    print(f"  Median                   {np.median(seq_lens):>10.0f}")
    print(f"  Std                      {seq_lens.std():>10.1f}")
    print(f"  Min                      {seq_lens.min():>10}")
    print(f"  Max (observed)           {seq_lens.max():>10}")
    print(f"  Max (configured)         {max_seq_len:>10}")
    print(f"  Truncated                {n_trunc:>10,}   ({n_trunc/n_tot:.1%} of samples)")

    # Histogram buckets
    buckets = [0, 50, 100, 200, 300, 400, 500, max_seq_len + 1]
    print()
    for lo, hi in zip(buckets[:-1], buckets[1:]):
        count = int(((seq_lens >= lo) & (seq_lens < hi)).sum())
        label_str = f"  [{lo:>3}, {hi-1 if hi <= max_seq_len else max_seq_len}{'+'if hi>max_seq_len else ''})"
        bar = _bar(count / n_tot, width=20)
        print(f"{label_str:<18}  {count:>6,}   {count/n_tot:>5.1%}  {bar}")

    # ── vocabulary ────────────────────────────────────────────────────────────
    concept_vocab = vocab["concept_vocab"]
    by_type: dict[str, int] = {}
    for tok in concept_vocab:
        if "::" in tok:
            etype = tok.split("::")[0]
            by_type[etype] = by_type.get(etype, 0) + 1

    # Unique tokens actually used in this dataset
    flat = concept_ids.flatten()
    special_max_id = 4   # [PAD]=0 [UNK]=1 [CLS]=2 [V_START]=3 [V_END]=4
    used_ids = flat[flat > special_max_id].unique()

    print("\n── Vocabulary ───────────────────────────────────────────────")
    print(f"  Total vocab size         {len(concept_vocab):>10,}")
    print(f"  Unique concepts used     {len(used_ids):>10,}")
    print()
    for etype, cnt in sorted(by_type.items(), key=lambda x: -x[1]):
        pct = cnt / len(concept_vocab)
        print(f"  {etype:<22}  {cnt:>8,}   {pct:>5.1%}  {_bar(pct, width=15)}")

    # ── age distribution ──────────────────────────────────────────────────────
    print("\n── Age at prediction time (decade buckets) ─────────────────")
    decade_labels = {i: f"{i*10}-{i*10+9}" for i in range(10)}
    age_arr = age_ids.numpy()
    for bucket in range(10):
        count = int((age_arr == bucket).sum())
        pct   = count / len(age_arr)
        print(f"  {decade_labels[bucket]:<8}  {count:>6,}   {pct:>5.1%}  {_bar(pct, width=20)}")

    # ── split summary ─────────────────────────────────────────────────────────
    splits_path = input_dir / "splits_summary.csv"
    if splits_path.exists():
        split_df = pd.read_csv(splits_path)
        print("\n── Train / Val / Test Splits ────────────────────────────────")
        print(split_df.to_string(index=False))

    # ── tensor info ───────────────────────────────────────────────────────────
    print("\n── Tensor shapes ────────────────────────────────────────────")
    print(f"  (n_samples, padded_len)  {list(concept_ids.shape)}")
    print(f"  Data dir:  {meta.get('data_dir', '?')}")
    print(f"  Saved to:  {input_dir}")
    print("=" * W)


if __name__ == "__main__":
    dir_arg = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_DIR
    main(dir_arg)
