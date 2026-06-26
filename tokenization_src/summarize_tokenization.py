"""
summarize_tokenization.py  —  ver1

Prints summary statistics for a tokenized cycle dataset and saves
matplotlib figures to the same directory.

Reads from tokenization_outputs/ver1/ (or a directory passed as argv[1]).

Statistics reported:
    Cohort          patients, samples, cycles per patient
    Labels          positive / negative counts, rate
    Sequence lengths  mean, median, std, min, max, % truncated
    Vocabulary      total size, breakdown by event type
    Splits          train / val / test counts (if splits.json exists)
    Age             distribution of decade buckets

Figures saved (PNG):
    label_distribution.png
    sequence_length_distributions.png
    drug_cycles_distribution.png
    vocabulary_breakdown.png
    age_distribution.png
    split_summary.png  (only if splits_summary.csv exists)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

# ── configuration ─────────────────────────────────────────────────────────────
DEFAULT_DIR = Path(__file__).resolve().parent.parent / "tokenization_outputs" / "ver1"
# ─────────────────────────────────────────────────────────────────────────────


def _bar(value: float, width: int = 30, char: str = "█") -> str:
    filled = round(value * width)
    return char * filled + "░" * (width - filled)


def _save_label_distribution(labels: torch.Tensor, samples: pd.DataFrame, out_dir: Path) -> None:
    n_pos = int(labels.sum())
    n_neg = int((labels == 0).sum())
    pat_labels = samples.groupby("subject_id")["binary_label"].max()
    n_pos_pat = int((pat_labels == 1).sum())
    n_neg_pat = int((pat_labels == 0).sum())

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, (pos, neg, title) in zip(axes, [
        (n_pos, n_neg, "Sample-level label distribution"),
        (n_pos_pat, n_neg_pat, "Patient-level label distribution"),
    ]):
        bars = ax.bar(["Negative (0)", "Positive (1)"], [neg, pos], color=["#4c72b0", "#dd8452"])
        ax.set_title(title)
        ax.set_ylabel("Count")
        total = pos + neg
        for bar, count in zip(bars, [neg, pos]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + total * 0.01,
                    f"{count:,}\n({count/total:.1%})", ha="center", va="bottom", fontsize=9)
        ax.set_ylim(0, max(neg, pos) * 1.18)

    fig.tight_layout()
    fig.savefig(out_dir / "label_distribution.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_dir / 'label_distribution.png'}")


def _save_sequence_length_plots(
    samples: pd.DataFrame,
    seq_lens: np.ndarray,
    max_seq_len: int,
    out_dir: Path,
) -> None:
    """4-panel figure: raw vs post-truncation × sample level vs patient level."""
    if "raw_seq_len" in samples.columns:
        raw_lens = samples["raw_seq_len"].values
    else:
        print("  WARNING: raw_seq_len not found in samples — re-tokenize to get true pre-truncation lengths. Falling back to seq_len.")
        raw_lens = samples["seq_len"].values

    tmp = samples.copy()
    tmp["trunc_len"] = seq_lens
    raw_col = "raw_seq_len" if "raw_seq_len" in samples.columns else "seq_len"
    pat_raw   = tmp.groupby("subject_id")[raw_col].mean().values
    pat_trunc = tmp.groupby("subject_id")["trunc_len"].mean().values

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("Sequence Length Distributions", fontsize=14, fontweight="bold")

    # (lens, title, ylabel, show_cap_line)
    panels = [
        (axes[0, 0], raw_lens,   "Raw — Sample Level",
         "Number of samples",  False),
        (axes[0, 1], pat_raw,    "Raw — Patient Level  (mean across cycles)",
         "Number of patients", False),
        (axes[1, 0], seq_lens,   "Post-Tokenization / Truncated — Sample Level",
         "Number of samples",  True),
        (axes[1, 1], pat_trunc,  "Post-Tokenization / Truncated — Patient Level  (mean across cycles)",
         "Number of patients", True),
    ]

    for ax, lens, title, ylabel, show_cap in panels:
        ax.hist(lens, bins=50, color="#4c72b0", edgecolor="white", linewidth=0.4)
        if show_cap:
            ax.axvline(max_seq_len, color="#c44e52", linestyle="--", linewidth=1.4,
                       label=f"max_seq_len = {max_seq_len}")
        ax.axvline(float(np.median(lens)), color="#55a868", linestyle="--", linewidth=1.4,
                   label=f"median = {np.median(lens):.0f}")
        ax.set_xlabel("Sequence length (tokens)")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8)

    fig.tight_layout()
    path = out_dir / "sequence_length_distributions.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def _save_drug_cycles_distribution(samples: pd.DataFrame, out_dir: Path) -> None:
    """2-panel: cycle-number histogram (sample level) + total cycles per patient (patient level)."""
    cpp = samples.groupby("subject_id")["cycle_number"].max()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Drug Cycle Distributions", fontsize=14, fontweight="bold")

    # Sample level: which cycle index each sample sits at
    ax = axes[0]
    cnum = samples["cycle_number"].values
    bins = np.arange(cnum.min() - 0.5, cnum.max() + 1.5, 1)
    ax.hist(cnum, bins=bins, color="#4c72b0", edgecolor="white", linewidth=0.4)
    ax.axvline(float(np.mean(cnum)), color="#c44e52", linestyle="--", linewidth=1.4,
               label=f"mean = {np.mean(cnum):.1f}")
    ax.axvline(float(np.median(cnum)), color="#55a868", linestyle="--", linewidth=1.4,
               label=f"median = {np.median(cnum):.0f}")
    ax.set_xlabel("Cycle number")
    ax.set_ylabel("Number of samples")
    ax.set_title("Cycle Number — Sample Level")
    ax.legend(fontsize=8)

    # Patient level: total drug cycles per patient
    ax = axes[1]
    cpp_vals = cpp.values
    bins = np.arange(cpp_vals.min() - 0.5, cpp_vals.max() + 1.5, 1)
    ax.hist(cpp_vals, bins=bins, color="#dd8452", edgecolor="white", linewidth=0.4)
    ax.axvline(float(np.mean(cpp_vals)), color="#c44e52", linestyle="--", linewidth=1.4,
               label=f"mean = {np.mean(cpp_vals):.1f}")
    ax.axvline(float(np.median(cpp_vals)), color="#55a868", linestyle="--", linewidth=1.4,
               label=f"median = {np.median(cpp_vals):.0f}")
    ax.set_xlabel("Total drug cycles")
    ax.set_ylabel("Number of patients")
    ax.set_title("Drug Cycles per Patient — Patient Level")
    ax.legend(fontsize=8)

    fig.tight_layout()
    path = out_dir / "drug_cycles_distribution.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def _save_vocabulary_breakdown(vocab: dict, concept_ids: torch.Tensor, out_dir: Path) -> None:
    concept_vocab = vocab["concept_vocab"]
    by_type: dict[str, int] = {}
    for tok in concept_vocab:
        if "::" in tok:
            etype = tok.split("::")[0]
            by_type[etype] = by_type.get(etype, 0) + 1

    etypes = sorted(by_type, key=lambda k: -by_type[k])
    counts = [by_type[e] for e in etypes]

    fig, ax = plt.subplots(figsize=(8, max(3, len(etypes) * 0.7 + 1)))
    bars = ax.barh(etypes[::-1], counts[::-1], color="#4c72b0")
    ax.set_xlabel("Number of unique tokens")
    ax.set_title("Vocabulary breakdown by event type")
    total = sum(counts)
    for bar, count in zip(bars, counts[::-1]):
        ax.text(bar.get_width() + total * 0.005, bar.get_y() + bar.get_height() / 2,
                f"{count:,}  ({count/total:.1%})", va="center", fontsize=9)
    ax.set_xlim(0, max(counts) * 1.2)
    fig.tight_layout()
    fig.savefig(out_dir / "vocabulary_breakdown.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_dir / 'vocabulary_breakdown.png'}")


def _save_age_distribution(age_ids: torch.Tensor, samples: pd.DataFrame, out_dir: Path) -> None:
    decade_labels = [f"{i*10}–{i*10+9}" for i in range(10)]
    age_arr = age_ids.numpy()
    sample_counts = [int((age_arr == i).sum()) for i in range(10)]

    # Patient-level: one age per patient (modal bucket across their cycles)
    pat_ages = samples.groupby("subject_id")["age_id"].agg(
        lambda x: x.mode().iloc[0]
    ).values
    patient_counts = [int((pat_ages == i).sum()) for i in range(10)]

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    fig.suptitle("Age Distribution (decade buckets)", fontsize=13, fontweight="bold")

    for ax, counts, ylabel, title in [
        (axes[0], sample_counts,  "Number of samples",  "Sample Level"),
        (axes[1], patient_counts, "Number of patients", "Patient Level"),
    ]:
        bars = ax.bar(decade_labels, counts, color="#4c72b0", edgecolor="white", linewidth=0.4)
        ax.set_xlabel("Age at prediction time")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=30)
        total = sum(counts)
        for bar, count in zip(bars, counts):
            if count > 0:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + total * 0.005,
                        f"{count:,}", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_dir / "age_distribution.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_dir / 'age_distribution.png'}")


def _save_split_summary(split_df: pd.DataFrame, out_dir: Path) -> None:
    splits = split_df["split"].tolist()
    n_pos = split_df["n_positive"].tolist()
    n_neg = split_df["n_negative"].tolist()

    x = np.arange(len(splits))
    width = 0.5

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x, n_neg, width, label="Negative (0)", color="#4c72b0")
    ax.bar(x, n_pos, width, bottom=n_neg, label="Positive (1)", color="#dd8452")
    ax.set_xticks(x)
    ax.set_xticklabels([s.capitalize() for s in splits])
    ax.set_ylabel("Number of samples")
    ax.set_title("Train / Val / Test split composition")
    ax.legend()
    for i, (pos, neg) in enumerate(zip(n_pos, n_neg)):
        total = pos + neg
        ax.text(i, total + total * 0.01, f"{total:,}", ha="center", va="bottom", fontsize=9)
    ax.set_ylim(0, max(p + n for p, n in zip(n_pos, n_neg)) * 1.12)
    fig.tight_layout()
    fig.savefig(out_dir / "split_summary.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_dir / 'split_summary.png'}")


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

    # ── figures ───────────────────────────────────────────────────────────────
    figures_dir = input_dir / "summarization_figures"
    figures_dir.mkdir(exist_ok=True)
    print(f"\n── Saving figures → {figures_dir} ──────────────────────────")
    _save_label_distribution(labels, samples, figures_dir)
    _save_sequence_length_plots(samples, seq_lens, max_seq_len, figures_dir)
    _save_drug_cycles_distribution(samples, figures_dir)
    _save_vocabulary_breakdown(vocab, concept_ids, figures_dir)
    _save_age_distribution(age_ids, samples, figures_dir)
    if splits_path.exists():
        _save_split_summary(split_df, figures_dir)


if __name__ == "__main__":
    dir_arg = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_DIR
    main(dir_arg)
