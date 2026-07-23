"""
run_xgboost.py

XGBoost baselines for the cardiotoxicity prediction task.

Three feature variants (all use the same patient-level stratified split
as the transformer, so AUROC numbers are directly comparable):

  clinical  — 16 structured fields from final_cycle_binary_modeling_table.parquet
               (drug-class exposures, baseline LVEF, cycle metadata, etc.)
  bow       — Bag-of-words count vectors built from tokenized concept_ids.pt
               (same vocabulary as the transformer, but no order information)
  combined  — clinical + bow features concatenated

Usage:
    python run_xgboost.py                          # all three variants, 5 seeds
    python run_xgboost.py --variants clinical bow  # subset
    python run_xgboost.py --seeds 42 43            # fewer seeds
    python run_xgboost.py --experiment my_run      # custom output folder name

Results are written to:
    experiment_outputs/<experiment>/XGB-<variant>/seed<N>/
        test_metrics.json          — all metrics for the primary (AUROC) run
        test_metrics_{metric}.json — per-metric checkpoint results (auroc/auprc/f1/sensitivity/specificity)
        config.json                — hyperparameters + feature info

Run compare_ablations.py on the output folder to compare with the transformer:
    python evaluation/compare_ablations.py experiment_outputs/<experiment>/
"""

from __future__ import annotations

import os
# Prevent segfault from PyTorch + XGBoost sharing conflicting OpenMP runtimes on macOS
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    log_loss, roc_auc_score, average_precision_score,
    precision_recall_fscore_support, confusion_matrix,
)
from sklearn.preprocessing import StandardScaler
import xgboost as xgb


# ── constants ─────────────────────────────────────────────────────────────────

TOKENIZATION_DIR = Path("tokenization_outputs/Jul1_512")
COHORT_TABLE     = Path("cohort_outputs/cycle_modeling_v3/final_cycle_binary_modeling_table.parquet")

SEEDS    = [42, 43, 44, 45, 46]
VARIANTS = ["clinical", "bow", "combined"]

# Structured feature columns from the cohort table
CLINICAL_COLS = [
    # drug class exposure flags
    "exposed_anthracycline",
    "exposed_immune_checkpoint_inhibitor",
    "exposed_her2_targeted",
    "exposed_taxane",
    "exposed_fluoropyrimidine",
    "exposed_vegf_inhibitor",
    "exposed_egfr_inhibitor",
    "exposed_tyrosine_kinase_inhibitor",
    "exposed_proteasome_inhibitor",
    "exposed_immunomodulatory_agent",
    # clinical context
    "pre_existing_cv_history",
    "baseline_lvef",
    # cycle metadata
    "cycle_number",
    "toxicity_window_days",
    "n_prescription_rows_in_cycle",
    "n_exposure_start_days_in_cycle",
]

# XGBoost hyperparameters — tuned conservatively for a small dataset (~1800 train rows)
XGB_PARAMS = dict(
    objective        = "binary:logistic",
    eval_metric      = "auc",
    n_estimators     = 500,
    learning_rate    = 0.05,
    max_depth        = 4,
    min_child_weight = 5,
    subsample        = 0.8,
    colsample_bytree = 0.8,
    reg_alpha        = 0.1,
    reg_lambda       = 1.0,
    early_stopping_rounds = 30,
    verbosity        = 0,
    random_state     = 0,   # tree construction seed; split seed varies per run
)


# ── split logic (mirrors model_src/dataset.py exactly) ────────────────────────

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


def compute_row_indices(
    samples: pd.DataFrame,
    seed: int,
    train_frac: float = 0.70,
    val_frac: float   = 0.15,
) -> dict[str, list[int]]:
    """Patient-level stratified split — identical to model_src/dataset.py."""
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

    val_size        = val_frac / (train_frac + val_frac)
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


# ── feature builders ──────────────────────────────────────────────────────────

def build_clinical_features(cohort: pd.DataFrame, samples: pd.DataFrame) -> np.ndarray:
    """
    Merge cohort structured fields onto samples (by subject_id + cycle_number),
    impute missing baseline_lvef with the column median, and return a float array.
    """
    right_cols = list(dict.fromkeys(["subject_id", "cycle_number"] + CLINICAL_COLS))
    cohort_right = cohort[right_cols].copy()
    cohort_right["cycle_number"] = cohort_right["cycle_number"].astype("Int64")
    merged = samples[["subject_id", "cycle_number"]].merge(
        cohort_right,
        on=["subject_id", "cycle_number"],
        how="left",
    )
    lvef_median = merged["baseline_lvef"].median()
    merged["baseline_lvef"] = merged["baseline_lvef"].fillna(lvef_median)
    merged["cycle_number"]  = merged["cycle_number"].fillna(1.0)
    merged["n_prescription_rows_in_cycle"] = merged["n_prescription_rows_in_cycle"].fillna(1.0)
    return merged[CLINICAL_COLS].to_numpy(dtype=np.float32)


def build_bow_features(
    concept_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    vocab_size: int,
    special_token_ids: set[int],
) -> np.ndarray:
    """
    Build bag-of-words count vectors from concept_ids.
    Ignores padding positions and a set of special tokens ([CLS], [PAD], etc.).
    Returns shape (N, vocab_size) float32.
    """
    concept_np = concept_ids.numpy()         # (N, max_seq_len)
    mask_np    = attention_mask.numpy()      # (N, max_seq_len) bool

    N = concept_np.shape[0]
    bow = np.zeros((N, vocab_size), dtype=np.float32)

    for i in range(N):
        valid_tokens = concept_np[i][mask_np[i].astype(bool)]
        for tok in valid_tokens:
            if tok not in special_token_ids:
                bow[i, tok] += 1.0

    return bow


# ── single run ────────────────────────────────────────────────────────────────

def run_one(
    variant: str,
    seed: int,
    X: np.ndarray,
    y: np.ndarray,
    row_indices: dict[str, list[int]],
    output_dir: Path,
    xgb_params: dict,
) -> dict:
    t0 = time.time()

    tr_idx  = row_indices["train"]
    val_idx = row_indices["val"]
    te_idx  = row_indices["test"]

    X_train, y_train = X[tr_idx],  y[tr_idx]
    X_val,   y_val   = X[val_idx], y[val_idx]
    X_test,  y_test  = X[te_idx],  y[te_idx]

    # Scale continuous features; leave binary flags unaffected (they stay 0/1)
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)

    # Balance classes via scale_pos_weight (mirrors transformer's inverse-freq weighting)
    neg = int((y_train == 0).sum())
    pos = int((y_train == 1).sum())
    spw = neg / pos if pos > 0 else 1.0

    model = xgb.XGBClassifier(
        **xgb_params,
        scale_pos_weight = spw,
        seed             = seed,
    )
    model.fit(
        X_train, y_train,
        eval_set        = [(X_val, y_val)],
        verbose         = False,
    )

    probs      = model.predict_proba(X_test)[:, 1]
    test_auroc = float(roc_auc_score(y_test, probs))
    test_auprc = float(average_precision_score(y_test, probs, pos_label=1))
    test_loss  = float(log_loss(y_test, probs))
    preds_hard = (probs >= 0.5).astype(int)
    _, test_sensitivity, test_f1, _ = precision_recall_fscore_support(
        y_test, preds_hard, average="binary", pos_label=1, zero_division=0
    )
    cm = confusion_matrix(y_test, preds_hard, labels=[0, 1])
    tn, fp = cm[0, 0], cm[0, 1]
    test_specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    elapsed = time.time() - t0
    print(f"    seed={seed}  AUROC={test_auroc:.4f}  AUPRC={test_auprc:.4f}  "
          f"F1={test_f1:.4f}  loss={test_loss:.4f}  "
          f"best_iter={model.best_iteration}  ({elapsed:.1f}s)")

    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "test_metrics.json", "w") as f:
        json.dump({
            "auroc":       test_auroc,
            "auprc":       test_auprc,
            "f1":          float(test_f1),
            "sensitivity": float(test_sensitivity),
            "specificity": test_specificity,
            "loss":        test_loss,
        }, f, indent=2)

    cfg = {
        "variant":  variant,
        "seed":     seed,
        "n_train":  len(tr_idx),
        "n_val":    len(val_idx),
        "n_test":   len(te_idx),
        "n_features": X.shape[1],
        "scale_pos_weight": spw,
        "best_iteration": model.best_iteration,
        "elapsed_s": elapsed,
        # store fusion/use_time/use_age so compare_ablations.py can read them
        "fusion":   "xgboost",
        "use_time": False,
        "use_age":  False,
        **xgb_params,
    }
    with open(output_dir / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)

    return {
        "auroc":       test_auroc,
        "auprc":       test_auprc,
        "f1":          float(test_f1),
        "sensitivity": float(test_sensitivity),
        "specificity": test_specificity,
        "loss":        test_loss,
    }


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run XGBoost baselines for cardiotoxicity prediction.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--variants", nargs="+", choices=VARIANTS, default=VARIANTS,
        help="Which feature variants to run.",
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=SEEDS,
        help="Random seeds for the train/val/test split.",
    )
    parser.add_argument(
        "--experiment", default="Jul2_ablations",
        help="Experiment folder name under experiment_outputs/.",
    )
    parser.add_argument(
        "--tokenization-dir", default=str(TOKENIZATION_DIR),
        help="Path to a tokenization_outputs/<name>/ directory.",
    )
    parser.add_argument(
        "--cohort-table", default=str(COHORT_TABLE),
        help="Path to final_cycle_binary_modeling_table.parquet.",
    )
    args = parser.parse_args()

    tok_dir    = Path(args.tokenization_dir)
    cohort_path = Path(args.cohort_table)
    out_root   = Path("experiment_outputs") / args.experiment

    # ── load tokenized data ───────────────────────────────────────────────────
    print(f"Loading tokenized data from: {tok_dir}")
    samples       = pd.read_parquet(tok_dir / "samples.parquet")
    labels_tensor = torch.load(tok_dir / "labels.pt", weights_only=True)
    y = labels_tensor.numpy().astype(np.int32)

    # ── load cohort table ─────────────────────────────────────────────────────
    print(f"Loading cohort table from: {cohort_path}")
    cohort = pd.read_parquet(cohort_path)

    # ── build feature matrices (once, reused across seeds) ───────────────────
    print("Building feature matrices …")
    X_clinical = None
    X_bow      = None

    if any(v in args.variants for v in ("clinical", "combined")):
        X_clinical = build_clinical_features(cohort, samples)
        print(f"  clinical  shape: {X_clinical.shape}")

    if any(v in args.variants for v in ("bow", "combined")):
        print(f"  loading concept_ids and attention_mask for BoW …")
        concept_ids    = torch.load(tok_dir / "concept_ids.pt",    weights_only=True)
        attention_mask = torch.load(tok_dir / "attention_mask.pt", weights_only=True)
        with open(tok_dir / "vocab.json") as f:
            vocab = json.load(f)
        concept_vocab: dict[str, int] = vocab["concept_vocab"]
        vocab_size = len(concept_vocab)
        special_token_ids = {
            v for k, v in concept_vocab.items()
            if k in {"[PAD]", "[CLS]", "[MASK]", "[UNK]", "[V_START]", "[V_END]", "[SEP]"}
            or k.startswith("W") or k.startswith("M") or k.startswith("LT") or k.startswith("[")
        }
        print(f"  building BoW (vocab_size={vocab_size}) — may take a moment …")
        X_bow = build_bow_features(concept_ids, attention_mask, vocab_size, special_token_ids)
        print(f"  bow       shape: {X_bow.shape}")

    feature_matrices: dict[str, np.ndarray] = {}
    if "clinical"  in args.variants: feature_matrices["clinical"]  = X_clinical
    if "bow"       in args.variants: feature_matrices["bow"]        = X_bow
    if "combined"  in args.variants:
        feature_matrices["combined"] = np.concatenate([X_clinical, X_bow], axis=1)

    # ── run ───────────────────────────────────────────────────────────────────
    print(f"\nRunning {len(args.variants)} variant(s) × {len(args.seeds)} seed(s) "
          f"→ {len(args.variants) * len(args.seeds)} total runs")
    print(f"Output root: {out_root}\n")

    all_results: dict[str, list[dict]] = {}

    for variant, X in feature_matrices.items():
        label = f"XGB-{variant}"
        print(f"── {label}  (n_features={X.shape[1]}) ──")
        all_results[label] = []

        for seed in args.seeds:
            row_indices = compute_row_indices(samples, seed)
            out_dir     = out_root / label / f"seed{seed}"

            metrics = run_one(
                variant     = variant,
                seed        = seed,
                X           = X,
                y           = y,
                row_indices = row_indices,
                output_dir  = out_dir,
                xgb_params  = XGB_PARAMS,
            )
            all_results[label].append(metrics)

        runs = all_results[label]
        print(f"  → mean AUROC={np.mean([r['auroc'] for r in runs]):.4f}  "
              f"AUPRC={np.mean([r['auprc'] for r in runs]):.4f}  "
              f"F1={np.mean([r['f1'] for r in runs]):.4f}\n")

    # ── summary ───────────────────────────────────────────────────────────────
    print("=" * 70)
    print(f"{'Label':<20}  {'AUROC':>10}  {'AUPRC':>10}  {'F1':>8}  {'Sens':>8}  {'Spec':>8}")
    print("=" * 70)
    for label, runs in all_results.items():
        print(
            f"  {label:<20}"
            f"  {np.mean([r['auroc'] for r in runs]):.4f}±{np.std([r['auroc'] for r in runs]):.4f}"
            f"  {np.mean([r['auprc'] for r in runs]):.4f}±{np.std([r['auprc'] for r in runs]):.4f}"
            f"  {np.mean([r['f1'] for r in runs]):.4f}"
            f"  {np.mean([r['sensitivity'] for r in runs]):.4f}"
            f"  {np.mean([r['specificity'] for r in runs]):.4f}"
        )
    print("=" * 70)
    print(f"\nTo compare with transformer ablations:")
    print(f"  python evaluation/compare_ablations.py {out_root}")


if __name__ == "__main__":
    main()
