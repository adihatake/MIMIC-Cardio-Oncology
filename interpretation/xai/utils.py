"""
utils.py

Shared constants, data-loading helpers, and split utilities for the xAI package.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from model_src.ehr_encoder import EHR_Encoder
from model_src.dataset import _compute_row_indices
from interpretation.interpret import _load_model, _load_tensors

# ── Colour palettes ───────────────────────────────────────────────────────────

LAB_NAMES: dict[str, str] = {
    "50963": "NTproBNP",   "51003": "Troponin T", "51002": "Troponin I",
    "52642": "Troponin I (alt)", "50912": "Creatinine", "50868": "Anion Gap",
    "50882": "Bicarbonate","50931": "Glucose",     "51006": "Urea Nitrogen",
    "51222": "Hemoglobin", "51301": "WBC",         "51265": "Platelets",
    "50983": "Sodium",     "50971": "Potassium",   "50960": "Magnesium",
    "50893": "Calcium",    "50861": "ALT",         "50878": "AST",
    "50863": "Alk Phosphatase", "50885": "Bilirubin", "50954": "LDH",
    "50902": "Chloride",   "50820": "pH",          "51279": "RBC",
    "50976": "Protein",
}

EVENT_TYPE_COLORS: dict[str, str] = {
    "special":    "#aaaaaa",   # neutral gray
    "diagnosis":  "#1f77b4",   # blue
    "procedure":  "#2ca02c",   # green
    "medication": "#ff7f0e",   # orange
    "lab":        "#d62728",   # red
}

LABEL_COLORS: dict[int, str] = {0: "#1f77b4", 1: "#d62728"}
LABEL_NAMES:  dict[int, str] = {0: "Non-cardiotoxic", 1: "Cardiotoxic"}

PERTURB_COLORS: list[str] = [
    "#e41a1c", "#377eb8", "#4daf4a", "#ff7f00",
    "#984ea3", "#a65628", "#f781bf", "#17becf",
]

# Tab10 minus blue (#1f77b4) and red (#d62728), which are reserved for label colours.
# Used to distinguish individual patients when overlaid on dataset CLS space plots.
PATIENT_OVERLAY_COLORS: list[str] = [
    "#ff7f0e",  # orange
    "#9467bd",  # purple
    "#2ca02c",  # green
    "#17becf",  # cyan
    "#e377c2",  # pink
    "#bcbd22",  # olive
    "#8c564b",  # brown
    "#7f7f7f",  # gray
]

# Tab10 palette — designed for maximum categorical discriminability
DRUG_CLASS_COLORS: dict[str, str] = {
    "anthracycline":               "#d62728",  # red
    "immune_checkpoint_inhibitor": "#1f77b4",  # blue
    "her2_targeted":               "#2ca02c",  # green
    "taxane":                      "#9467bd",  # purple
    "fluoropyrimidine":            "#ff7f0e",  # orange
    "vegf_inhibitor":              "#17becf",  # cyan
    "egfr_inhibitor":              "#e377c2",  # pink
    "tyrosine_kinase_inhibitor":   "#8c564b",  # brown
    "proteasome_inhibitor":        "#bcbd22",  # olive
    "immunomodulatory_agent":      "#7f7f7f",  # gray
    "other_oncology":              "#c5b0d5",  # light purple
}


# ── Vocabulary helpers ────────────────────────────────────────────────────────

def load_vocab(data_dir: Path) -> tuple[dict[str, int], dict[int, str]]:
    """Return (vocab, inv_vocab) from vocab.json."""
    with open(data_dir / "vocab.json") as f:
        data = json.load(f)
    vocab: dict[str, int] = data["concept_vocab"]
    inv_vocab: dict[int, str] = {v: k for k, v in vocab.items()}
    return vocab, inv_vocab


def load_samples_df(data_dir: Path) -> pd.DataFrame:
    """Load per-sample metadata. Prefers parquet, falls back to CSV.
    Always returns a 0..N-1 RangeIndex so iloc[i] matches tensor row i.
    """
    parquet = data_dir / "samples.parquet"
    if parquet.exists():
        return pd.read_parquet(parquet).reset_index(drop=True)
    csv_path = data_dir / "samples.csv"
    if csv_path.exists():
        return pd.read_csv(csv_path).reset_index(drop=True)
    raise FileNotFoundError(f"No samples.parquet or samples.csv in {data_dir}")


# ── Top-level setup ───────────────────────────────────────────────────────────

def load_setup(
    model_dir: Path,
    data_dir: Path,
    device: torch.device,
    checkpoint_metric: str = "auroc",
) -> tuple[EHR_Encoder, dict, pd.DataFrame,
           dict[str, torch.Tensor], dict[str, int], dict[int, str],
           dict[str, list[int]]]:
    """
    Load everything needed for xAI analysis.

    Returns
        model, cfg, samples_df, tensors, vocab, inv_vocab, split_indices
    where split_indices = {"train": [...], "val": [...], "test": [...]}.
    """
    model, cfg, ckpt = _load_model(model_dir, device, checkpoint_metric)
    tensors          = _load_tensors(data_dir)
    vocab, inv_vocab = load_vocab(data_dir)
    samples_df       = load_samples_df(data_dir)
    seed             = cfg.get("seed", 42)
    split_indices    = _compute_row_indices(data_dir, seed)

    sizes = {s: len(v) for s, v in split_indices.items()}
    print(f"Model     : {model_dir.name}  "
          f"({cfg['num_layers']}L × {cfg['num_heads']}H × {cfg['d_model']}d)")
    print(f"Checkpoint: {ckpt}")
    print(f"Splits    : {sizes}")

    return model, cfg, samples_df, tensors, vocab, inv_vocab, split_indices


# ── Index selection ───────────────────────────────────────────────────────────

def get_indices(
    split_indices: dict[str, list[int]],
    split_filter: str = "test",
) -> tuple[list[int], list[str]]:
    """
    Return (row_indices, split_labels) for the requested filter.

    split_filter: "test", "val", "train", or "all".
    split_labels records which split each index belongs to (used for
    colouring in combined plots).
    """
    if split_filter == "all":
        combined, labels = [], []
        for s in ("train", "val", "test"):
            combined.extend(split_indices[s])
            labels.extend([s] * len(split_indices[s]))
        return combined, labels
    idxs = split_indices[split_filter]
    return idxs, [split_filter] * len(idxs)


def enrich_samples_with_drug_info(
    samples_df: pd.DataFrame,
    data_dir: Path,
    cohort_table_path: Path | None = None,
) -> pd.DataFrame:
    """
    Join drug class / prescription-count columns from the modeling table onto
    samples_df.  The modeling table is auto-discovered via metadata.json
    (modeling_dir field), or can be supplied explicitly via cohort_table_path.

    Adds columns (when available):
        drug_classes_in_cycle   primary drug class string for this cycle
        drugs_in_cycle          specific drug name(s)
        n_prescription_rows_in_cycle   prescription row count (dose proxy)
        primary_drug_class      drug class at cycle 1 per patient (for all cycles)

    Returns the original df unchanged if the modeling table cannot be found.
    """
    if cohort_table_path is None:
        meta_path = data_dir / "metadata.json"
        if not meta_path.exists():
            print("  [drug info] metadata.json not found — skipping enrichment")
            return samples_df
        with open(meta_path) as f:
            meta = json.load(f)
        modeling_dir = Path(meta.get("modeling_dir", ""))
        if not modeling_dir.is_absolute():
            modeling_dir = REPO_ROOT / modeling_dir
        candidates = [
            modeling_dir / "final_cycle_binary_modeling_table.parquet",
            modeling_dir / "final_cycle_modeling_table.parquet",
        ]
        cohort_table_path = next((p for p in candidates if p.exists()), None)

    if cohort_table_path is None or not cohort_table_path.exists():
        print("  [drug info] modeling table not found — skipping enrichment")
        return samples_df

    keep = ["subject_id", "cycle_number",
            "drug_classes_in_cycle", "drugs_in_cycle",
            "n_prescription_rows_in_cycle"]
    cohort = pd.read_parquet(cohort_table_path, columns=keep)
    cohort["cycle_number"] = cohort["cycle_number"].astype(int)

    enriched = samples_df.merge(cohort, on=["subject_id", "cycle_number"], how="left")

    # primary_drug_class: drug class at each patient's first (lowest) cycle number
    first_class = (
        enriched.dropna(subset=["drug_classes_in_cycle"])
        .sort_values("cycle_number")
        .groupby("subject_id")["drug_classes_in_cycle"]
        .first()
        .rename("primary_drug_class")
    )
    enriched = enriched.merge(first_class, on="subject_id", how="left")

    print(f"  [drug info] enriched with drug class from {cohort_table_path.name} "
          f"({enriched['drug_classes_in_cycle'].notna().sum()}/{len(enriched)} rows matched)")
    return enriched.reset_index(drop=True)


def find_patient_indices(
    samples_df: pd.DataFrame,
    subject_id: int,
    split_indices: dict[str, list[int]] | None = None,
) -> list[int]:
    """Return all row indices for a subject_id, sorted by cycle_number.

    If split_indices is given, only indices present in any split are returned
    (filters out samples excluded from modelling).
    """
    rows = samples_df[samples_df["subject_id"] == subject_id]
    if rows.empty:
        raise ValueError(f"subject_id={subject_id} not found in samples_df")
    idxs = rows.sort_values("cycle_number").index.tolist()
    if split_indices is not None:
        all_idxs = set(
            split_indices["train"] + split_indices["val"] + split_indices["test"]
        )
        idxs = [i for i in idxs if i in all_idxs]
    return idxs
