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
    "special":    "#999999",
    "diagnosis":  "#4e79a7",
    "procedure":  "#76b7b2",
    "medication": "#f28e2b",
    "lab":        "#e15759",
}

LABEL_COLORS: dict[int, str] = {0: "#2980b9", 1: "#c0392b"}
LABEL_NAMES:  dict[int, str] = {0: "Non-cardiotoxic", 1: "Cardiotoxic"}

PERTURB_COLORS: list[str] = [
    "#e41a1c", "#377eb8", "#4daf4a", "#ff7f00",
    "#984ea3", "#a65628", "#f781bf", "#999999",
]


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
