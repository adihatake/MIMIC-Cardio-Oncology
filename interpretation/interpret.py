"""
interpret.py

xAI interpretability for EHR_Encoder cardiotoxicity predictions.

Three techniques
----------------
1. Raw per-head attention     — (seq × seq) heatmap per layer and head
2. Attention rollout          — cross-layer aggregated CLS relevance
                               (Abnar & Zuidema 2020, "Quantifying Attention Flow")
3. Integrated Gradients (IG)  — token attribution via Captum;
                               baseline is an all-zero sequence in post-LayerNorm
                               embedding space (not PAD token embeddings)

Usage
-----
  # Explain sample at row index 42 in the full tokenized dataset
  python interpretation/interpret.py \\
      --model-dir experiment_outputs/run1 \\
      --data-dir  tokenization_outputs/Jul17_512_all_labs \\
      --sample-idx 42

  # Explain a specific patient + cycle
  python interpretation/interpret.py \\
      --model-dir experiment_outputs/run1 \\
      --data-dir  tokenization_outputs/Jul17_512_all_labs \\
      --subject-id 12345 --cycle-number 2

  # More IG integration steps for accuracy (default 100)
  python interpretation/interpret.py ... --ig-steps 200

  # Skip IG if captum is not installed
  python interpretation/interpret.py ... --skip-ig

  # Show top 20 tokens instead of 30 in bar charts
  python interpretation/interpret.py ... --top-k 20

Outputs  (written to --output-dir)
--------------------------------------
  attention_L{i}_H{j}.png              attention heatmap per (layer, head); token labels
                                       include [V{n}] visit suffix so repeated features
                                       (same lab at different visits) are distinguishable
  attention_rollout.png                rollout relevance bar chart (top-k tokens, CLS excluded)
  integrated_gradients.png             signed IG bar chart — red = pro-toxic, blue = protective
                                       sorted by absolute value; CLS excluded
  integrated_gradients_magnitude.png   unsigned L2-norm IG bar chart (reference)
  attributions.csv                     per-token rollout + IG scores (signed + L2), sorted by L2
  feature_importance.csv               per-feature-type aggregation: mean/std IG, count,
                                       mean visit, mean days before prediction
  feature_importance.png               top-k features by mean |signed IG| with ±1 std error bars
                                       and per-bar annotation of occurrence count + mean timing
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # non-interactive — works in HPC/SSH environments
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from model_src.ehr_encoder import EHR_Encoder


# ── Human-readable names for common MIMIC-IV lab itemids ─────────────────────
LAB_NAMES: dict[str, str] = {
    "50963": "NTproBNP",
    "51003": "Troponin T",
    "51002": "Troponin I",
    "52642": "Troponin I (alt)",
    "50912": "Creatinine",
    "50868": "Anion Gap",
    "50882": "Bicarbonate",
    "50931": "Glucose",
    "51006": "Urea Nitrogen",
    "51222": "Hemoglobin",
    "51301": "WBC",
    "51265": "Platelets",
    "50983": "Sodium",
    "50971": "Potassium",
    "50960": "Magnesium",
    "50893": "Calcium",
    "50861": "ALT",
    "50878": "AST",
    "50863": "Alk Phosphatase",
    "50885": "Bilirubin",
    "50954": "LDH",
    "50902": "Chloride",
    "50820": "pH",
    "51279": "RBC",
    "50976": "Protein",
}

EVENT_TYPE_COLORS: dict[str, str] = {
    "special":    "#999999",
    "diagnosis":  "#4e79a7",
    "procedure":  "#76b7b2",
    "medication": "#f28e2b",
    "lab":        "#e15759",
}


# ── Model and data loading ────────────────────────────────────────────────────

def _load_model(
    model_dir:         Path,
    device:            torch.device,
    checkpoint_metric: str = "auroc",
) -> tuple[EHR_Encoder, dict, str]:
    with open(model_dir / "config.json") as f:
        cfg = json.load(f)
    model = EHR_Encoder(
        num_concepts   = cfg["num_concepts"],
        max_num_visits = cfg["max_num_visits"],
        d_model        = cfg["d_model"],
        num_heads      = cfg["num_heads"],
        num_layers     = cfg["num_layers"],
        ff_dim         = cfg["ff_dim"],
        dropout        = cfg.get("dropout", 0.1),
        max_seq_len    = cfg["max_seq_len"],
        fusion         = cfg.get("fusion",   "add"),
        use_time       = cfg.get("use_time", False),
        use_age        = cfg.get("use_age",  False),
    ).to(device)
    ckpt_name = f"best_model_{checkpoint_metric}.pt"
    ckpt      = model_dir / ckpt_name
    if not ckpt.exists():
        ckpt      = model_dir / "best_model.pt"
        ckpt_name = "best_model.pt"
    model.load_state_dict(
        torch.load(ckpt, map_location=device, weights_only=True)
    )
    model.eval()
    return model, cfg, ckpt_name


def _load_tensors(data_dir: Path) -> dict[str, torch.Tensor]:
    tensors: dict[str, torch.Tensor] = {
        "concept_ids":  torch.load(data_dir / "concept_ids.pt",  weights_only=True),
        "type_ids":     torch.load(data_dir / "type_ids.pt",     weights_only=True),
        "visit_ids":    torch.load(data_dir / "visit_ids.pt",    weights_only=True),
        "position_ids": torch.load(data_dir / "position_ids.pt", weights_only=True),
        "age_ids":      torch.load(data_dir / "age_ids.pt",      weights_only=True),
        "labels":       torch.load(data_dir / "labels.pt",       weights_only=True),
    }
    for key in ("dates", "age_years"):
        path = data_dir / f"{key}.pt"
        if path.exists():
            tensors[key] = torch.load(path, weights_only=True)
    return tensors


def _load_samples_meta(data_dir: Path) -> list[dict]:
    rows = []
    with open(data_dir / "samples.csv") as f:
        for row in csv.DictReader(f):
            rows.append({
                "subject_id":      int(row["subject_id"]),
                "cycle_number":    int(row["cycle_number"]),
                "prediction_time": row["prediction_time"],
            })
    return rows


def _find_sample_idx(
    samples_meta: list[dict], subject_id: int, cycle_number: int
) -> int:
    for i, m in enumerate(samples_meta):
        if m["subject_id"] == subject_id and m["cycle_number"] == cycle_number:
            return i
    raise ValueError(
        f"No sample found for subject_id={subject_id}, cycle_number={cycle_number}"
    )


def _get_batch(
    tensors: dict[str, torch.Tensor], idx: int, device: torch.device
) -> dict[str, torch.Tensor]:
    """Extract a single sample as a batch-of-one on the target device."""
    return {k: v[[idx]].to(device) for k, v in tensors.items()}


# ── Token decoding ─────────────────────────────────────────────────────────────

def _build_inv_vocab(vocab_path: Path) -> dict[int, str]:
    with open(vocab_path) as f:
        vocab = json.load(f)
    return {v: k for k, v in vocab["concept_vocab"].items()}


def _human_label(raw_token: str) -> str:
    """Convert internal concept string to a short, readable label."""
    if "::" not in raw_token:
        return raw_token
    prefix, value = raw_token.split("::", 1)
    if prefix == "lab":
        itemid = value.split("_")[0]   # strip any quantile-bucket suffix (_Q1 etc.)
        return f"lab:{LAB_NAMES.get(itemid, itemid)}"
    if prefix == "diagnosis":
        return f"dx:{value.split('_')[0]}"
    if prefix == "procedure":
        return f"proc:{value.split('_')[0]}"
    if prefix == "medication":
        return f"rx:{value.split('_')[0][:22]}"
    return raw_token[:28]


def _event_type(raw_token: str) -> str:
    if "::" not in raw_token:
        return "special"
    prefix = raw_token.split("::")[0]
    return prefix if prefix in EVENT_TYPE_COLORS else "special"


def _decode_tokens(
    concept_ids_1d: torch.Tensor,
    inv_vocab: dict[int, str],
    visit_ids_1d: torch.Tensor | None = None,
) -> tuple[list[str], list[str]]:
    """Return (raw_tokens, human_labels) for every position.

    If visit_ids_1d is provided, each label gets a [V{n}] suffix so repeated
    features (e.g. the same lab drawn at multiple visits) are distinguishable
    in heatmaps.
    """
    raw    = [inv_vocab.get(int(cid), f"[{int(cid)}]") for cid in concept_ids_1d]
    labels = [_human_label(r) for r in raw]
    if visit_ids_1d is not None:
        visits = visit_ids_1d.tolist()
        labels = [f"{lbl} [V{v}]" for lbl, v in zip(labels, visits)]
    return raw, labels


def _prediction_days(prediction_time_str: str) -> int:
    """Convert prediction_time string to integer days since 2000-01-01."""
    from datetime import date, datetime
    origin = date(2000, 1, 1)
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            d = datetime.strptime(prediction_time_str, fmt).date()
            return (d - origin).days
        except ValueError:
            continue
    raise ValueError(f"Cannot parse prediction_time: {prediction_time_str}")


# ── Attention rollout ─────────────────────────────────────────────────────────

def compute_rollout(all_attn: list[torch.Tensor]) -> torch.Tensor:
    """
    Attention rollout (Abnar & Zuidema 2020).

    Propagates attention through all layers accounting for residual connections.
    Returns the CLS-row relevance across token positions.

    all_attn  list of (1, num_heads, seq, seq) pre-dropout softmax weights
    Returns   (seq_len,) relevance score for each token position
    """
    device  = all_attn[0].device
    seq_len = all_attn[0].shape[-1]
    I       = torch.eye(seq_len, device=device)

    joint = I.clone()
    for attn in all_attn:
        a = attn[0].mean(dim=0)               # average heads: (seq, seq)
        a = (a + I) / 2                        # residual
        a = a / a.sum(dim=-1, keepdim=True)    # renormalize rows
        joint = a @ joint

    return joint[0]   # CLS row → (seq_len,)


# ── Integrated Gradients ──────────────────────────────────────────────────────

class _EmbeddingWrapper(torch.nn.Module):
    """
    Accepts a pre-computed embedding tensor and runs the transformer
    stack + classifier.  This lets Captum integrate over the continuous
    embedding space without re-entering the discrete token lookup.
    concept_ids are forwarded only to rebuild the padding mask.
    """
    def __init__(self, model: EHR_Encoder) -> None:
        super().__init__()
        self.model = model

    def forward(
        self, embedding: torch.Tensor, concept_ids: torch.Tensor
    ) -> torch.Tensor:
        padding_mask = (concept_ids != 0).long()
        x = embedding
        for layer in self.model.layers:
            x = layer(x, padding_mask)
        x   = self.model.norm(x)
        cls = self.model.cls_dropout(x[:, 0, :])
        return self.model.classifier(cls)


def compute_integrated_gradients(
    model: EHR_Encoder,
    batch: dict[str, torch.Tensor],
    target_class: int = 1,
    n_steps: int = 100,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """
    Compute Integrated Gradients for each token position.

    Baseline: all-zero tensor in post-LayerNorm embedding space.
    This represents a neutral "nothing here" sequence, distinct from
    the learned PAD-token embedding.

    Returns
        token_attr_l2      (seq_len,) L2-norm of per-token attribution (unsigned magnitude)
        token_attr_signed  (seq_len,) sum over d_model (signed: + = pro-toxic, - = protective)
        convergence_delta  |Σ attr - (f(x) - f(baseline))|  (lower is better)
    """
    try:
        from captum.attr import IntegratedGradients
    except ImportError:
        raise ImportError(
            "captum is required for Integrated Gradients.\n"
            "Install with:  pip install captum"
        )

    model.eval()

    with torch.no_grad():
        embedding = model.embedding(
            batch["concept_ids"],
            batch["type_ids"],
            batch["visit_ids"],
            batch["position_ids"],
            batch["age_ids"],
            batch.get("dates"),
            batch.get("age_years"),
        )   # (1, seq_len, d_model)

    # Baseline: zero sequence in embedding space — not the PAD embedding,
    # just literal zeros.  The interpolation path is  baseline → embedding.
    baseline  = torch.zeros_like(embedding)
    embedding = embedding.detach().requires_grad_(True)

    wrapper = _EmbeddingWrapper(model)
    ig      = IntegratedGradients(wrapper)

    attributions, delta = ig.attribute(
        inputs                  = embedding,
        baselines               = baseline,
        additional_forward_args = (batch["concept_ids"],),
        target                  = target_class,
        n_steps                 = n_steps,
        return_convergence_delta= True,
    )
    attr = attributions[0].detach()                # (seq_len, d_model)
    token_attr_l2     = attr.norm(dim=-1)          # unsigned magnitude
    token_attr_signed = attr.sum(dim=-1)           # signed: + pushes toward target class
    return token_attr_l2, token_attr_signed, float(delta.abs().item())


# ── Visualization ─────────────────────────────────────────────────────────────

def _legend_handles() -> list[mpatches.Patch]:
    return [
        mpatches.Patch(color=c, label=t) for t, c in EVENT_TYPE_COLORS.items()
    ]


def plot_attention_heatmap(
    attn: torch.Tensor,
    token_labels: list[str],
    layer: int,
    head: int,
    output_path: Path,
    max_tokens: int = 60,
) -> None:
    """
    attn: (seq_len, seq_len) single-head attention weights.
    Rows = queries (who attends), columns = keys (what is attended to).
    Row 0 is the CLS token used for classification.
    """
    n      = min(len(token_labels), max_tokens)
    data   = attn[:n, :n].cpu().float().numpy()
    labels = token_labels[:n]

    side = max(8.0, n * 0.22)
    fig, ax = plt.subplots(figsize=(side, side * 0.9))
    im = ax.imshow(data, aspect="auto", cmap="viridis", vmin=0, vmax=data.max())
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=90, fontsize=6)
    ax.set_yticklabels(labels, fontsize=6)
    ax.set_title(f"Attention  —  Layer {layer}  Head {head}", fontsize=11, pad=10)
    ax.set_xlabel("Key (attended to)", fontsize=9)
    ax.set_ylabel("Query (attending)", fontsize=9)

    # Highlight the CLS row
    ax.axhline(y=0.5, color="red", linewidth=0.8, alpha=0.6)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_token_bar(
    scores: torch.Tensor,
    token_labels: list[str],
    raw_tokens: list[str],
    title: str,
    ylabel: str,
    output_path: Path,
    top_k: int = 30,
    convergence_delta: float | None = None,
) -> None:
    """Bar chart of per-token scores, coloured by event type, top-k tokens shown."""
    scores_np = scores.cpu().float().numpy()
    order     = np.argsort(scores_np)[::-1][:top_k]

    top_scores = scores_np[order]
    top_labels = [token_labels[i] for i in order]
    top_raw    = [raw_tokens[i]   for i in order]
    colors     = [EVENT_TYPE_COLORS.get(_event_type(r), "#999999") for r in top_raw]

    fig, ax = plt.subplots(figsize=(max(10, top_k * 0.45), 5))
    ax.bar(range(top_k), top_scores, color=colors, edgecolor="white", linewidth=0.4)
    ax.set_xticks(range(top_k))
    ax.set_xticklabels(top_labels, rotation=55, ha="right", fontsize=7)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=11)
    ax.legend(handles=_legend_handles(), fontsize=7, loc="upper right")

    if convergence_delta is not None:
        ax.text(
            0.01, 0.97, f"convergence Δ = {convergence_delta:.5f}",
            transform=ax.transAxes, fontsize=7, va="top", color="gray",
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_signed_ig_bar(
    scores: torch.Tensor,
    token_labels: list[str],
    raw_tokens: list[str],
    title: str,
    output_path: Path,
    top_k: int = 30,
    convergence_delta: float | None = None,
) -> None:
    """
    Bar chart of signed IG scores sorted by absolute value, coloured by direction.

    Positive (red)  → token pushes prediction toward the positive (cardiotoxic) class.
    Negative (blue) → token is protective / pushes prediction away from cardiotoxicity.

    CLS is excluded — its attribution is trivially high and uninformative.
    """
    scores_np = scores.cpu().float().numpy()
    order     = np.argsort(np.abs(scores_np))[::-1][:top_k]

    top_scores = scores_np[order]
    top_labels = [token_labels[i] for i in order]
    top_raw    = [raw_tokens[i]   for i in order]
    colors     = ["#c0392b" if s >= 0 else "#2980b9" for s in top_scores]

    fig, ax = plt.subplots(figsize=(max(10, top_k * 0.45), 5))
    ax.bar(range(len(top_scores)), top_scores, color=colors,
           edgecolor="white", linewidth=0.4)
    ax.axhline(0, color="black", linewidth=0.6, alpha=0.5)
    ax.set_xticks(range(len(top_scores)))
    ax.set_xticklabels(top_labels, rotation=55, ha="right", fontsize=7)
    ax.set_ylabel("Signed IG attribution (sum over d_model)", fontsize=9)
    ax.set_title(title, fontsize=11)

    import matplotlib.patches as mpatches
    legend_handles = [
        mpatches.Patch(color="#c0392b", label="pro-toxic  (+)"),
        mpatches.Patch(color="#2980b9", label="protective (−)"),
    ]
    ax.legend(handles=legend_handles, fontsize=8, loc="upper right")

    if convergence_delta is not None:
        ax.text(
            0.01, 0.97, f"convergence Δ = {convergence_delta:.5f}",
            transform=ax.transAxes, fontsize=7, va="top", color="gray",
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── Feature aggregation ───────────────────────────────────────────────────────

def build_feature_aggregation(
    raw_tokens:      list[str],
    token_labels:    list[str],
    rollout_scores:  torch.Tensor,
    visit_ids:       torch.Tensor,
    days_before:     torch.Tensor,
    ig_l2:           torch.Tensor | None = None,
    ig_signed:       torch.Tensor | None = None,
) -> pd.DataFrame:
    """
    Group all token occurrences by their base type (raw_token) and summarise
    attribution scores + temporal location across occurrences.

    Columns
    -------
    token           raw concept string (base type, no visit suffix)
    label           human-readable name
    event_type      diagnosis / lab / medication / procedure / special
    count           number of occurrences in this sequence
    mean_rollout    mean rollout relevance across occurrences
    mean_ig_l2      mean unsigned IG magnitude  (None if IG was skipped)
    mean_ig_signed  mean signed IG              (None if IG was skipped)
    std_ig_signed   std of signed IG across occurrences
    mean_visit      mean visit number
    mean_days_before mean days before prediction (negative = before prediction)
    earliest_days   earliest (most negative) occurrence
    latest_days     latest occurrence
    """
    rows = []
    n = len(raw_tokens)
    roll_np   = rollout_scores.cpu().float().numpy()
    vis_np    = visit_ids.cpu().numpy()
    days_np   = days_before.cpu().float().numpy()
    ig_l2_np  = ig_l2.cpu().float().numpy()     if ig_l2     is not None else None
    ig_sig_np = ig_signed.cpu().float().numpy() if ig_signed is not None else None

    for i in range(n):
        rows.append({
            "token":      raw_tokens[i],
            "label":      _human_label(raw_tokens[i]),
            "event_type": _event_type(raw_tokens[i]),
            "rollout":    float(roll_np[i]),
            "ig_l2":      float(ig_l2_np[i])  if ig_l2_np  is not None else None,
            "ig_signed":  float(ig_sig_np[i]) if ig_sig_np is not None else None,
            "visit":      int(vis_np[i]),
            "days_before": float(days_np[i]),
        })

    df = pd.DataFrame(rows)
    agg: dict[str, object] = {
        "label":        ("label",       "first"),
        "event_type":   ("event_type",  "first"),
        "count":        ("rollout",     "count"),
        "mean_rollout": ("rollout",     "mean"),
    }
    if ig_l2_np is not None:
        agg["mean_ig_l2"]     = ("ig_l2",     "mean")
        agg["mean_ig_signed"] = ("ig_signed",  "mean")
        agg["std_ig_signed"]  = ("ig_signed",  "std")
    agg["mean_visit"]       = ("visit",      "mean")
    agg["mean_days_before"] = ("days_before", "mean")
    agg["earliest_days"]    = ("days_before", "min")
    agg["latest_days"]      = ("days_before", "max")

    result = df.groupby("token").agg(**{k: v for k, v in agg.items()}).reset_index()
    sort_col = "mean_ig_l2" if ig_l2_np is not None else "mean_rollout"
    return result.sort_values(sort_col, ascending=False).reset_index(drop=True)


def plot_feature_importance(
    feat_df:    pd.DataFrame,
    output_path: Path,
    title:      str,
    top_k:      int = 30,
) -> None:
    """
    Bar chart of top-k features by mean |signed IG|, coloured by direction.
    Error bars show ±1 std across occurrences. Annotation shows mean visit and
    mean days before prediction.
    """
    if "mean_ig_signed" not in feat_df.columns:
        return  # IG was skipped

    df = feat_df.copy()
    df["abs_mean_ig"] = df["mean_ig_signed"].abs()
    df = df.nlargest(top_k, "abs_mean_ig").reset_index(drop=True)

    colors = ["#c0392b" if v >= 0 else "#2980b9" for v in df["mean_ig_signed"]]
    errs   = df["std_ig_signed"].fillna(0).tolist()

    fig, ax = plt.subplots(figsize=(max(10, top_k * 0.5), 6))
    bars = ax.bar(
        range(len(df)),
        df["mean_ig_signed"],
        color=colors,
        edgecolor="white",
        linewidth=0.4,
        yerr=errs,
        capsize=3,
        error_kw={"elinewidth": 0.8, "ecolor": "gray"},
    )
    ax.axhline(0, color="black", linewidth=0.6, alpha=0.5)

    # Annotate each bar with count and mean days before prediction
    for i, (_, row) in enumerate(df.iterrows()):
        yval  = float(row["mean_ig_signed"])
        count = int(row["count"])
        days  = float(row["mean_days_before"])
        va    = "bottom" if yval >= 0 else "top"
        offset = ax.get_ylim()[1] * 0.02 if yval >= 0 else ax.get_ylim()[0] * 0.02
        ax.text(
            i, yval + (abs(offset) if yval >= 0 else -abs(offset)),
            f"n={count}\n{days:.0f}d",
            ha="center", va=va, fontsize=5.5, color="#444444",
        )

    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(df["label"], rotation=55, ha="right", fontsize=7)
    ax.set_ylabel("Mean signed IG attribution (±1 std)", fontsize=9)
    ax.set_title(title, fontsize=11)

    import matplotlib.patches as mpatches
    ax.legend(handles=[
        mpatches.Patch(color="#c0392b", label="pro-toxic  (+)"),
        mpatches.Patch(color="#2980b9", label="protective (−)"),
    ], fontsize=8, loc="upper right")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── CSV export ─────────────────────────────────────────────────────────────────

def save_attributions_csv(
    output_path: Path,
    token_labels: list[str],
    raw_tokens: list[str],
    rollout_scores: torch.Tensor,
    ig_scores_l2: torch.Tensor | None,
    ig_scores_signed: torch.Tensor | None,
) -> None:
    rows = []
    for i, (label, raw, roll) in enumerate(
        zip(token_labels, raw_tokens, rollout_scores.cpu().tolist())
    ):
        row: dict = {"position": i, "token": raw, "label": label, "rollout_score": roll}
        if ig_scores_l2 is not None:
            row["ig_score_l2"]     = float(ig_scores_l2[i].item())
            row["ig_score_signed"] = float(ig_scores_signed[i].item())
        rows.append(row)

    df = pd.DataFrame(rows)
    sort_col = "ig_score_l2" if ig_scores_l2 is not None else "rollout_score"
    df = df.sort_values(sort_col, ascending=False)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


# ── Main ──────────────────────────────────────────────────────────────────────

def explain(
    model_dir:         Path,
    data_dir:          Path,
    sample_idx:        int | None,
    subject_id:        int | None,
    cycle_number:      int | None,
    output_dir:        Path,
    ig_steps:          int  = 100,
    top_k:             int  = 30,
    device:            str  = "auto",
    skip_ig:           bool = False,
    checkpoint_metric: str  = "auroc",
) -> None:

    if device == "auto":
        _device = torch.device(
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
    else:
        _device = torch.device(device)

    print(f"Device       : {_device}")

    # ── load model + data ────────────────────────────────────────────────────
    model, cfg, ckpt_name = _load_model(model_dir, _device, checkpoint_metric)
    num_layers = cfg["num_layers"]
    num_heads  = cfg["num_heads"]
    print(f"Model        : {model_dir.name}  "
          f"({num_layers}L × {num_heads}H × {cfg['d_model']}d)")
    print(f"Checkpoint   : {ckpt_name}")

    tensors      = _load_tensors(data_dir)
    samples_meta = _load_samples_meta(data_dir)
    inv_vocab    = _build_inv_vocab(data_dir / "vocab.json")

    # ── resolve sample ────────────────────────────────────────────────────────
    if sample_idx is None:
        if subject_id is None or cycle_number is None:
            raise ValueError(
                "Provide either --sample-idx or both --subject-id and --cycle-number"
            )
        sample_idx = _find_sample_idx(samples_meta, subject_id, cycle_number)

    meta  = samples_meta[sample_idx]
    label = int(tensors["labels"][sample_idx].item())
    print(f"Sample       : idx={sample_idx}  "
          f"subject={meta['subject_id']}  cycle={meta['cycle_number']}  "
          f"pred_time={meta['prediction_time']}  true_label={label}")

    batch = _get_batch(tensors, sample_idx, _device)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── forward pass with attention ──────────────────────────────────────────
    with torch.no_grad():
        logits, all_attn = model(
            batch["concept_ids"],
            batch["type_ids"],
            batch["visit_ids"],
            batch["position_ids"],
            batch["age_ids"],
            batch.get("dates"),
            batch.get("age_years"),
            return_attention=True,
        )

    prob_pos = float(F.softmax(logits, dim=-1)[0, 1].item())
    print(f"Prediction   : P(cardiotoxic) = {prob_pos:.4f}  (true label = {label})")

    # Limit all visualization to non-padding token positions.
    # CLS is always position 0 — exclude it from bar charts (its attribution
    # is trivially high since the classifier reads directly from it).
    concept_ids_1d = batch["concept_ids"][0].cpu()
    visit_ids_1d   = batch["visit_ids"][0].cpu()
    dates_1d       = batch["dates"][0].cpu() if "dates" in batch else None
    n_active       = int((concept_ids_1d != 0).sum().item())

    # Heatmap labels include [V{n}] visit suffix so repeated features are distinguishable
    raw_tokens, token_labels = _decode_tokens(
        concept_ids_1d[:n_active], inv_vocab, visit_ids_1d[:n_active]
    )

    # Days before prediction for each token (negative = before prediction time)
    pred_days   = _prediction_days(meta["prediction_time"])
    days_before = (dates_1d[:n_active].float() - pred_days) if dates_1d is not None \
                  else torch.zeros(n_active)

    # Slices used for bar charts and aggregation: skip position 0 (CLS)
    raw_content    = raw_tokens[1:]
    labels_content = token_labels[1:]
    visit_content  = visit_ids_1d[1:n_active]
    days_content   = days_before[1:]
    n_content      = len(raw_content)

    title_prefix = (
        f"subject {meta['subject_id']} cycle {meta['cycle_number']} "
        f"| P(tox)={prob_pos:.3f}"
    )

    # ── 1. Raw per-head attention heatmaps ───────────────────────────────────
    print(f"\nSaving attention heatmaps ({num_layers} layers × {num_heads} heads)...")
    for li, layer_attn in enumerate(all_attn):
        for hi in range(num_heads):
            # (n_active, n_active) single-head weights
            head_attn = layer_attn[0, hi, :n_active, :n_active]
            plot_attention_heatmap(
                attn         = head_attn,
                token_labels = token_labels,
                layer        = li,
                head         = hi,
                output_path  = output_dir / f"attention_L{li}_H{hi}.png",
            )
    print(f"  {num_layers * num_heads} heatmaps written to {output_dir}")

    # ── 2. Attention rollout ─────────────────────────────────────────────────
    print("\nComputing attention rollout...")
    rollout        = compute_rollout(all_attn)                 # (seq_len,)
    rollout_active = rollout[:n_active].cpu()
    rollout_content = rollout_active[1:]                       # skip CLS

    plot_token_bar(
        scores       = rollout_content,
        token_labels = labels_content,
        raw_tokens   = raw_content,
        title        = f"Attention Rollout — {title_prefix}",
        ylabel       = "Rollout relevance score",
        output_path  = output_dir / "attention_rollout.png",
        top_k        = min(top_k, n_content),
    )
    print("  Saved attention_rollout.png")

    # ── 3. Integrated Gradients ──────────────────────────────────────────────
    ig_l2: torch.Tensor | None     = None
    ig_signed: torch.Tensor | None = None
    if not skip_ig:
        print(
            f"\nComputing Integrated Gradients "
            f"({ig_steps} steps, baseline = zero embedding)..."
        )
        try:
            ig_full_l2, ig_full_signed, delta = compute_integrated_gradients(
                model, batch, target_class=1, n_steps=ig_steps,
            )
            # Skip CLS (position 0) for all IG plots
            ig_l2    = ig_full_l2[1:n_active].cpu()
            ig_signed = ig_full_signed[1:n_active].cpu()

            delta_warn = "  ✓" if delta < 0.01 else "  ⚠ consider more --ig-steps"
            print(f"  Convergence delta: {delta:.6f}{delta_warn}")

            # Signed IG — primary plot: sort by absolute value, colour by direction
            plot_signed_ig_bar(
                scores            = ig_signed,
                token_labels      = labels_content,
                raw_tokens        = raw_content,
                title             = f"Integrated Gradients (signed) — {title_prefix}",
                output_path       = output_dir / "integrated_gradients.png",
                top_k             = min(top_k, n_content),
                convergence_delta = delta,
            )
            print("  Saved integrated_gradients.png")

            # L2-norm IG — unsigned magnitude reference
            plot_token_bar(
                scores            = ig_l2,
                token_labels      = labels_content,
                raw_tokens        = raw_content,
                title             = f"Integrated Gradients (magnitude) — {title_prefix}",
                ylabel            = "IG attribution (L2 norm over d_model)",
                output_path       = output_dir / "integrated_gradients_magnitude.png",
                top_k             = min(top_k, n_content),
                convergence_delta = delta,
            )
            print("  Saved integrated_gradients_magnitude.png")
        except ImportError as e:
            print(f"  Skipping IG: {e}")
    else:
        print("\nIntegrated Gradients skipped (--skip-ig).")

    # ── 4. Attributions CSV ──────────────────────────────────────────────────
    save_attributions_csv(
        output_path      = output_dir / "attributions.csv",
        token_labels     = labels_content,
        raw_tokens       = raw_content,
        rollout_scores   = rollout_content,
        ig_scores_l2     = ig_l2,
        ig_scores_signed = ig_signed,
    )
    print(f"\nAttributions CSV → {output_dir / 'attributions.csv'}")

    # ── 5. Feature aggregation ────────────────────────────────────────────────
    # Groups repeated occurrences of the same token type, computes mean IG
    # and temporal location (mean days before prediction, mean visit number).
    feat_df = build_feature_aggregation(
        raw_tokens     = raw_content,
        token_labels   = labels_content,
        rollout_scores = rollout_content,
        visit_ids      = visit_content,
        days_before    = days_content,
        ig_l2          = ig_l2,
        ig_signed      = ig_signed,
    )
    feat_csv = output_dir / "feature_importance.csv"
    feat_df.to_csv(feat_csv, index=False)
    print(f"Feature summary CSV → {feat_csv}")

    plot_feature_importance(
        feat_df     = feat_df,
        output_path = output_dir / "feature_importance.png",
        title       = f"Feature importance (aggregated) — {title_prefix}",
        top_k       = min(top_k, len(feat_df)),
    )
    print(f"Feature importance plot → {output_dir / 'feature_importance.png'}")
    print(f"All outputs → {output_dir}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Interpret EHR_Encoder predictions (attention + Integrated Gradients).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--model-dir", required=True,
        help="Experiment directory containing config.json and best_model_*.pt checkpoints.",
    )
    p.add_argument(
        "--data-dir", default=None,
        help="Tokenization directory (default: read from model config.json)",
    )
    p.add_argument(
        "--sample-idx", type=int, default=None,
        help="Row index in the full tokenized dataset to explain",
    )
    p.add_argument(
        "--subject-id", type=int, default=None,
        help="MIMIC subject_id (use with --cycle-number)",
    )
    p.add_argument(
        "--cycle-number", type=int, default=None,
        help="Cycle number to explain (use with --subject-id)",
    )
    p.add_argument(
        "--output-dir", default=None,
        help="Output directory (default: interpretation/outputs/<subject>_cycle<n>/)",
    )
    p.add_argument(
        "--ig-steps", type=int, default=100,
        help="Integration steps for Integrated Gradients (more = more accurate, slower)",
    )
    p.add_argument(
        "--skip-ig", action="store_true",
        help="Skip Integrated Gradients (useful if captum is not installed)",
    )
    p.add_argument(
        "--top-k", type=int, default=30,
        help="Number of top-scoring tokens to show in bar charts",
    )
    p.add_argument(
        "--device", default="auto",
        help="Device: 'auto', 'cpu', 'cuda', or 'mps'",
    )
    p.add_argument(
        "--checkpoint-metric", default="auroc", dest="checkpoint_metric",
        choices=["auroc", "auprc", "f1", "sensitivity", "specificity"],
        help="Which per-metric checkpoint to load (best_model_{metric}.pt). "
             "Falls back to best_model.pt for older runs.",
    )
    return p.parse_args()


def main() -> None:
    args      = _parse_args()
    model_dir = Path(args.model_dir)

    if not (model_dir / "config.json").exists():
        print(f"Missing config.json in {model_dir}")
        sys.exit(1)

    ckpt = model_dir / f"best_model_{args.checkpoint_metric}.pt"
    if not ckpt.exists():
        ckpt = model_dir / "best_model.pt"
    if not ckpt.exists():
        print(f"No checkpoint found in {model_dir}. "
              f"Expected best_model_{args.checkpoint_metric}.pt or best_model.pt.")
        sys.exit(1)

    with open(model_dir / "config.json") as f:
        cfg = json.load(f)

    data_dir_str = args.data_dir or cfg.get("data_dir")
    if not data_dir_str:
        print("Cannot determine data_dir. Pass --data-dir explicitly.")
        sys.exit(1)
    data_dir = Path(data_dir_str)

    if args.output_dir:
        output_dir = Path(args.output_dir)
    elif args.subject_id is not None and args.cycle_number is not None:
        output_dir = (
            REPO_ROOT / "interpretation" / "outputs"
            / f"{args.subject_id}_cycle{args.cycle_number}"
        )
    elif args.sample_idx is not None:
        output_dir = (
            REPO_ROOT / "interpretation" / "outputs"
            / f"sample_{args.sample_idx}"
        )
    else:
        print("Provide --sample-idx or both --subject-id and --cycle-number.")
        sys.exit(1)

    explain(
        model_dir          = model_dir,
        data_dir           = data_dir,
        sample_idx         = args.sample_idx,
        subject_id         = args.subject_id,
        cycle_number       = args.cycle_number,
        output_dir         = output_dir,
        ig_steps           = args.ig_steps,
        top_k              = args.top_k,
        device             = args.device,
        skip_ig            = args.skip_ig,
        checkpoint_metric  = args.checkpoint_metric,
    )


if __name__ == "__main__":
    main()
