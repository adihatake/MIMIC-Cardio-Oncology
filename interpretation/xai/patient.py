"""
patient.py

Per-patient xAI visualizations:
  - CLS trajectory across chemotherapy cycles
  - Attention rollout comparison across cycles (bar + heatmap)
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from interpretation.interpret import (
    _decode_tokens, _event_type,
    compute_rollout, compute_integrated_gradients,
)
from .utils import EVENT_TYPE_COLORS, LABEL_COLORS, LABEL_NAMES
from model_src.ehr_encoder import EHR_Encoder


# ── Per-cycle data extraction ─────────────────────────────────────────────────

def get_patient_cycle_data(
    model: EHR_Encoder,
    tensors: dict[str, torch.Tensor],
    inv_vocab: dict[int, str],
    samples_df: pd.DataFrame,
    patient_indices: list[int],
    device: torch.device,
    ig_steps: int = 100,
    skip_ig: bool = False,
) -> list[dict]:
    """
    Run inference for every cycle of a patient and collect interpretability data.

    Returns one dict per cycle with keys:
      idx, cycle_number, label, prob, cls_embedding,
      raw_tokens, token_labels, rollout, n_active,
      ig_l2, ig_signed  (None when skip_ig=True)
    """
    results: list[dict] = []
    for idx in patient_indices:
        row   = samples_df.iloc[idx]
        batch = {k: v[[idx]].to(device) for k, v in tensors.items()}

        # Forward with attention
        with torch.no_grad():
            logits, all_attn = model(
                batch["concept_ids"], batch["type_ids"], batch["visit_ids"],
                batch["position_ids"], batch["age_ids"],
                batch.get("dates"), batch.get("age_years"),
                return_attention=True,
            )
        prob = float(F.softmax(logits, dim=-1)[0, 1].item())

        # CLS embedding (post-norm, no dropout)
        with torch.no_grad():
            x = model.embedding(
                batch["concept_ids"], batch["type_ids"], batch["visit_ids"],
                batch["position_ids"], batch["age_ids"],
                batch.get("dates"), batch.get("age_years"),
            )
            pad_mask = (batch["concept_ids"] != 0).long()
            if batch.get("task_ids") is not None and model.task_embed is not None:
                tt       = model.task_embed(batch["task_ids"]).unsqueeze(1)
                x        = torch.cat([tt, x], dim=1)
                tm       = torch.ones(x.size(0), 1, dtype=pad_mask.dtype, device=device)
                pad_mask = torch.cat([tm, pad_mask], dim=1)
            for layer in model.layers:
                x = layer(x, pad_mask)
            x = model.norm(x)
            cls_emb = x[0, 0, :].cpu()

        # Attention rollout
        rollout = compute_rollout(all_attn)

        # Token decoding
        concept_ids_1d = batch["concept_ids"][0].cpu()
        visit_ids_1d   = batch["visit_ids"][0].cpu()
        n_active       = int((concept_ids_1d != 0).sum().item())
        raw_tokens, token_labels = _decode_tokens(
            concept_ids_1d[:n_active], inv_vocab, visit_ids_1d[:n_active]
        )
        rollout_content = rollout[:n_active].cpu()[1:]   # skip CLS

        # Integrated Gradients (optional)
        ig_l2, ig_signed = None, None
        if not skip_ig:
            try:
                il2, isig, _ = compute_integrated_gradients(
                    model, batch, target_class=1, n_steps=ig_steps
                )
                ig_l2    = il2[1:n_active].cpu()
                ig_signed = isig[1:n_active].cpu()
            except Exception as e:
                print(f"  IG failed for idx={idx}: {e}")

        results.append({
            "idx":          idx,
            "cycle_number": int(row["cycle_number"]),
            "label":        int(tensors["labels"][idx].item()),
            "prob":         prob,
            "cls_embedding": cls_emb,
            "raw_tokens":   raw_tokens[1:],     # skip CLS
            "token_labels": token_labels[1:],
            "rollout":      rollout_content,
            "n_active":     n_active,
            "ig_l2":        ig_l2,
            "ig_signed":    ig_signed,
        })

    return results


# ── Plot 3: CLS trajectory ────────────────────────────────────────────────────

def plot_cls_trajectory(
    cycle_data: list[dict],
    patient_coords_2d: np.ndarray,
    subject_id: int,
    coords_background: np.ndarray | None,
    background_labels: np.ndarray | None,
    output_path: Path,
    method_name: str = "UMAP",
) -> None:
    """
    CLS trajectory for one patient overlaid on a faint background of test-set
    samples, with annotated cycle points and arrows showing temporal direction.

    patient_coords_2d : (n_cycles, 2) — already in the same space as background.
    coords_background : (N_bg, 2)     — test/all projections (faint background).
    background_labels : (N_bg,)       — true labels used for background colouring.
    """
    n_cycles  = patient_coords_2d.shape[0]
    cycle_nums = [d["cycle_number"] for d in cycle_data]
    probs      = [d["prob"] for d in cycle_data]
    labels     = [d["label"] for d in cycle_data]
    palette    = plt.cm.plasma(np.linspace(0.15, 0.85, max(n_cycles, 2)))

    fig, ax = plt.subplots(figsize=(9, 7))

    if coords_background is not None and len(coords_background) > 0:
        bg_c = ([LABEL_COLORS.get(int(l), "#aaaaaa") for l in background_labels]
                if background_labels is not None else "#cccccc")
        ax.scatter(coords_background[:, 0], coords_background[:, 1],
                   c=bg_c, alpha=0.07, s=14, linewidths=0,
                   rasterized=True, zorder=1)

    for i in range(n_cycles - 1):
        x0, y0 = patient_coords_2d[i]
        x1, y1 = patient_coords_2d[i + 1]
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="->", color="#333333", lw=1.4),
                    zorder=3)

    for i, (coord, cyc, prob, lbl) in enumerate(
        zip(patient_coords_2d, cycle_nums, probs, labels)
    ):
        ax.scatter(coord[0], coord[1], c=[palette[i]], s=160, zorder=4,
                   edgecolors=LABEL_COLORS.get(lbl, "#888888"), linewidths=2.0)
        ax.annotate(f"  Cycle {cyc}\n  P={prob:.2f}", xy=(coord[0], coord[1]),
                    fontsize=8.5, color="#111111", ha="left", va="bottom", zorder=5)

    ax.set_xlabel(f"{method_name} dim 1", fontsize=10)
    ax.set_ylabel(f"{method_name} dim 2", fontsize=10)
    ax.set_title(f"CLS Trajectory — Patient {subject_id}", fontsize=12)
    ax.legend(handles=[
        mpatches.Patch(color=palette[i], label=f"Cycle {c}")
        for i, c in enumerate(cycle_nums)
    ] + [
        mpatches.Patch(color=LABEL_COLORS[0], label="True: non-toxic (border)"),
        mpatches.Patch(color=LABEL_COLORS[1], label="True: cardiotoxic (border)"),
    ], fontsize=8, loc="upper left")
    ax.text(0.99, 0.01, method_name, transform=ax.transAxes,
            fontsize=8, color="gray", ha="right", va="bottom")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {output_path.name}")


# ── Plot 4a: Rollout per cycle (side-by-side bars) ────────────────────────────

def plot_rollout_per_cycle(
    cycle_data: list[dict],
    subject_id: int,
    output_path: Path,
    top_k: int = 20,
) -> None:
    """
    One horizontal bar chart per cycle arranged side-by-side.
    A shared feature set (union of top features across cycles) lets the reader
    track how attention shifts from cycle to cycle.
    """
    n_cycles = len(cycle_data)
    if n_cycles == 0:
        return

    # Build union of top features
    featured: dict[str, str] = {}   # raw_token → base label
    for cd in cycle_data:
        rollout = cd["rollout"].cpu().float().numpy()
        for i in np.argsort(rollout)[::-1][: top_k // 2 + 1]:
            raw = cd["raw_tokens"][i]
            if raw not in featured:
                featured[raw] = cd["token_labels"][i].split(" [V")[0]

    # Sort by mean rollout across cycles, cap at top_k
    mean_score: dict[str, float] = {
        raw: np.mean([
            float(cd["rollout"][j].item())
            for cd in cycle_data
            for j, r in enumerate(cd["raw_tokens"]) if r == raw
        ])
        for raw in featured
    }
    sorted_tokens = sorted(featured, key=lambda t: -mean_score[t])[:top_k]
    sorted_labels = [featured[t] for t in sorted_tokens]

    fig, axes = plt.subplots(
        1, n_cycles,
        figsize=(5 * n_cycles, max(5, top_k * 0.28)),
        sharey=True,
    )
    if n_cycles == 1:
        axes = [axes]

    for ci, (cd, ax) in enumerate(zip(cycle_data, axes)):
        # Max rollout per base token type across visits
        token_score: dict[str, float] = {}
        for j, raw in enumerate(cd["raw_tokens"]):
            base = raw.split(" [V")[0]
            token_score[base] = max(token_score.get(base, 0.0),
                                    float(cd["rollout"][j].item()))

        scores = np.array([token_score.get(t, 0.0) for t in sorted_tokens])
        colors = [EVENT_TYPE_COLORS.get(_event_type(t), "#999999") for t in sorted_tokens]

        ax.barh(range(len(sorted_tokens)), scores, color=colors,
                edgecolor="white", linewidth=0.4)
        ax.invert_yaxis()
        ax.set_title(
            f"Cycle {cd['cycle_number']}\n"
            f"P={cd['prob']:.2f}  {'tox' if cd['label'] else 'neg'}",
            fontsize=9,
        )
        ax.set_xlabel("Rollout relevance", fontsize=8)
        if ci == 0:
            ax.set_yticks(range(len(sorted_labels)))
            ax.set_yticklabels(sorted_labels, fontsize=7)
        ax.tick_params(axis="x", labelsize=7)

    axes[-1].legend(handles=[
        mpatches.Patch(color=c, label=t)
        for t, c in EVENT_TYPE_COLORS.items() if t != "special"
    ], fontsize=7, loc="lower right")

    fig.suptitle(f"Attention Rollout Across Cycles — Patient {subject_id}", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {output_path.name}")


# ── Plot 4b: Rollout heatmap (features × cycles) ─────────────────────────────

def plot_rollout_heatmap(
    cycle_data: list[dict],
    subject_id: int,
    output_path: Path,
    top_k: int = 25,
) -> None:
    """
    Heatmap with features as rows and cycles as columns.
    Highlights which features are consistently important vs cycle-specific.
    """
    n_cycles = len(cycle_data)
    if n_cycles == 0:
        return

    featured: dict[str, str] = {}
    for cd in cycle_data:
        rollout = cd["rollout"].cpu().float().numpy()
        for i in np.argsort(rollout)[::-1][:top_k]:
            raw = cd["raw_tokens"][i]
            if raw not in featured:
                featured[raw] = cd["token_labels"][i].split(" [V")[0]

    sorted_tokens = sorted(
        featured,
        key=lambda t: max(
            (float(cd["rollout"][j].item())
             for cd in cycle_data
             for j, r in enumerate(cd["raw_tokens"]) if r == t),
            default=0.0,
        ),
        reverse=True,
    )[:top_k]
    sorted_labels = [featured[t] for t in sorted_tokens]
    token_to_row  = {t: i for i, t in enumerate(sorted_tokens)}

    matrix = np.zeros((len(sorted_tokens), n_cycles))
    for ci, cd in enumerate(cycle_data):
        for j, raw in enumerate(cd["raw_tokens"]):
            if raw in token_to_row:
                ri = token_to_row[raw]
                matrix[ri, ci] = max(matrix[ri, ci],
                                     float(cd["rollout"][j].item()))

    col_labels = [f"Cycle {cd['cycle_number']}\nP={cd['prob']:.2f}"
                  for cd in cycle_data]

    fig, ax = plt.subplots(
        figsize=(max(5, n_cycles * 1.5), max(6, top_k * 0.3))
    )
    im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd", vmin=0)
    plt.colorbar(im, ax=ax, fraction=0.04, label="Rollout score")
    ax.set_xticks(range(n_cycles))
    ax.set_xticklabels(col_labels, fontsize=9)
    ax.set_yticks(range(len(sorted_labels)))
    ax.set_yticklabels(sorted_labels, fontsize=7)
    ax.set_title(f"Rollout Heatmap — Patient {subject_id}", fontsize=12)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {output_path.name}")
