"""
perturbation.py

Perturbation analysis: modify patient token sequences and observe how the
CLS representation and prediction probability change.

Supported perturbation types (specified as dicts in run_xai.py):

  "zero_feature"   zero out all tokens matching a feature prefix
                   e.g. {"name": "Remove NTproBNP",
                         "type": "zero_feature",
                         "feature_pattern": "lab::50963"}

  "replace_token"  swap one specific token string for another
                   e.g. {"name": "NTproBNP Q4 → Q1",
                         "type": "replace_token",
                         "from_token": "lab::50963_Q4",
                         "to_token":   "lab::50963_Q1"}

  "remove_visit"   zero out all tokens belonging to a visit (by relative index)
                   e.g. {"name": "Remove last visit",
                         "type": "remove_visit",
                         "visit_offset": -1}   # -1 = last, 0 = first
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from model_src.ehr_encoder import EHR_Encoder
from .utils import LABEL_COLORS, PERTURB_COLORS


# ── Token modification ────────────────────────────────────────────────────────

def apply_perturbation(
    concept_ids_1d: torch.Tensor,
    visit_ids_1d: torch.Tensor,
    inv_vocab: dict[int, str],
    vocab: dict[str, int],
    spec: dict,
) -> torch.Tensor:
    """
    Apply one perturbation to a concept_ids sequence.
    Always preserves position 0 (CLS token).
    Returns a modified copy of concept_ids_1d.
    """
    modified = concept_ids_1d.clone()
    n = len(modified)
    ptype = spec["type"]

    if ptype == "zero_feature":
        pattern = spec["feature_pattern"]
        for i in range(1, n):
            if inv_vocab.get(int(modified[i].item()), "").startswith(pattern):
                modified[i] = 0

    elif ptype == "replace_token":
        from_id = vocab.get(spec["from_token"])
        to_id   = vocab.get(spec["to_token"])
        if from_id is None:
            print(f"    Warning: '{spec['from_token']}' not in vocabulary — skipping")
            return modified
        if to_id is None:
            print(f"    Warning: '{spec['to_token']}' not in vocabulary — skipping")
            return modified
        mask      = (modified == from_id)
        mask[0]   = False   # never touch CLS
        modified[mask] = to_id

    elif ptype == "remove_visit":
        offset        = spec.get("visit_offset", -1)
        vis_vals      = visit_ids_1d[1:n].cpu()
        unique_visits = sorted(vis_vals[vis_vals > 0].unique().tolist())
        if not unique_visits:
            return modified
        target_idx   = len(unique_visits) + offset if offset < 0 else offset
        if 0 <= target_idx < len(unique_visits):
            target_visit = int(unique_visits[target_idx])
            for i in range(1, n):
                if int(visit_ids_1d[i].item()) == target_visit:
                    modified[i] = 0

    else:
        raise ValueError(
            f"Unknown perturbation type '{ptype}'. "
            "Expected 'zero_feature', 'replace_token', or 'remove_visit'."
        )

    return modified


# ── Running all perturbations ─────────────────────────────────────────────────

@torch.no_grad()
def run_perturbation_analysis(
    model: EHR_Encoder,
    tensors: dict[str, torch.Tensor],
    inv_vocab: dict[int, str],
    vocab: dict[str, int],
    patient_indices: list[int],
    perturbation_specs: list[dict],
    device: torch.device,
) -> dict[str, list[dict]]:
    """
    For each patient cycle and each perturbation, compute the modified
    prediction probability and CLS embedding.

    Returns {perturbation_name: [per-cycle dict, ...]} where each dict has:
      idx, prob_orig, prob_perturb, delta_prob, cls_emb
    """
    # Original probabilities
    orig_probs: dict[int, float] = {}
    for idx in patient_indices:
        batch  = {k: v[[idx]].to(device) for k, v in tensors.items()}
        logits = model(batch["concept_ids"], batch["type_ids"], batch["visit_ids"],
                       batch["position_ids"], batch["age_ids"],
                       batch.get("dates"), batch.get("age_years"))
        orig_probs[idx] = float(F.softmax(logits, dim=-1)[0, 1].item())

    results: dict[str, list[dict]] = {s["name"]: [] for s in perturbation_specs}

    for spec in perturbation_specs:
        print(f"  Perturbation: {spec['name']}")
        for idx in patient_indices:
            perturbed_ids = apply_perturbation(
                tensors["concept_ids"][idx].cpu(),
                tensors["visit_ids"][idx].cpu(),
                inv_vocab, vocab, spec,
            )
            batch = {k: v[[idx]].to(device) for k, v in tensors.items()}
            batch["concept_ids"] = perturbed_ids.unsqueeze(0).to(device)

            logits = model(batch["concept_ids"], batch["type_ids"], batch["visit_ids"],
                           batch["position_ids"], batch["age_ids"],
                           batch.get("dates"), batch.get("age_years"))
            prob_p = float(F.softmax(logits, dim=-1)[0, 1].item())

            # CLS embedding under perturbation
            x = model.embedding(batch["concept_ids"], batch["type_ids"],
                                 batch["visit_ids"], batch["position_ids"],
                                 batch["age_ids"], batch.get("dates"),
                                 batch.get("age_years"))
            pad_mask = (batch["concept_ids"] != 0).long()
            for layer in model.layers:
                x = layer(x, pad_mask)
            x = model.norm(x)

            results[spec["name"]].append({
                "idx":          idx,
                "prob_orig":    orig_probs[idx],
                "prob_perturb": prob_p,
                "delta_prob":   prob_p - orig_probs[idx],
                "cls_emb":      x[0, 0, :].cpu(),
            })

    return results


# ── Plot 5a: Perturbed CLS trajectory ────────────────────────────────────────

def plot_perturbation_trajectory(
    original_cycle_data: list[dict],
    perturb_results: dict[str, list[dict]],
    subject_id: int,
    patient_coords_2d: np.ndarray,
    perturb_coords_2d: dict[str, np.ndarray],
    coords_background: np.ndarray | None,
    background_labels: np.ndarray | None,
    output_path: Path,
    method_name: str = "UMAP",
) -> None:
    """
    CLS trajectory under each perturbation (dashed) overlaid on the original
    trajectory (solid) and the test-set background cluster.
    """
    n_cycles = len(original_cycle_data)

    fig, ax = plt.subplots(figsize=(9, 7))

    if coords_background is not None and len(coords_background) > 0:
        bg_c = ([LABEL_COLORS.get(int(l), "#aaaaaa") for l in background_labels]
                if background_labels is not None else "#cccccc")
        ax.scatter(coords_background[:, 0], coords_background[:, 1],
                   c=bg_c, alpha=0.07, s=14, linewidths=0,
                   rasterized=True, zorder=1)

    # Original trajectory
    for i in range(n_cycles - 1):
        x0, y0 = patient_coords_2d[i]
        x1, y1 = patient_coords_2d[i + 1]
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="->", color="#222222", lw=1.8),
                    zorder=4)
    ax.plot(patient_coords_2d[:, 0], patient_coords_2d[:, 1],
            color="#222222", lw=1.5, zorder=3)
    ax.scatter(patient_coords_2d[:, 0], patient_coords_2d[:, 1],
               c="#222222", s=80, zorder=5, label="Original")
    for i, cd in enumerate(original_cycle_data):
        ax.annotate(f" C{cd['cycle_number']} P={cd['prob']:.2f}",
                    xy=patient_coords_2d[i], fontsize=7.5, color="#000000", zorder=6)

    # Perturbed trajectories
    for pi, (name, coords_p) in enumerate(perturb_coords_2d.items()):
        color = PERTURB_COLORS[pi % len(PERTURB_COLORS)]
        ax.plot(coords_p[:, 0], coords_p[:, 1],
                color=color, lw=1.4, ls="--", alpha=0.85,
                label=f"[perturb] {name}", zorder=3)
        ax.scatter(coords_p[:, 0], coords_p[:, 1],
                   c=color, s=55, alpha=0.85, zorder=4)
        for i, res in enumerate(perturb_results[name]):
            ax.annotate(f" P={res['prob_perturb']:.2f}",
                        xy=coords_p[i], fontsize=6.5, color=color, alpha=0.9, zorder=5)

    ax.set_xlabel(f"{method_name} dim 1", fontsize=10)
    ax.set_ylabel(f"{method_name} dim 2", fontsize=10)
    ax.set_title(f"Perturbation CLS Trajectory — Patient {subject_id}", fontsize=12)
    ax.legend(fontsize=8, loc="upper left")
    ax.text(0.99, 0.01, method_name, transform=ax.transAxes,
            fontsize=8, color="gray", ha="right", va="bottom")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {output_path.name}")


# ── Plot 5b: ΔP(cardiotoxic) per perturbation ────────────────────────────────

def plot_perturbation_delta(
    original_cycle_data: list[dict],
    perturb_results: dict[str, list[dict]],
    subject_id: int,
    output_path: Path,
) -> None:
    """
    Two-panel figure:
      Left  — grouped ΔP bars (one group per cycle, one bar per perturbation)
      Right — absolute P(cardiotoxic) per cycle for each perturbation
    """
    cycle_nums = [cd["cycle_number"] for cd in original_cycle_data]
    n_cycles   = len(cycle_nums)
    perturb_names = list(perturb_results.keys())
    n_perturbs    = len(perturb_names)

    x     = np.arange(n_cycles)
    width = 0.7 / max(n_perturbs, 1)

    fig, (ax, ax2) = plt.subplots(
        1, 2, figsize=(10 + n_perturbs, 5),
        gridspec_kw={"width_ratios": [2, 1]},
    )

    # Left: ΔP bars
    for pi, name in enumerate(perturb_names):
        deltas = [r["delta_prob"] for r in perturb_results[name]]
        color  = PERTURB_COLORS[pi % len(PERTURB_COLORS)]
        offset = (pi - n_perturbs / 2 + 0.5) * width
        ax.bar(x + offset, deltas, width=width * 0.9,
               color=color, label=name, alpha=0.85, edgecolor="white")

    ax.axhline(0, color="black", lw=0.8, alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([f"Cycle {c}" for c in cycle_nums], fontsize=9)
    ax.set_ylabel("ΔP(cardiotoxic) = P_perturbed − P_original", fontsize=9)
    ax.set_title(f"Perturbation Effect — Patient {subject_id}", fontsize=11)
    ax.legend(fontsize=8, loc="upper right")

    # Right: absolute probabilities
    orig_probs = [cd["prob"] for cd in original_cycle_data]
    ax2.plot(range(n_cycles), orig_probs, "ko-", lw=1.8, label="Original", zorder=4)
    for pi, name in enumerate(perturb_names):
        perturb_probs = [r["prob_perturb"] for r in perturb_results[name]]
        color = PERTURB_COLORS[pi % len(PERTURB_COLORS)]
        ax2.plot(range(n_cycles), perturb_probs, "s--",
                 color=color, lw=1.2, alpha=0.85, label=name, zorder=3)
    ax2.axhline(0.5, color="gray", lw=0.7, ls=":", alpha=0.7)
    ax2.set_xticks(range(n_cycles))
    ax2.set_xticklabels([f"C{c}" for c in cycle_nums], fontsize=8)
    ax2.set_ylabel("P(cardiotoxic)", fontsize=9)
    ax2.set_ylim(0, 1)
    ax2.set_title("Prediction Probability", fontsize=10)
    ax2.legend(fontsize=7, loc="upper left")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {output_path.name}")
