"""
population.py

Dataset-level xAI visualizations:
  - CLS embedding space (UMAP/PCA coloured by label, probability, cycle)
  - Population-level aggregate Integrated Gradients
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

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from interpretation.interpret import (
    _decode_tokens, _event_type, _human_label,
    compute_integrated_gradients,
)
from .utils import (EVENT_TYPE_COLORS, LABEL_COLORS, LABEL_NAMES,
                    DRUG_CLASS_COLORS, PATIENT_OVERLAY_COLORS)
from .embeddings import reduce_dim


# ── Plot 1: CLS embedding space ───────────────────────────────────────────────

def plot_cls_embedding_space(
    embeddings: torch.Tensor,
    labels: np.ndarray,
    probs: np.ndarray,
    samples_df: pd.DataFrame,
    indices: list[int],
    output_dir: Path,
    method: str = "umap",
    split_filter: str = "test",
    random_state: int = 42,
    patient_overlay: dict[int, torch.Tensor] | None = None,
) -> np.ndarray:
    """
    Produce CLS embedding space scatter plots coloured by various attributes.

    patient_overlay: optional dict of {subject_id: (n_cycles, d_model) tensor}.
    When provided, all patient embeddings are projected jointly with the
    background so they share the same coordinate space, then overlaid as
    stars on every subplot.

    Returns the (N_background, 2) projection coordinates.
    """
    emb_np = embeddings.float().numpy()
    n_bg   = len(emb_np)

    # ── Joint projection (background + patients in one call) ──────────────────
    if patient_overlay:
        pat_emb_list = [v.float().numpy() for v in patient_overlay.values()]
        pat_sizes    = [len(e) for e in pat_emb_list]
        combined     = np.vstack([emb_np] + pat_emb_list)
        print(f"\nReducing {n_bg} background + {sum(pat_sizes)} patient embeddings "
              f"jointly with {method.upper()}...")
        coords_all, method_name = reduce_dim(combined, method=method,
                                             random_state=random_state)
        coords = coords_all[:n_bg]
        patient_coords: dict[int, np.ndarray] = {}
        ptr = n_bg
        for sid, size in zip(patient_overlay.keys(), pat_sizes):
            patient_coords[sid] = coords_all[ptr : ptr + size]
            ptr += size
    else:
        print(f"\nReducing {n_bg} embeddings with {method.upper()}...")
        coords, method_name = reduce_dim(emb_np, method=method,
                                         random_state=random_state)
        patient_coords = {}

    print(f"  {method_name} complete. Shape: {coords.shape}")
    output_dir.mkdir(parents=True, exist_ok=True)

    subset_df  = samples_df.iloc[indices].reset_index(drop=True)
    cycle_nums = (subset_df["cycle_number"].values
                  if "cycle_number" in subset_df.columns
                  else np.ones(len(indices), dtype=int))

    xlabel = f"{method_name} dim 1"
    ylabel = f"{method_name} dim 2"

    # Pre-compute cycle numbers for each patient overlay
    pat_cycle_nums: dict[int, list[int]] = {}
    for sid in patient_coords:
        pat_rows = (samples_df[samples_df["subject_id"] == sid]
                    .sort_values("cycle_number"))
        pat_cycle_nums[sid] = pat_rows["cycle_number"].tolist()

    def _method_tag(ax: plt.Axes) -> None:
        ax.text(0.99, 0.01, method_name, transform=ax.transAxes,
                fontsize=8, color="gray", ha="right", va="bottom")

    def _overlay_patients(ax: plt.Axes) -> list:
        """Draw star markers for each patient and return legend handles."""
        handles = []
        for pi, (sid, pcoords) in enumerate(patient_coords.items()):
            color  = PATIENT_OVERLAY_COLORS[pi % len(PATIENT_OVERLAY_COLORS)]
            cycles = pat_cycle_nums.get(sid, list(range(1, len(pcoords) + 1)))
            for j, (px, py) in enumerate(pcoords):
                cyc = cycles[j] if j < len(cycles) else j + 1
                ax.scatter(px, py, c=color, s=300, marker="*", zorder=6,
                           edgecolors="white", linewidths=0.5)
                ax.annotate(f" C{cyc}", xy=(px, py), fontsize=7.5, color=color,
                            fontweight="bold", zorder=7, ha="left", va="bottom")
            handles.append(mpatches.Patch(color=color, label=f"Patient {sid}"))
        return handles

    # ── by true label ─────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 7))
    for lbl in [0, 1]:
        mask = labels == lbl
        if not mask.any():
            continue
        ax.scatter(coords[mask, 0], coords[mask, 1], c=LABEL_COLORS[lbl],
                   label=f"{LABEL_NAMES[lbl]} (n={mask.sum()})",
                   alpha=0.55, s=20, linewidths=0, rasterized=True)
    pat_handles = _overlay_patients(ax)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(f"CLS Embedding Space — {split_filter} split (true label)", fontsize=12)
    ax.legend(handles=ax.get_legend_handles_labels()[0] + pat_handles,
              labels=ax.get_legend_handles_labels()[1] + [h.get_label() for h in pat_handles],
              fontsize=9, markerscale=1.8)
    _method_tag(ax)
    fig.tight_layout()
    fig.savefig(output_dir / "cls_space_by_label.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved cls_space_by_label.png")

    # ── by prediction probability ──────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 7))
    sc = ax.scatter(coords[:, 0], coords[:, 1], c=probs, cmap="RdBu_r",
                    vmin=0, vmax=1, alpha=0.6, s=20, linewidths=0, rasterized=True)
    plt.colorbar(sc, ax=ax, label="P(cardiotoxic)", fraction=0.035, pad=0.02)
    pat_handles = _overlay_patients(ax)
    if pat_handles:
        ax.legend(handles=pat_handles, fontsize=9, loc="upper left")
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(f"CLS Embedding Space — {split_filter} split (P(cardiotoxic))",
                 fontsize=12)
    _method_tag(ax)
    fig.tight_layout()
    fig.savefig(output_dir / "cls_space_by_prob.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved cls_space_by_prob.png")

    # ── by cycle number ────────────────────────────────────────────────────────
    unique_cycles = np.unique(cycle_nums)
    palette = plt.cm.plasma(np.linspace(0.1, 0.9, max(len(unique_cycles), 2)))

    fig, ax = plt.subplots(figsize=(9, 7))
    for ci, cyc in enumerate(unique_cycles):
        mask = cycle_nums == cyc
        ax.scatter(coords[mask, 0], coords[mask, 1], c=[palette[ci]],
                   label=f"Cycle {int(cyc)} (n={mask.sum()})",
                   alpha=0.55, s=20, linewidths=0, rasterized=True)
    pat_handles = _overlay_patients(ax)
    handles, labels_list = ax.get_legend_handles_labels()
    ax.legend(handles=handles + pat_handles,
              labels=labels_list + [h.get_label() for h in pat_handles],
              fontsize=8, markerscale=1.8, ncol=min(3, len(unique_cycles)))
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(f"CLS Embedding Space — {split_filter} split (cycle number)", fontsize=12)
    _method_tag(ax)
    fig.tight_layout()
    fig.savefig(output_dir / "cls_space_by_cycle.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved cls_space_by_cycle.png")

    # ── by drug class (per cycle) ─────────────────────────────────────────────
    if "drug_classes_in_cycle" in subset_df.columns:
        drug_classes   = subset_df["drug_classes_in_cycle"].fillna("unknown").values
        unique_classes = sorted(set(drug_classes))
        fig, ax = plt.subplots(figsize=(9, 7))
        for dc in unique_classes:
            mask  = drug_classes == dc
            color = DRUG_CLASS_COLORS.get(dc, "#aaaaaa")
            ax.scatter(coords[mask, 0], coords[mask, 1], c=color,
                       label=dc.replace("_", " ") + f" (n={mask.sum()})",
                       alpha=0.6, s=20, linewidths=0, rasterized=True)
        pat_handles = _overlay_patients(ax)
        handles, labels_list = ax.get_legend_handles_labels()
        ax.legend(handles=handles + pat_handles,
                  labels=labels_list + [h.get_label() for h in pat_handles],
                  fontsize=8, markerscale=1.8,
                  bbox_to_anchor=(1.01, 1), loc="upper left",
                  borderaxespad=0, frameon=True)
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(f"CLS Embedding Space — {split_filter} split (drug class per cycle)",
                     fontsize=12)
        _method_tag(ax)
        fig.tight_layout()
        fig.savefig(output_dir / "cls_space_by_drug_class.png", dpi=150,
                    bbox_inches="tight")
        plt.close(fig)
        print("  Saved cls_space_by_drug_class.png")

    # ── by primary regimen ────────────────────────────────────────────────────
    if "primary_drug_class" in subset_df.columns:
        primary        = subset_df["primary_drug_class"].fillna("unknown").values
        unique_primary = sorted(set(primary))
        fig, ax = plt.subplots(figsize=(9, 7))
        for dc in unique_primary:
            mask  = primary == dc
            color = DRUG_CLASS_COLORS.get(dc, "#aaaaaa")
            ax.scatter(coords[mask, 0], coords[mask, 1], c=color,
                       label=dc.replace("_", " ") + f" (n={mask.sum()})",
                       alpha=0.6, s=20, linewidths=0, rasterized=True)
        pat_handles = _overlay_patients(ax)
        handles, labels_list = ax.get_legend_handles_labels()
        ax.legend(handles=handles + pat_handles,
                  labels=labels_list + [h.get_label() for h in pat_handles],
                  fontsize=8, markerscale=1.8,
                  bbox_to_anchor=(1.01, 1), loc="upper left",
                  borderaxespad=0, frameon=True)
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(f"CLS Embedding Space — {split_filter} split (primary regimen)",
                     fontsize=12)
        _method_tag(ax)
        fig.tight_layout()
        fig.savefig(output_dir / "cls_space_by_primary_regimen.png", dpi=150,
                    bbox_inches="tight")
        plt.close(fig)
        print("  Saved cls_space_by_primary_regimen.png")

    # ── by prescription count (dose proxy, continuous) ────────────────────────
    if "n_prescription_rows_in_cycle" in subset_df.columns:
        rx_counts = subset_df["n_prescription_rows_in_cycle"].fillna(0).values.astype(float)
        fig, ax = plt.subplots(figsize=(9, 7))
        sc = ax.scatter(coords[:, 0], coords[:, 1], c=rx_counts, cmap="viridis",
                        alpha=0.6, s=20, linewidths=0, rasterized=True)
        plt.colorbar(sc, ax=ax, label="Prescription rows in cycle (dose proxy)",
                     fraction=0.035, pad=0.02)
        pat_handles = _overlay_patients(ax)
        if pat_handles:
            ax.legend(handles=pat_handles, fontsize=9, loc="upper left")
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(f"CLS Embedding Space — {split_filter} split (cumulative dose proxy)",
                     fontsize=12)
        _method_tag(ax)
        fig.tight_layout()
        fig.savefig(output_dir / "cls_space_by_rx_count.png", dpi=150,
                    bbox_inches="tight")
        plt.close(fig)
        print("  Saved cls_space_by_rx_count.png")

    np.save(output_dir / "cls_coords_2d.npy", coords)
    print("  Saved cls_coords_2d.npy")
    return coords


# ── Plot 2: Population aggregate IG ───────────────────────────────────────────

def compute_population_ig(
    model,
    tensors: dict[str, torch.Tensor],
    inv_vocab: dict[int, str],
    indices: list[int],
    device: torch.device,
    ig_steps: int = 30,
    max_samples: int = 50,
    cache_path: Path | None = None,
) -> pd.DataFrame:
    """
    Compute and aggregate signed IG attribution across a population of samples.

    If cache_path exists the cached CSV is returned without re-computing.
    Otherwise computes IG on up to max_samples randomly drawn from indices,
    saves the cache, and returns the aggregated DataFrame.

    Columns: token, label, event_type, mean_ig_signed, std_ig_signed, sem_ig_signed,
             n, abs_mean_ig
    """
    if cache_path is not None and cache_path.exists():
        print(f"  Loading cached population IG from {cache_path}")
        return pd.read_csv(cache_path)

    rng = np.random.default_rng(42)
    sampled = (rng.choice(indices, size=max_samples, replace=False).tolist()
               if len(indices) > max_samples else list(indices))
    print(f"  Computing IG on {len(sampled)}/{len(indices)} samples "
          f"({ig_steps} steps each)...")

    rows: list[dict] = []
    for si, idx in enumerate(sampled):
        print(f"    sample {si + 1}/{len(sampled)}\r", end="", flush=True)
        batch = {k: v[[idx]].to(device) for k, v in tensors.items()}

        concept_ids_1d = batch["concept_ids"][0].cpu()
        n_active = int((concept_ids_1d != 0).sum().item())
        if n_active < 3:
            continue

        raw_tokens, _ = _decode_tokens(concept_ids_1d[:n_active], inv_vocab)

        try:
            _, ig_signed, _ = compute_integrated_gradients(
                model, batch, target_class=1, n_steps=ig_steps
            )
            ig_signed_content = ig_signed[1:n_active].cpu()
        except Exception as e:
            print(f"\n    IG failed for idx={idx}: {e}")
            continue

        for raw, sig in zip(raw_tokens[1:], ig_signed_content.tolist()):
            rows.append({
                "token":      raw,
                "label":      _human_label(raw),
                "event_type": _event_type(raw),
                "ig_signed":  sig,
            })

    print()
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    agg = (
        df.groupby(["token", "label", "event_type"])["ig_signed"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"mean": "mean_ig_signed",
                          "std":  "std_ig_signed",
                          "count": "n"})
    )
    agg["sem_ig_signed"] = agg["std_ig_signed"] / np.sqrt(agg["n"])
    agg["abs_mean_ig"]   = agg["mean_ig_signed"].abs()
    agg = agg.sort_values("abs_mean_ig", ascending=False).reset_index(drop=True)

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        agg.to_csv(cache_path, index=False)
        print(f"  Cached → {cache_path}")

    return agg


def plot_population_ig(
    ig_df: pd.DataFrame,
    output_path: Path,
    top_k: int = 25,
    split_filter: str = "test",
    n_samples: int | None = None,
) -> None:
    """
    Horizontal bar chart of the top-K features by |mean signed IG| with ±1 SEM
    error bars, coloured by direction (red = pro-toxic, blue = protective).
    """
    if ig_df.empty or "mean_ig_signed" not in ig_df.columns:
        print("  Skipping population IG plot: no data")
        return

    df = (ig_df.nlargest(top_k, "abs_mean_ig")
                .sort_values("mean_ig_signed", ascending=True)
                .reset_index(drop=True))

    colors = ["#c0392b" if v >= 0 else "#2980b9" for v in df["mean_ig_signed"]]
    errs   = df["sem_ig_signed"].fillna(0).tolist()

    fig, ax = plt.subplots(figsize=(8, max(5, top_k * 0.28)))
    ax.barh(range(len(df)), df["mean_ig_signed"],
            xerr=errs, color=colors, edgecolor="white", linewidth=0.4,
            capsize=3, error_kw={"elinewidth": 0.7, "ecolor": "#555555"})
    ax.axvline(0, color="black", linewidth=0.7, alpha=0.5)

    for i, (_, row) in enumerate(df.iterrows()):
        xv  = float(row["mean_ig_signed"])
        xe  = float(errs[i])
        tip = xv + xe if xv >= 0 else xv - xe
        ha  = "left" if xv >= 0 else "right"
        ax.text(tip * 1.02 if xv >= 0 else tip * 0.98,
                i, f"n={int(row['n'])}", ha=ha, va="center",
                fontsize=5.5, color="#444444")

    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df["label"], fontsize=7)
    ax.set_xlabel("Mean signed IG attribution  (±1 SEM)", fontsize=9)
    n_label = f"  ({n_samples} samples)" if n_samples else ""
    ax.set_title(
        f"Population Feature Importance — {split_filter}{n_label}\n"
        f"Top {top_k} features by |mean signed IG|",
        fontsize=11,
    )
    ax.legend(handles=[
        mpatches.Patch(color="#c0392b", label="pro-toxic  (+)"),
        mpatches.Patch(color="#2980b9", label="protective (−)"),
    ], fontsize=8, loc="lower right")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {output_path.name}")
