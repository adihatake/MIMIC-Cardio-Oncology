"""
plot_history.py

Plot training history (loss + validation metrics) from one or more runs.

When a directory contains seed subdirectories (each with history.json), curves
are aggregated: the solid line is the mean across seeds and the shaded band is
±1 std.  Pass a single seed directory for no aggregation.

Usage:
    # One variant — aggregates seed* subdirs automatically
    python evaluation/plot_history.py \\
        --model-dir experiment_outputs/July23/arch_sweep/M

    # Compare variants side-by-side
    python evaluation/plot_history.py \\
        --model-dir experiment_outputs/July23/arch_sweep/S \\
                    experiment_outputs/July23/arch_sweep/M \\
                    experiment_outputs/July23/arch_sweep/L

    # Single seed directory (no aggregation, no band)
    python evaluation/plot_history.py \\
        --model-dir experiment_outputs/July23/arch_sweep/M/seed42

    # Choose metrics and save
    python evaluation/plot_history.py \\
        --model-dir experiment_outputs/July23/arch_sweep/M \\
        --metrics auroc auprc f1 --save arch_M.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

METRIC_LABELS = {
    "auroc":       "Validation AUROC",
    "auprc":       "Validation AUPRC",
    "f1":          "Validation F1",
    "sensitivity": "Validation Sensitivity",
    "specificity": "Validation Specificity",
}

ALL_METRICS = list(METRIC_LABELS.keys())


# ── data loading ──────────────────────────────────────────────────────────────

def _load_history(path: Path) -> list[dict]:
    if not path.exists():
        print(f"history.json not found: {path}")
        sys.exit(1)
    with open(path) as f:
        return json.load(f)


def _load_variant(model_dir: Path) -> tuple[list[list[dict]], str]:
    """Return (list_of_histories, label).

    - If model_dir has history.json → single run, no aggregation.
    - Otherwise discover all immediate subdirs that have history.json (seeds).
    """
    direct = model_dir / "history.json"
    if direct.exists():
        return [_load_history(direct)], model_dir.name

    seed_dirs = sorted(
        d for d in model_dir.iterdir()
        if d.is_dir() and (d / "history.json").exists()
    )
    if not seed_dirs:
        print(f"No history.json found in {model_dir} or its subdirectories.")
        sys.exit(1)

    return [_load_history(d / "history.json") for d in seed_dirs], model_dir.name


# ── aggregation ───────────────────────────────────────────────────────────────

def _agg(
    histories: list[list[dict]], key: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean, mean−std, mean+std across seeds for a given metric key."""
    min_len = min(len(h) for h in histories)
    arr     = np.array([[row[key] for row in hist[:min_len]] for hist in histories])
    mean    = arr.mean(axis=0)
    std     = arr.std(axis=0)
    return mean, mean - std, mean + std


# ── plotting ──────────────────────────────────────────────────────────────────

def plot(
    model_dirs: list[Path],
    metrics:    list[str],
    save_path:  Path | None,
    dpi:        int,
    figsize:    tuple[float, float] | None = None,
) -> None:
    try:
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker
    except ImportError:
        print("matplotlib is required.  pip install matplotlib")
        sys.exit(1)

    n_variants = len(model_dirs)
    n_cols     = 1 + len(metrics)

    if figsize is None:
        figsize = (max(20, n_cols * 6.0), max(6, n_variants * 0.8 + 5))

    fig, axes = plt.subplots(1, n_cols, figsize=figsize)
    if n_cols == 1:
        axes = [axes]
    axes = list(axes)

    ax_loss     = axes[0]
    metric_axes = axes[1:]

    colors       = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    labels_seen  : list[str]                                     = []
    any_multirun : bool                                          = False
    best_ann     : dict[str, list[tuple[str, float, float, int]]] = {m: [] for m in metrics}

    for i, model_dir in enumerate(model_dirs):
        histories, label = _load_variant(model_dir)
        labels_seen.append(label)
        color        = colors[i % len(colors)]
        n_seeds      = len(histories)
        any_multirun = any_multirun or n_seeds > 1
        min_len = min(len(h) for h in histories)
        epochs  = [row["epoch"] for row in histories[0][:min_len]]

        # ── loss panel ───────────────────────────────────────────────────────
        tr_mean, tr_lo, tr_hi = _agg(histories, "train_loss")
        vl_mean, vl_lo, vl_hi = _agg(histories, "loss")

        ax_loss.plot(epochs, tr_mean, color=color, ls="--", alpha=0.6, lw=1.2)
        ax_loss.plot(epochs, vl_mean, color=color, ls="-",  lw=1.5, label=label)
        if n_seeds > 1:
            ax_loss.fill_between(epochs, tr_lo, tr_hi, color=color, alpha=0.10)
            ax_loss.fill_between(epochs, vl_lo, vl_hi, color=color, alpha=0.18)

        # ── metric panels ────────────────────────────────────────────────────
        for ax, metric in zip(metric_axes, metrics):
            if metric not in histories[0][0]:
                ax.text(0.5, 0.5, f"{metric}\nnot in history",
                        ha="center", va="center", transform=ax.transAxes, color="gray")
                continue

            m_mean, m_lo, m_hi = _agg(histories, metric)
            ax.plot(epochs, m_mean, color=color, ls="-", lw=1.5)
            if n_seeds > 1:
                ax.fill_between(epochs, m_lo, m_hi, color=color, alpha=0.2)

            best_idx   = int(np.argmax(m_mean))
            best_val   = float(m_mean[best_idx])
            best_std   = float(np.std([max(row[metric] for row in hist) for hist in histories]))
            best_epoch = epochs[best_idx]
            ax.axvline(best_epoch, color=color, ls=":", alpha=0.4, lw=0.8)
            best_ann[metric].append((label, best_val, best_std, best_epoch))

    # ── shared legend ─────────────────────────────────────────────────────────
    from matplotlib.lines import Line2D

    variant_handles = [
        Line2D([0], [0], color=colors[i % len(colors)], lw=1.5, label=labels_seen[i])
        for i in range(n_variants)
    ]
    style_handles = [
        Line2D([0], [0], color="k", lw=1.5, ls="-",  label="val (mean)"),
        Line2D([0], [0], color="k", lw=1.2, ls="--", alpha=0.6, label="train (mean)"),
    ]
    if any_multirun:
        from matplotlib.patches import Patch
        style_handles.append(Patch(color="gray", alpha=0.3, label="±1 std"))

    fig.legend(handles=variant_handles + style_handles,
               loc="center left", fontsize=8,
               bbox_to_anchor=(0.88, 0.5), borderaxespad=0.5,
               framealpha=0.95, edgecolor="0.8")

    # ── loss panel ────────────────────────────────────────────────────────────
    ax_loss.set_title("Loss", fontsize=13, fontweight="bold")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Cross-entropy loss")
    ax_loss.xaxis.set_major_locator(ticker.MaxNLocator(nbins=6, integer=True))
    ax_loss.grid(alpha=0.3)

    # ── metric panels ─────────────────────────────────────────────────────────
    _ann_kw = dict(fontsize=7, family="monospace", va="top", ha="right",
                   bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.85, ec="0.8"))

    for ax, metric in zip(metric_axes, metrics):
        ax.set_title(METRIC_LABELS.get(metric, metric), fontsize=13, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric.upper())
        ax.set_ylim(0, 1)
        ax.axhline(0.5, color="gray", ls="--", lw=0.8)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=6, integer=True))
        ax.grid(alpha=0.3)
        if best_ann[metric]:
            lines = [
                f"{lbl}: {bv:.3f}±{bs:.3f} @ ep{be}"
                for lbl, bv, bs, be in best_ann[metric]
            ]
            ax.text(0.97, 0.97, "\n".join(lines), transform=ax.transAxes, **_ann_kw)

    fig.tight_layout(rect=[0, 0, 0.87, 1])

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"Saved to: {save_path}")
    else:
        plt.show()


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Plot training loss and validation metrics. "
                    "Pass a variant directory to aggregate across seed subdirectories.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model-dir", nargs="+", required=True,
                   help="Variant directory (parent of seed* dirs) or a single seed dir.")
    p.add_argument("--metrics", nargs="+", default=["auroc", "auprc", "f1"],
                   choices=ALL_METRICS,
                   help="Validation metrics to plot (one panel each).")
    p.add_argument("--save",  default=None,
                   help="Save path (PNG/PDF/SVG). Displays interactively if omitted.")
    p.add_argument("--dpi",   type=int, default=150)
    p.add_argument("--figsize", nargs=2, type=float, metavar=("W", "H"), default=None,
                   help="Figure size in inches (auto-scaled if omitted).")
    return p.parse_args()


def main() -> None:
    args      = parse_args()
    model_dirs = [Path(d) for d in args.model_dir]
    save_path  = Path(args.save) if args.save else None
    figsize    = tuple(args.figsize) if args.figsize else None
    plot(model_dirs, args.metrics, save_path, args.dpi, figsize=figsize)


if __name__ == "__main__":
    main()
