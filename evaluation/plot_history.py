"""
plot_history.py

Plot training history (loss curves + validation metrics) from one or more runs.

Usage:
    # Single run — display interactively (default: loss + auroc + auprc + f1)
    python evaluation/plot_history.py --model-dir experiment_outputs/test1

    # Choose which validation metrics to plot
    python evaluation/plot_history.py --model-dir experiment_outputs/test1 \\
        --metrics auroc auprc f1 sensitivity specificity

    # Save to file instead of displaying
    python evaluation/plot_history.py --model-dir experiment_outputs/test1 \\
        --save experiment_outputs/test1/training_curves.png

    # Compare multiple runs on the same plot
    python evaluation/plot_history.py \\
        --model-dir experiment_outputs/run1 experiment_outputs/run2 \\
        --save comparison.png
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

METRIC_LABELS = {
    "auroc":       "Validation AUROC",
    "auprc":       "Validation AUPRC",
    "f1":          "Validation F1",
    "sensitivity": "Validation Sensitivity",
    "specificity": "Validation Specificity",
}

ALL_METRICS = list(METRIC_LABELS.keys())


def _load_history(model_dir: Path) -> list[dict]:
    history_path = model_dir / "history.json"
    if not history_path.exists():
        print(f"No history.json found in {model_dir}")
        sys.exit(1)
    with open(history_path) as f:
        return json.load(f)


def _run_label(model_dir: Path, cfg: dict | None) -> str:
    """Short label for legend entries — just the run name."""
    return model_dir.name


def _arch_subtitle(cfg: dict | None) -> str:
    """One-line architecture string for use in panel subtitles."""
    if cfg:
        return f"d={cfg.get('d_model','?')} L={cfg.get('num_layers','?')}"
    return ""


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

    n_runs    = len(model_dirs)
    n_metrics = len(metrics)
    n_cols    = 1 + n_metrics          # loss panel + one per metric
    n_rows    = 1

    if figsize is None:
        figsize = (max(20, n_cols * 6.0), max(6, n_runs * 1.2 + 4))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_cols == 1:
        axes = [axes]
    axes = list(axes)

    ax_loss      = axes[0]
    metric_axes  = axes[1:]

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for i, model_dir in enumerate(model_dirs):
        history = _load_history(model_dir)
        cfg      = None
        cfg_path = model_dir / "config.json"
        if cfg_path.exists():
            with open(cfg_path) as f:
                cfg = json.load(f)

        label = _run_label(model_dir, cfg)
        color = colors[i % len(colors)]
        epochs = [h["epoch"] for h in history]

        train_loss = [h["train_loss"] for h in history]
        val_loss   = [h["loss"]       for h in history]

        ax_loss.plot(epochs, train_loss, color=color, linestyle="--", alpha=0.7,
                     label=f"{label} — train")
        ax_loss.plot(epochs, val_loss,   color=color, linestyle="-",
                     label=f"{label} — val")

        for ax, metric in zip(metric_axes, metrics):
            if metric not in history[0]:
                ax.text(0.5, 0.5, f"{metric}\nnot in history",
                        ha="center", va="center", transform=ax.transAxes, color="gray")
                continue

            vals       = [h[metric] for h in history]
            best_epoch = max(history, key=lambda h: h.get(metric, float("-inf")))["epoch"]
            best_val   = max(h.get(metric, float("-inf")) for h in history)

            ax.plot(epochs, vals, color=color, linestyle="-",
                    label=f"{label}  ({best_val:.3f} @ ep{best_epoch})")
            ax.axvline(best_epoch, color=color, linestyle=":", alpha=0.4, linewidth=0.8)

    _legend_kw = dict(fontsize=8, loc="upper left",
                      bbox_to_anchor=(1.02, 1), borderaxespad=0,
                      framealpha=0.9, edgecolor="0.8")

    # ── loss panel ────────────────────────────────────────────────────────────
    ax_loss.set_title("Loss", fontsize=13, fontweight="bold")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Cross-entropy loss")
    ax_loss.xaxis.set_major_locator(ticker.MaxNLocator(nbins=6, integer=True))
    ax_loss.grid(alpha=0.3)
    ax_loss.legend(**_legend_kw)

    # ── metric panels ─────────────────────────────────────────────────────────
    for ax, metric in zip(metric_axes, metrics):
        ax.set_title(METRIC_LABELS.get(metric, metric), fontsize=13, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric.upper())
        ax.set_ylim(0, 1)
        ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, label="random (0.5)")
        ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=6, integer=True))
        ax.grid(alpha=0.3)
        ax.legend(**_legend_kw)

    fig.tight_layout(rect=[0, 0, 0.88, 1])

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"Saved to: {save_path}")
    else:
        plt.show()


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Plot training loss and validation metrics for one or more runs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model-dir", nargs="+", required=True,
                   help="One or more experiment output directories.")
    p.add_argument("--metrics", nargs="+", default=["auroc", "auprc", "f1"],
                   choices=ALL_METRICS,
                   help="Validation metrics to plot (one panel each).")
    p.add_argument("--save",  default=None,
                   help="Path to save the figure (PNG/PDF/SVG). "
                        "If omitted, the plot is displayed interactively.")
    p.add_argument("--dpi",   type=int, default=150,
                   help="Output DPI when saving.")
    p.add_argument(
        "--figsize", nargs=2, type=float, metavar=("W", "H"), default=None,
        help="Figure width and height in inches (e.g. --figsize 20 8). "
             "Auto-scales by number of panels if omitted.",
    )
    return p.parse_args()


def main() -> None:
    args       = parse_args()
    model_dirs = [Path(d) for d in args.model_dir]
    save_path  = Path(args.save) if args.save else None
    figsize    = tuple(args.figsize) if args.figsize else None
    plot(model_dirs, args.metrics, save_path, args.dpi, figsize=figsize)


if __name__ == "__main__":
    main()
