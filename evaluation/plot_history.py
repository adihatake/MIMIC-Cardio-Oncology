"""
plot_history.py

Plot training history (loss curves + val AUROC) from one or more experiment runs.

Usage:
    # Single run — display interactively
    python evaluation/plot_history.py --model-dir experiment_outputs/test1

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
import sys
from pathlib import Path


def _load_history(model_dir: Path) -> list[dict]:
    history_path = model_dir / "history.json"
    if not history_path.exists():
        print(f"No history.json found in {model_dir}")
        sys.exit(1)
    with open(history_path) as f:
        return json.load(f)


def _run_label(model_dir: Path, cfg: dict | None) -> str:
    label = model_dir.name
    if cfg:
        label += f"  (d={cfg.get('d_model','?')} L={cfg.get('num_layers','?')})"
    return label


def plot(
    model_dirs: list[Path],
    save_path: Path | None,
    dpi: int,
    figsize: tuple[float, float] | None = None,
) -> None:
    try:
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker
    except ImportError:
        print("matplotlib is required.  pip install matplotlib")
        sys.exit(1)

    n = len(model_dirs)
    if figsize is None:
        figsize = (max(16, n * 3), 7)

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    ax_loss, ax_auroc = axes

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for i, model_dir in enumerate(model_dirs):
        history = _load_history(model_dir)
        cfg      = None
        cfg_path = model_dir / "config.json"
        if cfg_path.exists():
            with open(cfg_path) as f:
                cfg = json.load(f)

        label  = _run_label(model_dir, cfg)
        color  = colors[i % len(colors)]
        epochs = [h["epoch"] for h in history]

        train_loss = [h["train_loss"] for h in history]
        val_loss   = [h["loss"]       for h in history]
        val_auroc  = [h["auroc"]      for h in history]

        best_epoch = max(history, key=lambda h: h["auroc"])["epoch"]
        best_auroc = max(h["auroc"] for h in history)

        ax_loss.plot(epochs, train_loss, color=color, linestyle="--", alpha=0.7,
                     label=f"{label} — train")
        ax_loss.plot(epochs, val_loss,   color=color, linestyle="-",
                     label=f"{label} — val")

        ax_auroc.plot(epochs, val_auroc, color=color, linestyle="-",
                      label=f"{label}  (best={best_auroc:.4f} @ ep{best_epoch})")
        ax_auroc.axvline(best_epoch, color=color, linestyle=":", alpha=0.5)

    # ── loss axes ─────────────────────────────────────────────────────────────
    ax_loss.set_title("Loss", fontsize=13, fontweight="bold")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Cross-entropy loss")
    ax_loss.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax_loss.grid(alpha=0.3)
    # Legend below the subplot; ncol spreads entries horizontally
    ax_loss.legend(
        fontsize=8, loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=max(1, min(4, n)),
        borderaxespad=0, frameon=True,
    )

    # ── AUROC axes ────────────────────────────────────────────────────────────
    ax_auroc.set_title("Validation AUROC", fontsize=13, fontweight="bold")
    ax_auroc.set_xlabel("Epoch")
    ax_auroc.set_ylabel("AUROC")
    ax_auroc.set_ylim(0, 1)
    ax_auroc.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, label="random (0.5)")
    ax_auroc.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax_auroc.grid(alpha=0.3)
    ax_auroc.legend(
        fontsize=8, loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=max(1, min(3, n + 1)),
        borderaxespad=0, frameon=True,
    )

    # Leave room below each subplot for the legends
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.28)

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"Saved to: {save_path}")
    else:
        plt.show()


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Plot training loss and AUROC curves for one or more runs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model-dir", nargs="+", required=True,
                   help="One or more experiment output directories.")
    p.add_argument("--save",  default=None,
                   help="Path to save the figure (PNG/PDF/SVG). "
                        "If omitted, the plot is displayed interactively.")
    p.add_argument("--dpi",   type=int, default=150,
                   help="Output DPI when saving.")
    p.add_argument(
        "--figsize", nargs=2, type=float, metavar=("W", "H"), default=None,
        help="Figure width and height in inches (e.g. --figsize 20 8). "
             "Auto-scales by number of runs if omitted.",
    )
    return p.parse_args()


def main() -> None:
    args       = parse_args()
    model_dirs = [Path(d) for d in args.model_dir]
    save_path  = Path(args.save) if args.save else None
    figsize    = tuple(args.figsize) if args.figsize else None
    plot(model_dirs, save_path, args.dpi, figsize=figsize)


if __name__ == "__main__":
    main()
