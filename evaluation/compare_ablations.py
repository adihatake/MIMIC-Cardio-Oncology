"""
compare_ablations.py

Scan an experiment directory for test_metrics.json files, group by ablation ID,
and report mean ± std AUROC (and loss) across seeds.

Expected directory structure:
    <root>/
      A0/seed42/test_metrics.json
      A0/seed43/test_metrics.json
      A1/seed42/test_metrics.json
      ...

Usage:
    # Print summary table
    python evaluation/compare_ablations.py experiment_outputs/Jul1_ablations/

    # Save a bar chart (requires matplotlib)
    python evaluation/compare_ablations.py experiment_outputs/Jul1_ablations/ \\
        --save experiment_outputs/Jul1_ablations/comparison.png

    # Sort by mean AUROC instead of ablation name
    python evaluation/compare_ablations.py experiment_outputs/Jul1_ablations/ --sort auroc
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


# ── data loading ──────────────────────────────────────────────────────────────

def _collect(root: Path) -> dict[str, list[dict]]:
    """
    Walk root and group test_metrics.json files by their grandparent directory
    name (the ablation ID).  Handles both:
        root/<ablation>/<seed>/test_metrics.json   ← standard multi-seed layout
        root/<ablation>/test_metrics.json           ← single-run layout
    """
    groups: dict[str, list[dict]] = defaultdict(list)

    for metrics_path in sorted(root.rglob("test_metrics.json")):
        rel = metrics_path.relative_to(root)
        parts = rel.parts  # e.g. ("A0", "seed42", "test_metrics.json")

        if len(parts) == 3:
            group_id = parts[0]   # ablation_id/seed/test_metrics.json
        elif len(parts) == 2:
            group_id = parts[0]   # ablation_id/test_metrics.json
        else:
            group_id = str(metrics_path.parent.relative_to(root))

        with open(metrics_path) as f:
            metrics = json.load(f)
        metrics["_path"] = str(metrics_path)

        # Attach config flags if available
        cfg_path = metrics_path.parent / "config.json"
        if cfg_path.exists():
            with open(cfg_path) as f:
                cfg = json.load(f)
            metrics["_fusion"]   = cfg.get("fusion",   cfg.get("embedding_mode", "?"))
            metrics["_use_time"] = cfg.get("use_time", "?")
            metrics["_use_age"]  = cfg.get("use_age",  "?")
        else:
            metrics["_fusion"]   = "?"
            metrics["_use_time"] = "?"
            metrics["_use_age"]  = "?"

        groups[group_id].append(metrics)

    return dict(groups)


def _stats(values: list[float]) -> dict:
    import statistics
    if not values:
        return {"mean": float("nan"), "std": float("nan"), "min": float("nan"), "max": float("nan")}
    mean = statistics.mean(values)
    std  = statistics.stdev(values) if len(values) > 1 else 0.0
    return {"mean": mean, "std": std, "min": min(values), "max": max(values)}


def summarise(groups: dict[str, list[dict]]) -> list[dict]:
    rows = []
    for group_id, entries in groups.items():
        aurocs = [e["auroc"] for e in entries if "auroc" in e]
        losses = [e["loss"]  for e in entries if "loss"  in e]

        first = entries[0]
        rows.append({
            "id":       group_id,
            "n":        len(entries),
            "fusion":   first.get("_fusion",   "?"),
            "use_time": first.get("_use_time", "?"),
            "use_age":  first.get("_use_age",  "?"),
            **{f"auroc_{k}": v for k, v in _stats(aurocs).items()},
            **{f"loss_{k}":  v for k, v in _stats(losses).items()},
        })
    return rows


# ── display ───────────────────────────────────────────────────────────────────

def _fmt(v: float, decimals: int = 4) -> str:
    return f"{v:.{decimals}f}" if v == v else "  —  "   # nan check


def print_table(rows: list[dict], sort_by: str) -> None:
    if sort_by == "auroc":
        rows = sorted(rows, key=lambda r: r["auroc_mean"], reverse=True)
    else:
        rows = sorted(rows, key=lambda r: r["id"])

    header = (
        f"{'ID':<8}  {'fusion':<8}  {'time':<5}  {'age':<5}  "
        f"{'n':>2}  {'AUROC mean':>10}  {'± std':>7}  {'min':>7}  {'max':>7}  "
        f"{'loss mean':>9}"
    )
    sep = "─" * len(header)
    print(sep)
    print(header)
    print(sep)
    for r in rows:
        print(
            f"{r['id']:<8}  {str(r['fusion']):<8}  {str(r['use_time']):<5}  {str(r['use_age']):<5}  "
            f"{r['n']:>2}  {_fmt(r['auroc_mean']):>10}  {_fmt(r['auroc_std']):>7}  "
            f"{_fmt(r['auroc_min']):>7}  {_fmt(r['auroc_max']):>7}  "
            f"{_fmt(r['loss_mean']):>9}"
        )
    print(sep)
    print(f"  {len(rows)} ablation(s) — best: {max(rows, key=lambda r: r['auroc_mean'])['id']}"
          f" ({_fmt(max(r['auroc_mean'] for r in rows))} mean AUROC)")


# ── plot ──────────────────────────────────────────────────────────────────────

def plot_comparison(rows: list[dict], save_path: Path | None, dpi: int) -> None:
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib is required.  pip install matplotlib")
        sys.exit(1)

    rows = sorted(rows, key=lambda r: r["id"])
    ids    = [r["id"]         for r in rows]
    means  = [r["auroc_mean"] for r in rows]
    stds   = [r["auroc_std"]  for r in rows]

    x = np.arange(len(ids))

    fig, ax = plt.subplots(figsize=(max(8, len(ids) * 1.2), 5))
    bars = ax.bar(x, means, yerr=stds, capsize=5, width=0.6,
                  color="steelblue", alpha=0.85, ecolor="black", error_kw={"linewidth": 1.5})

    # Annotate each bar with mean ± std
    for bar, mean, std in zip(bars, means, stds):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + std + 0.005,
            f"{mean:.3f}\n±{std:.3f}",
            ha="center", va="bottom", fontsize=8,
        )

    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, label="random (0.5)")
    ax.set_xticks(x)
    ax.set_xticklabels(ids, fontsize=11)
    ax.set_ylabel("Test AUROC", fontsize=12)
    ax.set_title("Ablation comparison — Test AUROC (mean ± std across seeds)", fontsize=13)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"Saved to: {save_path}")
    else:
        plt.show()


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compare test AUROC across ablation runs (mean ± std across seeds).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("root", help="Root experiment directory (e.g. experiment_outputs/Jul1_ablations/)")
    p.add_argument("--save",  default=None,
                   help="Save bar chart to this path (PNG/PDF). Omit to display interactively.")
    p.add_argument("--sort",  default="id", choices=["id", "auroc"],
                   help="Sort table rows by ablation ID or by mean AUROC.")
    p.add_argument("--dpi",   type=int, default=150)
    p.add_argument("--no-plot", action="store_true",
                   help="Print table only, skip the bar chart.")
    return p.parse_args()


def main() -> None:
    args  = parse_args()
    root  = Path(args.root)

    if not root.exists():
        print(f"Directory not found: {root}")
        sys.exit(1)

    groups = _collect(root)
    if not groups:
        print(f"No test_metrics.json files found under {root}")
        sys.exit(1)

    rows = summarise(groups)
    print_table(rows, sort_by=args.sort)

    if not args.no_plot:
        save_path = Path(args.save) if args.save else None
        plot_comparison(rows, save_path, args.dpi)


if __name__ == "__main__":
    main()
