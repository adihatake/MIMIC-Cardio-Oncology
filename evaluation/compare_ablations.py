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

def _collect(root: Path, metric: str = "auroc") -> dict[str, list[dict]]:
    """
    Walk root and group test_metrics files by ablation ID.  Handles both:
        root/<ablation>/<seed>/test_metrics.json   ← standard multi-seed layout
        root/<ablation>/test_metrics.json           ← single-run layout

    For each run directory, prefers test_metrics_{metric}.json (the checkpoint
    optimised for that metric) and falls back to test_metrics.json.
    """
    groups: dict[str, list[dict]] = defaultdict(list)

    # Collect all run directories that contain any test_metrics*.json
    run_dirs: set[Path] = set()
    for p in root.rglob("test_metrics*.json"):
        run_dirs.add(p.parent)

    for run_dir in sorted(run_dirs):
        # pick the right metrics file
        preferred = run_dir / f"test_metrics_{metric}.json"
        fallback  = run_dir / "test_metrics.json"
        metrics_path = preferred if preferred.exists() else fallback
        if not metrics_path.exists():
            continue

        rel   = run_dir.relative_to(root)
        parts = rel.parts

        if len(parts) == 2:
            group_id = parts[0]   # ablation_id/seed
        elif len(parts) == 1:
            group_id = parts[0]   # ablation_id
        else:
            group_id = str(rel)

        with open(metrics_path) as f:
            metrics = json.load(f)
        metrics["_path"]       = str(metrics_path)
        metrics["_ckpt_metric"] = metric if preferred.exists() else "auroc (fallback)"

        cfg_path = run_dir / "config.json"
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
        aurocs       = [e["auroc"]       for e in entries if "auroc"       in e]
        auprcs       = [e["auprc"]       for e in entries if "auprc"       in e]
        f1s          = [e["f1"]          for e in entries if "f1"          in e]
        sensitivities = [e["sensitivity"] for e in entries if "sensitivity" in e]
        specificities = [e["specificity"] for e in entries if "specificity" in e]
        losses       = [e["loss"]        for e in entries if "loss"        in e]

        first = entries[0]
        rows.append({
            "id":       group_id,
            "n":        len(entries),
            "fusion":   first.get("_fusion",   "?"),
            "use_time": first.get("_use_time", "?"),
            "use_age":  first.get("_use_age",  "?"),
            **{f"auroc_{k}":       v for k, v in _stats(aurocs).items()},
            **{f"auprc_{k}":       v for k, v in _stats(auprcs).items()},
            **{f"f1_{k}":          v for k, v in _stats(f1s).items()},
            **{f"sensitivity_{k}": v for k, v in _stats(sensitivities).items()},
            **{f"specificity_{k}": v for k, v in _stats(specificities).items()},
            **{f"loss_{k}":        v for k, v in _stats(losses).items()},
        })
    return rows


# ── display ───────────────────────────────────────────────────────────────────

def _fmt(v: float, decimals: int = 4) -> str:
    return f"{v:.{decimals}f}" if v == v else "  —  "   # nan check


def print_table(rows: list[dict], sort_by: str) -> None:
    sort_key = f"{sort_by}_mean" if sort_by != "id" else "id"
    if sort_by != "id":
        rows = sorted(rows, key=lambda r: r.get(sort_key, float("-inf")), reverse=True)
    else:
        rows = sorted(rows, key=lambda r: r["id"])

    header = (
        f"{'ID':<8}  {'fusion':<8}  {'time':<5}  {'age':<5}  "
        f"{'n':>2}  {'AUROC':>10}  {'±':>6}  {'AUPRC':>8}  {'±':>6}  "
        f"{'F1':>6}  {'Sens':>6}  {'Spec':>6}  {'loss':>7}"
    )
    sep = "─" * len(header)
    print(sep)
    print(header)
    print(sep)
    for r in rows:
        print(
            f"{r['id']:<8}  {str(r['fusion']):<8}  {str(r['use_time']):<5}  {str(r['use_age']):<5}  "
            f"{r['n']:>2}  {_fmt(r['auroc_mean']):>10}  {_fmt(r['auroc_std']):>6}  "
            f"{_fmt(r.get('auprc_mean', float('nan'))):>8}  {_fmt(r.get('auprc_std', float('nan'))):>6}  "
            f"{_fmt(r.get('f1_mean', float('nan'))):>6}  "
            f"{_fmt(r.get('sensitivity_mean', float('nan'))):>6}  "
            f"{_fmt(r.get('specificity_mean', float('nan'))):>6}  "
            f"{_fmt(r['loss_mean']):>7}"
        )
    print(sep)
    best_row = max(rows, key=lambda r: r.get(sort_key, float("-inf"))) if sort_by != "id" else rows[0]
    best_val = best_row.get(sort_key, float("nan"))
    print(f"  {len(rows)} ablation(s) — best by {sort_by}: {best_row['id']}"
          f" ({_fmt(best_val)} mean {sort_by.upper()})")


# ── plot ──────────────────────────────────────────────────────────────────────

def plot_comparison(
    rows:      list[dict],
    save_path: Path | None,
    dpi:       int,
    metric:    str = "auroc",
    figsize:   tuple[float, float] | None = None,
) -> None:
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        import numpy as np
    except ImportError:
        print("matplotlib is required.  pip install matplotlib")
        sys.exit(1)

    mean_key = f"{metric}_mean"
    std_key  = f"{metric}_std"

    rows  = sorted(rows, key=lambda r: r["id"])
    ids   = [r["id"]                          for r in rows]
    means = [r.get(mean_key, float("nan"))    for r in rows]
    stds  = [r.get(std_key,  0.0)            for r in rows]
    n     = len(ids)

    x = np.arange(n)

    # Color: XGBoost baselines get a warm orange, transformer ablations stay steelblue
    colors = ["#e07b39" if rid.upper().startswith("XGB") else "steelblue" for rid in ids]

    if figsize is None:
        figsize = (max(10, n * 1.8), max(6, n * 0.5))

    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.bar(
        x, means, yerr=stds, capsize=5, width=0.6,
        color=colors, alpha=0.85, ecolor="black", error_kw={"linewidth": 1.5},
    )

    for bar, mean, std in zip(bars, means, stds):
        if mean != mean:   # nan
            continue
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + std + 0.008,
            f"{mean:.3f}\n±{std:.3f}",
            ha="center", va="bottom", fontsize=9,
        )

    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(ids, fontsize=11, rotation=30, ha="right")
    ax.set_ylabel(f"Test {metric.upper()}", fontsize=13)
    ax.set_title(f"Ablation comparison — Test {metric.upper()} (mean ± std across seeds)", fontsize=14)
    ax.set_ylim(0, 1.12)
    ax.grid(axis="y", alpha=0.3)

    legend_handles = [
        mpatches.Patch(color="steelblue", alpha=0.85, label="Transformer ablation"),
        mpatches.Patch(color="#e07b39",   alpha=0.85, label="XGBoost baseline"),
        plt.Line2D([0], [0], color="gray", linestyle="--", linewidth=0.9, label="Random (0.5)"),
    ]
    ax.legend(
        handles=legend_handles,
        fontsize=10,
        loc="upper left",
        bbox_to_anchor=(1.01, 1),
        borderaxespad=0,
        frameon=True,
    )

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"Saved to: {save_path}")
    else:
        plt.show()


# ── CLI ───────────────────────────────────────────────────────────────────────

METRIC_CHOICES = ["auroc", "auprc", "f1", "sensitivity", "specificity"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compare test metrics across ablation runs (mean ± std across seeds).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("root", help="Root experiment directory (e.g. experiment_outputs/Jul1_ablations/)")
    p.add_argument("--save",   default=None,
                   help="Save bar chart to this path (PNG/PDF). Omit to display interactively.")
    p.add_argument("--sort",   default="id", choices=["id"] + METRIC_CHOICES,
                   help="Sort table rows by ablation ID or by any mean metric.")
    p.add_argument("--metric", default="auroc", choices=METRIC_CHOICES,
                   help="Metric to display in the bar chart.")
    p.add_argument("--dpi",    type=int, default=150)
    p.add_argument(
        "--figsize", nargs=2, type=float, metavar=("W", "H"), default=None,
        help="Figure width and height in inches (e.g. --figsize 20 8). "
             "Auto-scales by number of bars if omitted.",
    )
    p.add_argument("--no-plot", action="store_true",
                   help="Print table only, skip the bar chart.")
    return p.parse_args()


def main() -> None:
    args  = parse_args()
    root  = Path(args.root)

    if not root.exists():
        print(f"Directory not found: {root}")
        sys.exit(1)

    groups = _collect(root, metric=args.metric)
    if not groups:
        print(f"No test_metrics.json files found under {root}")
        sys.exit(1)

    rows = summarise(groups)
    print_table(rows, sort_by=args.sort)

    if not args.no_plot:
        save_path = Path(args.save) if args.save else None
        figsize   = tuple(args.figsize) if args.figsize else None
        plot_comparison(rows, save_path, args.dpi, metric=args.metric, figsize=figsize)


if __name__ == "__main__":
    main()
