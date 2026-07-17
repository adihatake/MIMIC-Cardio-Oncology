"""
visualize_attributions.py

Reads attributions.csv produced by interpret.py and generates summary plots.

Outputs (written alongside the CSV unless --output-dir is given)
----------------------------------------------------------------
  comparison_top{K}.png    IG vs rollout dual bar chart, same top-K tokens on both panels
  event_type_breakdown.png  Total attribution by event type (IG + rollout, grouped bars)
  rollout_vs_ig_scatter.png Per-token scatter: rollout score vs IG score

Usage
-----
  # Explain outputs are in interpretation/outputs/12345_cycle2/
  python interpretation/visualize_attributions.py \\
      --input-dir interpretation/outputs/12345_cycle2

  # Or point directly at the CSV
  python interpretation/visualize_attributions.py \\
      --csv interpretation/outputs/sample_42/attributions.csv

  # Write plots to a different directory
  python interpretation/visualize_attributions.py \\
      --input-dir interpretation/outputs/12345_cycle2 \\
      --output-dir interpretation/figures/12345_cycle2

  # Adjust how many tokens appear in the comparison chart
  python interpretation/visualize_attributions.py \\
      --input-dir interpretation/outputs/12345_cycle2 --top-k 20
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── Event type helpers (mirrors interpret.py — no shared import needed) ───────

EVENT_TYPE_COLORS: dict[str, str] = {
    "special":    "#999999",
    "diagnosis":  "#4e79a7",
    "procedure":  "#76b7b2",
    "medication": "#f28e2b",
    "lab":        "#e15759",
}

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


def _event_type(raw_token: str) -> str:
    if "::" not in raw_token:
        return "special"
    prefix = raw_token.split("::")[0]
    return prefix if prefix in EVENT_TYPE_COLORS else "special"


def _legend_handles(types: list[str] | None = None) -> list[mpatches.Patch]:
    items = types or list(EVENT_TYPE_COLORS.keys())
    return [mpatches.Patch(color=EVENT_TYPE_COLORS[t], label=t) for t in items if t in EVENT_TYPE_COLORS]


# ── Plot 1: IG vs Rollout dual bar chart ─────────────────────────────────────

def plot_comparison(
    df: pd.DataFrame,
    output_path: Path,
    top_k: int = 30,
    title_prefix: str = "",
) -> None:
    """
    Two vertically stacked panels showing the same top-K tokens (ranked by IG),
    coloured by event type.  Top panel = IG attribution, bottom panel = rollout.

    If no ig_score column is present (IG was skipped), only the rollout panel
    is drawn.
    """
    has_ig = "ig_score" in df.columns and df["ig_score"].notna().any()

    # Rank by IG score if available, else by rollout
    rank_col = "ig_score" if has_ig else "rollout_score"
    top = df.nlargest(top_k, rank_col).reset_index(drop=True)

    colors = [EVENT_TYPE_COLORS.get(_event_type(r), "#999999") for r in top["token"]]
    x      = np.arange(len(top))
    labels = top["label"].tolist()

    n_panels = 2 if has_ig else 1
    fig, axes = plt.subplots(
        n_panels, 1,
        figsize=(max(10, top_k * 0.45), 4.5 * n_panels),
        sharex=True,
    )
    if n_panels == 1:
        axes = [axes]

    panel_idx = 0

    if has_ig:
        ax = axes[panel_idx]
        ax.bar(x, top["ig_score"], color=colors, edgecolor="white", linewidth=0.4)
        ax.set_ylabel("IG attribution\n(L2 norm over d_model)", fontsize=9)
        ax.set_title(
            (f"{title_prefix}  —  " if title_prefix else "") + f"Top {top_k} tokens by IG score",
            fontsize=11,
        )
        ax.legend(handles=_legend_handles(), fontsize=7, loc="upper right")
        panel_idx += 1

    ax = axes[panel_idx]
    ax.bar(x, top["rollout_score"], color=colors, edgecolor="white", linewidth=0.4)
    ax.set_ylabel("Rollout relevance score", fontsize=9)
    if not has_ig:
        ax.set_title(
            (f"{title_prefix}  —  " if title_prefix else "") + f"Top {top_k} tokens by rollout score",
            fontsize=11,
        )
        ax.legend(handles=_legend_handles(), fontsize=7, loc="upper right")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=55, ha="right", fontsize=7)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {output_path.name}")


# ── Plot 2: Event-type breakdown ─────────────────────────────────────────────

def plot_event_type_breakdown(
    df: pd.DataFrame,
    output_path: Path,
    title_prefix: str = "",
) -> None:
    """
    Horizontal grouped bar chart: total IG and rollout attribution per event type.
    Helps answer "does the model rely more on labs than diagnoses?"
    """
    has_ig = "ig_score" in df.columns and df["ig_score"].notna().any()

    df = df.copy()
    df["event_type"] = df["token"].apply(_event_type)

    score_cols = ["rollout_score"] + (["ig_score"] if has_ig else [])
    breakdown = (
        df.groupby("event_type")[score_cols]
        .sum()
        .reindex(list(EVENT_TYPE_COLORS.keys()))
        .dropna(how="all")
        .fillna(0)
    )

    n_types = len(breakdown)
    y       = np.arange(n_types)
    bar_h   = 0.35

    fig, ax = plt.subplots(figsize=(8, max(3, n_types * 0.7)))

    rollout_bars = ax.barh(
        y - bar_h / 2 if has_ig else y,
        breakdown["rollout_score"],
        height=bar_h,
        color=[EVENT_TYPE_COLORS.get(t, "#999999") for t in breakdown.index],
        alpha=0.65,
        label="Rollout",
        edgecolor="white",
    )

    if has_ig:
        ax.barh(
            y + bar_h / 2,
            breakdown["ig_score"],
            height=bar_h,
            color=[EVENT_TYPE_COLORS.get(t, "#999999") for t in breakdown.index],
            alpha=1.0,
            label="IG",
            edgecolor="white",
            hatch="///",
        )

    ax.set_yticks(y)
    ax.set_yticklabels(breakdown.index, fontsize=10)
    ax.set_xlabel("Total attribution (sum across all tokens of this type)", fontsize=9)
    ax.set_title(
        (f"{title_prefix}  —  " if title_prefix else "") + "Attribution by event type",
        fontsize=11,
    )
    ax.legend(fontsize=9)
    ax.invert_yaxis()

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {output_path.name}")


# ── Plot 3: Rollout vs IG scatter ────────────────────────────────────────────

def plot_scatter(
    df: pd.DataFrame,
    output_path: Path,
    title_prefix: str = "",
) -> None:
    """
    Scatter plot of rollout score vs IG score for every token.
    Tokens where both methods agree (both high) appear in the top-right.
    Divergences reveal where attention and gradient-based attribution disagree.
    Not produced if IG scores are missing.
    """
    if "ig_score" not in df.columns or not df["ig_score"].notna().any():
        return

    df = df.copy()
    df["event_type"] = df["token"].apply(_event_type)

    fig, ax = plt.subplots(figsize=(7, 6))

    for etype, group in df.groupby("event_type"):
        color = EVENT_TYPE_COLORS.get(etype, "#999999")
        ax.scatter(
            group["rollout_score"],
            group["ig_score"],
            c=color,
            alpha=0.6,
            s=25,
            label=etype,
            linewidths=0,
        )

    # Annotate the top-5 tokens by IG score
    top5 = df.nlargest(5, "ig_score")
    for _, row in top5.iterrows():
        ax.annotate(
            row["label"],
            xy=(row["rollout_score"], row["ig_score"]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=6.5,
            color="#333333",
        )

    ax.set_xlabel("Rollout relevance score", fontsize=9)
    ax.set_ylabel("IG attribution (L2 norm)", fontsize=9)
    ax.set_title(
        (f"{title_prefix}  —  " if title_prefix else "") + "Rollout vs Integrated Gradients",
        fontsize=11,
    )
    ax.legend(fontsize=8, loc="upper left", markerscale=1.4)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {output_path.name}")


# ── Public entry point ────────────────────────────────────────────────────────

def visualize(
    csv_path: Path,
    output_dir: Path,
    top_k: int = 30,
    title_prefix: str = "",
) -> None:
    """
    Load attributions.csv and write all three plots to output_dir.
    Safe to call even if ig_score column is absent (IG was skipped).
    """
    csv_path   = Path(csv_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not csv_path.exists():
        print(f"  [visualize] attributions.csv not found at {csv_path} — skipping")
        return

    df = pd.read_csv(csv_path)
    print(f"  Loaded {len(df)} tokens from {csv_path.name}")

    if title_prefix == "" and output_dir.name:
        title_prefix = output_dir.name.replace("_", " ")

    plot_comparison(
        df, output_dir / f"comparison_top{top_k}.png",
        top_k=top_k, title_prefix=title_prefix,
    )
    plot_event_type_breakdown(
        df, output_dir / "event_type_breakdown.png",
        title_prefix=title_prefix,
    )
    plot_scatter(
        df, output_dir / "rollout_vs_ig_scatter.png",
        title_prefix=title_prefix,
    )


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Visualize attributions.csv from interpret.py.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--input-dir", default=None,
        help="interpret.py output directory containing attributions.csv",
    )
    src.add_argument(
        "--csv", default=None,
        help="Direct path to an attributions.csv file",
    )
    p.add_argument(
        "--output-dir", default=None,
        help="Where to write plots (default: same directory as the CSV)",
    )
    p.add_argument(
        "--top-k", type=int, default=30,
        help="Number of top-scoring tokens in the comparison chart",
    )
    p.add_argument(
        "--title", default="",
        help="Optional prefix for all plot titles",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    if args.input_dir:
        csv_path = Path(args.input_dir) / "attributions.csv"
    else:
        csv_path = Path(args.csv)

    output_dir = Path(args.output_dir) if args.output_dir else csv_path.parent

    visualize(csv_path, output_dir, top_k=args.top_k, title_prefix=args.title)


if __name__ == "__main__":
    main()
