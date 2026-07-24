"""
run_comparisons.py

Generate all sweep comparison plots in one go.  An alternative to calling
plot_history.py from the CLI for each sweep individually.

Each comparison group produces one figure saved under:
    experiment_outputs/July23/comparisons/

Usage:
    python run_comparisons.py                  # all sweeps
    python run_comparisons.py --sweep arch lr  # specific sweeps only
    python run_comparisons.py --show           # display instead of saving

Variants with missing results are skipped automatically so you can run this
incrementally while training is still in progress.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Import plot() directly — no subprocess, no CLI parsing overhead.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluation.plot_history import plot

# ── paths ─────────────────────────────────────────────────────────────────────
OUT_ROOT   = Path("experiment_outputs/July23")
SAVE_DIR   = OUT_ROOT / "comparisons"
METRICS    = ["auroc", "auprc", "f1"]

# ── sweep definitions ─────────────────────────────────────────────────────────
# Each entry: (sweep_key, display_title, [variant_dirs])
SWEEPS: list[tuple[str, str, list[Path]]] = [
    (
        "arch",
        "Architecture sweep  (S / M / L)",
        [
            OUT_ROOT / "arch_sweep" / "S",
            OUT_ROOT / "arch_sweep" / "M",
            OUT_ROOT / "arch_sweep" / "L",
        ],
    ),
    (
        "lr",
        "Learning rate sweep  (LR1–LR5)",
        [OUT_ROOT / "lr_sweep" / v for v in ["LR1", "LR2", "LR3", "LR4", "LR5"]],
    ),
    (
        "wd",
        "Weight decay sweep  (WD1–WD5)",
        [OUT_ROOT / "wd_sweep" / v for v in ["WD1", "WD2", "WD3", "WD4", "WD5"]],
    ),
    (
        "dropout",
        "Dropout sweep  (D1–D5)",
        [OUT_ROOT / "dropout_sweep" / v for v in ["D1", "D2", "D3", "D4", "D5"]],
    ),
    (
        "ls",
        "Label smoothing sweep  (LS1–LS5)",
        [OUT_ROOT / "ls_sweep" / v for v in ["LS1", "LS2", "LS3", "LS4", "LS5"]],
    ),
]


# ── helpers ───────────────────────────────────────────────────────────────────

def _available(variant_dirs: list[Path]) -> list[Path]:
    """Return only variant dirs that contain at least one seed with history.json."""
    ready = []
    for d in variant_dirs:
        has_direct = (d / "history.json").exists()
        has_seeds  = any(
            (sub / "history.json").exists()
            for sub in d.iterdir()
            if sub.is_dir()
        ) if d.exists() else False
        if has_direct or has_seeds:
            ready.append(d)
    return ready


# ── main ──────────────────────────────────────────────────────────────────────

def run(selected: list[str] | None, show: bool) -> None:
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    for sweep_key, title, variant_dirs in SWEEPS:
        if selected and sweep_key not in selected:
            continue

        ready = _available(variant_dirs)
        if not ready:
            print(f"[skip] {sweep_key}: no results found yet")
            continue

        skipped = len(variant_dirs) - len(ready)
        label   = f"[{sweep_key}] {title}"
        if skipped:
            label += f"  ({skipped} variant(s) missing, skipped)"
        print(label)

        save_path = None if show else SAVE_DIR / f"{sweep_key}_sweep.png"
        plot(
            model_dirs = ready,
            metrics    = METRICS,
            save_path  = save_path,
            dpi        = 150,
        )

    if not show:
        print(f"\nFigures saved to: {SAVE_DIR}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate all sweep comparison plots.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--sweep", nargs="+",
        choices=["arch", "lr", "wd", "dropout", "ls"],
        default=None,
        help="Which sweeps to plot (default: all).",
    )
    p.add_argument(
        "--show", action="store_true",
        help="Display interactively instead of saving to file.",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(selected=args.sweep, show=args.show)
