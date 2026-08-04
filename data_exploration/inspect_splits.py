"""
inspect_splits.py

Show which patients (subject_ids) are assigned to train / val / test for a
given tokenization directory and seed.  The seed can be supplied directly or
read automatically from a model run's config.json.

Usage:
    # From a model run (reads seed + data_dir from config.json)
    python data_exploration/inspect_splits.py --model-dir experiment_outputs/run1

    # Explicit data_dir + seed
    python data_exploration/inspect_splits.py \\
        --data-dir tokenization_outputs/Jul17_512_all_labs --seed 42

    # Show only one split
    python data_exploration/inspect_splits.py --model-dir experiment_outputs/run1 --split test

    # Look up which split a specific patient falls into
    python data_exploration/inspect_splits.py --model-dir experiment_outputs/run1 --subject-id 10006008

    # Export patient lists to CSV
    python data_exploration/inspect_splits.py --model-dir experiment_outputs/run1 --save splits.csv

    # Compare splits across multiple seeds (same data_dir)
    python data_exploration/inspect_splits.py \\
        --data-dir tokenization_outputs/Jul17_512_all_labs --seed 42 52 62
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from model_src.dataset import _compute_row_indices


# ── helpers ───────────────────────────────────────────────────────────────────

def _load_config(model_dir: Path) -> dict:
    cfg_path = model_dir / "config.json"
    if not cfg_path.exists():
        print(f"ERROR: config.json not found in {model_dir}")
        sys.exit(1)
    with open(cfg_path) as f:
        return json.load(f)


def get_split_subjects(
    data_dir: Path,
    seed: int,
) -> tuple[dict[str, list[int]], pd.DataFrame, dict[str, list[int]]]:
    """Return (split_subjects, samples_df, row_indices) for the given data_dir and seed."""
    samples = pd.read_parquet(data_dir / "samples.parquet").reset_index(drop=True)
    row_indices = _compute_row_indices(data_dir, seed)
    split_subjects = {
        split: sorted(samples.iloc[rows]["subject_id"].unique().tolist())
        for split, rows in row_indices.items()
    }
    return split_subjects, samples, row_indices


# ── display ───────────────────────────────────────────────────────────────────

def print_summary(
    split_subjects: dict[str, list[int]],
    row_indices: dict[str, list[int]],
    seed: int,
    data_dir: Path,
    show_ids: bool,
) -> None:
    total_patients = sum(len(v) for v in split_subjects.values())
    total_rows     = sum(len(v) for v in row_indices.values())
    print(f"\ndata_dir : {data_dir}")
    print(f"seed     : {seed}")
    print(f"patients : {total_patients}  |  rows (cycles) : {total_rows}\n")

    header = f"{'split':<6}  {'patients':>8}  {'cycles':>7}  {'% patients':>10}"
    print("─" * len(header))
    print(header)
    print("─" * len(header))
    for split in ("train", "val", "test"):
        sids = split_subjects.get(split, [])
        rows = row_indices.get(split, [])
        pct  = 100 * len(sids) / total_patients if total_patients else 0
        print(f"{split:<6}  {len(sids):>8}  {len(rows):>7}  {pct:>9.1f}%")
    print("─" * len(header))

    if show_ids:
        for split in ("train", "val", "test"):
            sids = split_subjects.get(split, [])
            print(f"\n{split} ({len(sids)} patients):")
            print("  " + ", ".join(str(s) for s in sids))


def lookup_subject(
    split_subjects: dict[str, list[int]],
    row_indices: dict[str, list[int]],
    samples: pd.DataFrame,
    subject_id: int,
    seed: int,
) -> None:
    for split, sids in split_subjects.items():
        if subject_id in sids:
            rows = [r for r in row_indices[split]
                    if samples.iloc[r]["subject_id"] == subject_id]
            print(f"\nsubject_id {subject_id} → {split.upper()}  "
                  f"(seed={seed}, {len(rows)} cycle row(s): {rows})")
            return
    print(f"\nsubject_id {subject_id} not found in any split for seed={seed}.")


def save_csv(
    split_subjects: dict[str, list[int]],
    row_indices: dict[str, list[int]],
    samples: pd.DataFrame,
    seed: int,
    save_path: Path,
) -> None:
    rows_out = []
    for split, sids in split_subjects.items():
        for sid in sids:
            cycle_rows = [r for r in row_indices[split]
                          if samples.iloc[r]["subject_id"] == sid]
            rows_out.append({
                "seed":       seed,
                "split":      split,
                "subject_id": sid,
                "n_cycles":   len(cycle_rows),
                "row_indices": ";".join(str(r) for r in cycle_rows),
            })
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        writer.writeheader()
        writer.writerows(rows_out)
    print(f"Saved: {save_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Inspect train/val/test patient assignments for a given seed.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    src = p.add_mutually_exclusive_group()
    src.add_argument("--model-dir", type=Path, default=None,
                     help="Model run directory; reads seed and data_dir from config.json.")
    src.add_argument("--data-dir", type=Path, default=None,
                     help="Tokenization directory (requires --seed).")

    p.add_argument("--seed", type=int, nargs="+", default=None,
                   help="Seed(s) to inspect. Required with --data-dir; "
                        "overrides config.json when used with --model-dir.")
    p.add_argument("--split", choices=["train", "val", "test"], default=None,
                   help="Show details for one split only.")
    p.add_argument("--subject-id", type=int, default=None,
                   help="Look up which split a specific patient belongs to.")
    p.add_argument("--show-ids", action="store_true",
                   help="Print all subject_ids for each split.")
    p.add_argument("--save", type=Path, default=None,
                   help="Export subject_id → split mapping to this CSV path.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ── resolve data_dir and seeds ────────────────────────────────────────────
    if args.model_dir is not None:
        cfg      = _load_config(args.model_dir)
        data_dir = REPO_ROOT / cfg["data_dir"]
        seeds    = args.seed if args.seed else [cfg["seed"]]
    elif args.data_dir is not None:
        data_dir = args.data_dir
        if not args.seed:
            print("ERROR: --seed is required when using --data-dir")
            sys.exit(1)
        seeds = args.seed
    else:
        print("ERROR: provide --model-dir or --data-dir")
        sys.exit(1)

    if not data_dir.exists():
        print(f"ERROR: data_dir not found: {data_dir}")
        sys.exit(1)

    # ── run for each seed ─────────────────────────────────────────────────────
    for seed in seeds:
        all_subjects, samples, all_rows = get_split_subjects(data_dir, seed)

        # filtered view (used for summary display only)
        view_subjects = ({args.split: all_subjects[args.split]} if args.split else all_subjects)
        view_rows     = ({args.split: all_rows[args.split]}     if args.split else all_rows)

        if args.subject_id is not None:
            lookup_subject(all_subjects, all_rows, samples, args.subject_id, seed)
        else:
            print_summary(view_subjects, view_rows, seed, data_dir,
                          show_ids=args.show_ids)

        if args.save:
            save_path = (args.save if len(seeds) == 1
                         else args.save.with_stem(f"{args.save.stem}_seed{seed}"))
            save_csv(all_subjects, all_rows, samples, seed, save_path)


if __name__ == "__main__":
    main()
