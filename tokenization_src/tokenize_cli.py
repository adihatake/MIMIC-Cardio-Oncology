"""
tokenize_cli.py — CLI entry point for tokenization, splitting, and summarizing.

Usage
-----
# Tokenize only (writes to tokenization_outputs/ver1/)
python tokenization_src/tokenize_cli.py --name ver1

# Tokenize from a named cohort
python tokenization_src/tokenize_cli.py --cohort cycle_modeling_v3 --name ver2

# Tokenize + split + summarize in one go
python tokenization_src/tokenize_cli.py --cohort cycle_modeling_v3 --name ver2 --all

# Tokenize + split (no summary)
python tokenization_src/tokenize_cli.py --name ver2 --split

# Adjust max sequence length
python tokenization_src/tokenize_cli.py --name ver2 --max-seq-len 512 --all

# Override raw-data path
python tokenization_src/tokenize_cli.py --name ver2 --data-dir /mnt/data/MIMIC_IV_raw_data --all
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running from the repo root or from within tokenization_src/
sys.path.insert(0, str(Path(__file__).resolve().parent))
import tokenize_cycle_sequences as tok_module
import split_dataset as split_module
import summarize_tokenization as summary_module

REPO_ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="tokenize_cli",
        description="Tokenize cycle sequences, split into train/val/test, and summarize.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--name",
        metavar="DIRNAME",
        default="ver1",
        help="Output directory name under tokenization_outputs/ (e.g. ver2, anthracycline_only).",
    )
    parser.add_argument(
        "--cohort",
        metavar="DIRNAME",
        default="cycle_modeling_ver2",
        help="Cohort directory name under cohort_outputs/ to read from.",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=600,
        metavar="N",
        help="Maximum token sequence length (truncates oldest events).",
    )
    parser.add_argument(
        "--data-dir",
        metavar="PATH",
        required=True,
        help="Path to the MIMIC_IV_raw_data directory.",
    )
    parser.add_argument(
        "--split",
        action="store_true",
        help="Run stratified patient-level train/val/test split after tokenizing.",
    )
    parser.add_argument(
        "--summarize",
        action="store_true",
        help="Print dataset summary statistics after tokenizing (and splitting, if --split).",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Equivalent to --split --summarize.",
    )
    parser.add_argument(
        "--bucket-labs",
        action="store_true",
        help="Append per-itemid quantile bucket (Q1-Q4) to lab tokens.",
    )
    parser.add_argument(
        "--bucket-medications",
        action="store_true",
        help="Append per-drug dose-tier bucket (Q1-Q4) to medication tokens.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    run_split     = args.split    or args.all
    run_summarize = args.summarize or args.all
    data_dir      = Path(args.data_dir).resolve()

    output_dir = REPO_ROOT / "tokenization_outputs" / args.name

    print("=" * 55)
    print("  TOKENIZE CLI")
    print("=" * 55)
    print(f"  cohort        : {args.cohort}")
    print(f"  output name   : {args.name}  →  {output_dir}")
    print(f"  max seq len   : {args.max_seq_len}")
    print(f"  data dir      : {data_dir}")
    print(f"  run split     : {run_split}")
    print(f"  run summarize : {run_summarize}")
    print(f"  bucket labs   : {args.bucket_labs}")
    print(f"  bucket meds   : {args.bucket_medications}")
    print("=" * 55)
    print()

    # ── step 1: tokenize ──────────────────────────────────────────────────────
    print("Step 1/3 — Tokenizing sequences...")
    tok_module.main(
        data_dir=data_dir,
        cohort_name=args.cohort,
        output_name=args.name,
        max_seq_len=args.max_seq_len,
        bucket_labs=args.bucket_labs,
        bucket_medications=args.bucket_medications,
    )

    # ── step 2: split ─────────────────────────────────────────────────────────
    if run_split:
        print("\nStep 2/3 — Splitting into train / val / test...")
        split_module.main(output_dir)
    else:
        print("\nStep 2/3 — Split skipped (pass --split or --all to enable).")

    # ── step 3: summarize ─────────────────────────────────────────────────────
    if run_summarize:
        print("\nStep 3/3 — Summarizing dataset...")
        summary_module.main(input_dir=output_dir)
    else:
        print("Step 3/3 — Summary skipped (pass --summarize or --all to enable).")


if __name__ == "__main__":
    main()
