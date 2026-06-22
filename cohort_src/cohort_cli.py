"""
cohort_cli.py — CLI entry point for cohort generation.

Usage
-----
# Run with defaults (writes to cohort_outputs/cycle_modeling_ver2/)
python cohort_src/cohort_cli.py

# Name the output directory
python cohort_src/cohort_cli.py --name cycle_modeling_v3

# Override the raw-data path as well
python cohort_src/cohort_cli.py --name cycle_modeling_v3 \\
    --data-dir /mnt/data/MIMIC_IV_raw_data
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running from the repo root or from within cohort_src/
sys.path.insert(0, str(Path(__file__).resolve().parent))
import generate_cycle_modeling_table as cohort_module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="cohort_cli",
        description="Generate cycle-level cardiotoxicity modelling table from MIMIC-IV.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--name",
        metavar="DIRNAME",
        default="cycle_modeling_ver2",
        help="Output directory name under cohort_outputs/ (e.g. cycle_modeling_v3).",
    )
    parser.add_argument(
        "--data-dir",
        metavar="PATH",
        default=None,
        help=(
            "Path to the MIMIC_IV_raw_data directory. "
            "Defaults to the path hard-coded in generate_cycle_modeling_table.py."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    data_dir = Path(args.data_dir).resolve() if args.data_dir else None

    print(f"cohort output name : {args.name}")
    if data_dir:
        print(f"data dir override  : {data_dir}")
    print()

    cohort_module.main(output_name=args.name, data_location=data_dir)


if __name__ == "__main__":
    main()
