"""
run_cohort.py

Run this once (or when SQL logic changes) to build the cohort table.

    python run_cohort.py
"""

from pathlib import Path

from configs import CohortConfig
import cohort_src.generate_cycle_modeling_table as cohort_module

# ── config ────────────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parent

cfg = CohortConfig(
    data_dir    = REPO_ROOT.parent / "MIMIC_IV_raw_data"
    output_name = "cycle_modeling_July24_v2",
)

# ── run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"data_dir    : {cfg.data_dir}")
    print(f"output_dir  : {cfg.output_dir}")
    print()
    cohort_module.main(
        data_location = cfg.data_dir,
        output_name   = cfg.output_name,
    )
