"""
run_cohort.py

Build the cohort table. Set the output name and SQL directories below, then run:

    python run_cohort.py

SQL directory options:
    sql_files/drug_cycles_sql/jul17/  — Jun 23 definitions (Jul17 training data)
    sql_files/drug_cycles_sql/jul24/  — Jul 24 definitions (revised cardiotoxicity)
"""

from pathlib import Path

import cohort_src.generate_cycle_modeling_table as cohort_module

REPO_ROOT = Path(__file__).resolve().parent

# ── configure here ────────────────────────────────────────────────────────────
OUTPUT_NAME           = "cycle_modeling_July24_v2"
CYCLE_SQL_DIR         = REPO_ROOT / "sql_files" / "drug_cycles_sql" / "jul24"
PRESCRIPTIONS_SQL_DIR = REPO_ROOT / "sql_files" / "prescriptions_sql" / "jul24"
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    data_dir = REPO_ROOT.parent / "MIMIC_IV_raw_data"

    print(f"output_name           : {OUTPUT_NAME}")
    print(f"cycle_sql_dir         : {CYCLE_SQL_DIR.name}")
    print(f"prescriptions_sql_dir : {PRESCRIPTIONS_SQL_DIR.name}")
    print(f"data_dir              : {data_dir}")
    print()

    cohort_module.main(
        data_location         = data_dir,
        output_name           = OUTPUT_NAME,
        cycle_sql_dir         = CYCLE_SQL_DIR,
        prescriptions_sql_dir = PRESCRIPTIONS_SQL_DIR,
    )
