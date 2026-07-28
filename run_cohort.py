"""
run_cohort.py

Build the cohort table. Set PIPELINE and OUTPUT_NAME below, then run:

    python run_cohort.py

To add a new pipeline: define a PipelineConfig in cohort_src/cohort_pipeline.py
and register it in PIPELINE_REGISTRY. No other Python changes needed.

PIPELINE options  (see cohort_src/cohort_pipeline.py):
    "hf_cardiotox"  — HF endpoint (3 drug classes, per-class monitoring windows)
    "main"          — LVEF + CV toxicity endpoint
"""

from pathlib import Path
import cohort_src.generate_cohort as cohort_module

REPO_ROOT = Path(__file__).resolve().parent

# ── configure here ────────────────────────────────────────────────────────────
PIPELINE    = "hf_cardiotox"    # "hf_cardiotox" | "main"
OUTPUT_NAME = "hf_cardiotox_v1"

# Override SQL dirs only when pointing at a non-default version subdirectory
# (e.g. "jul28_alt" instead of the canonical "hf_cardiotox" folder).
# Leave as None to use the defaults: sql_files/drug_cycles_sql/<PIPELINE>/
CYCLE_SQL_DIR         = None
PRESCRIPTIONS_SQL_DIR = None
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    data_dir = REPO_ROOT.parent / "MIMIC_IV_raw_data"

    print(f"pipeline    : {PIPELINE}")
    print(f"output_name : {OUTPUT_NAME}")
    print(f"data_dir    : {data_dir}")
    print()

    cohort_module.main(
        data_location         = data_dir,
        pipeline              = PIPELINE,
        output_name           = OUTPUT_NAME,
        cycle_sql_dir         = CYCLE_SQL_DIR,
        prescriptions_sql_dir = PRESCRIPTIONS_SQL_DIR,
    )
