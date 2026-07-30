"""
run_cohort.py

Build the cohort table. Set PIPELINE and OUTPUT_NAME below, then run:

    python run_cohort.py

To add a new pipeline: define a PipelineConfig in cohort_src/cohort_pipeline.py
and register it in PIPELINE_REGISTRY. No other Python changes needed.

PIPELINE options  (see cohort_src/cohort_pipeline.py):
    "anthracycline_only_exposure" — anthracycline-first patients only, combined CTRCD endpoint
    "hf_cardiotox_v2"            — combined CTRCD endpoint (ICD HF + LVEF + GLS, 3 drug classes, ESC 2022 windows)
    "pan_cancer_ctrcd"           — 10 pan-cancer drug classes, combined CTRCD endpoint, uniform 365-day window
    "pan_cancer_bimodal"         — same as pan_cancer_ctrcd but acute (90d) vs moderate (365d) windows
    "hf_cardiotox"               — ICD HF endpoint only (3 drug classes, per-class windows)
    "main"                       — LVEF + CV toxicity endpoint

Canonical training table convention:
    final_cycle_binary_modeling_table.parquet is always the INCLUSIVE table
    (retains pre-existing HF/CMP patients — largest N for prediction models).
    The strict table is written separately for sensitivity / etiological analyses.
"""

from pathlib import Path
import cohort_src.generate_cohort as cohort_module

REPO_ROOT = Path(__file__).resolve().parent

# ── configure here ────────────────────────────────────────────────────────────
PIPELINE    = "pan_cancer_bimodal"    # see options above
OUTPUT_NAME = "pan_cancer_bimodal_v1"

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
