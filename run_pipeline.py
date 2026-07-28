"""
run_pipeline.py

Run cohort + tokenization in one go. Configure below, then run:

    python run_pipeline.py

For day-to-day work, run the individual scripts instead:
    python run_cohort.py
    python run_tokenization.py

PIPELINE options  (see cohort_src/cohort_pipeline.py):
    "hf_cardiotox"  — HF endpoint (3 drug classes, per-class monitoring windows)
    "main"          — LVEF + CV toxicity endpoint
"""

from pathlib import Path

from configs import TokenizationConfig
import cohort_src.generate_cohort as cohort_module
import tokenization_src.tokenize_cycle_sequences as tok_module
import tokenization_src.summarize_tokenization as summary_module

REPO_ROOT = Path(__file__).resolve().parent

# ── configure here ────────────────────────────────────────────────────────────
PIPELINE           = "hf_cardiotox"   # "hf_cardiotox" | "main"
COHORT_OUTPUT_NAME = "hf_cardiotox_v1"

RUN_COHORT   = True
RUN_TOKENIZE = True

# Override SQL dirs only when pointing at a non-default version subdirectory.
# Leave as None to use the defaults: sql_files/drug_cycles_sql/<PIPELINE>/
CYCLE_SQL_DIR         = None
PRESCRIPTIONS_SQL_DIR = None
# ─────────────────────────────────────────────────────────────────────────────

_BASE_TOK = dict(
    data_dir                = REPO_ROOT.parent / "MIMIC_IV_raw_data",
    cohort_name             = COHORT_OUTPUT_NAME,
    max_seq_len             = 512,
    run_split               = False,
    run_summarize           = True,
    insert_visit_delimiters = True,
    only_abnormal_labs      = False,
    include_all_labs        = True,
)

TOK_RUNS = [
    TokenizationConfig(**_BASE_TOK, output_name=COHORT_OUTPUT_NAME),
]

if __name__ == "__main__":
    data_dir = REPO_ROOT.parent / "MIMIC_IV_raw_data"

    if RUN_COHORT:
        print("=" * 55 + f"\n  COHORT  [{PIPELINE}]\n" + "=" * 55)
        cohort_module.main(
            data_location         = data_dir,
            pipeline              = PIPELINE,
            output_name           = COHORT_OUTPUT_NAME,
            cycle_sql_dir         = CYCLE_SQL_DIR,
            prescriptions_sql_dir = PRESCRIPTIONS_SQL_DIR,
        )

    if RUN_TOKENIZE:
        for i, cfg in enumerate(TOK_RUNS, 1):
            print(f"\n{'=' * 55}")
            print(f"  TOKENIZE {i}/{len(TOK_RUNS)}  →  {cfg.output_dir.name}")
            print(f"{'=' * 55}")
            tok_module.main(
                data_dir                = cfg.data_dir,
                cohort_name             = cfg.cohort_name,
                output_name             = cfg.output_name,
                max_seq_len             = cfg.max_seq_len,
                insert_att              = cfg.insert_att,
                insert_visit_delimiters = cfg.insert_visit_delimiters,
                bucket_labs             = cfg.bucket_labs,
                bucket_medications      = cfg.bucket_medications,
                only_abnormal_labs      = cfg.only_abnormal_labs,
                include_all_labs        = cfg.include_all_labs,
            )
            if cfg.run_summarize:
                summary_module.main(cfg.output_dir)
