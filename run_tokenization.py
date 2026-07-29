"""
run_tokenization.py

Tokenize a cohort. Set cohort_name, output names, and flags below, then run:

    python run_tokenization.py

Run after run_cohort.py produces the cohort you want to tokenize.
The tokenizer reads final_cycle_binary_modeling_table.parquet from
cohort_outputs/<cohort_name>/. For the hf_cardiotox pipeline this is the
strict binary table (no pre-existing HF or CMP).
"""

from pathlib import Path

from configs import TokenizationConfig
import tokenization_src.tokenize_cycle_sequences as tok_module
import tokenization_src.summarize_tokenization as summary_module

REPO_ROOT = Path(__file__).resolve().parent

# ── configure here ────────────────────────────────────────────────────────────
_BASE = dict(
    data_dir                = REPO_ROOT.parent / "MIMIC_IV_raw_data",
    cohort_name             = "hf_cardiotox_v2",
    max_seq_len             = 512,
    run_split               = False,
    run_summarize           = True,
    insert_visit_delimiters = True,
    only_abnormal_labs      = False,
    include_all_labs        = True,
)

RUNS = [
    TokenizationConfig(**_BASE, output_name="hf_cardiotox_v2_all_labs",     cardiac_labs_only=False),
    TokenizationConfig(**_BASE, output_name="hf_cardiotox_v2_cardiac_labs", cardiac_labs_only=True),
]
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    for i, cfg in enumerate(RUNS, 1):
        print(f"\n{'=' * 55}")
        print(f"  Tokenization {i}/{len(RUNS)}  →  {cfg.output_dir.name}")
        print(f"    cohort_name            : {cfg.cohort_name}")
        print(f"    insert_att             : {cfg.insert_att}")
        print(f"    insert_visit_delimiters: {cfg.insert_visit_delimiters}")
        print(f"    bucket_labs            : {cfg.bucket_labs}")
        print(f"    bucket_medications     : {cfg.bucket_medications}")
        print(f"    only_abnormal_labs     : {cfg.only_abnormal_labs}")
        print(f"    include_all_labs       : {cfg.include_all_labs}")
        print(f"{'=' * 55}\n")

        print("── tokenize ────────────────────────────────────────────────────────")
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
            print("\n── summarize ───────────────────────────────────────────────────────")
            summary_module.main(cfg.output_dir)
