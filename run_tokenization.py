"""
run_tokenization.py

Define tokenization variants in RUNS and execute them all with:
    python run_tokenization.py

Each entry produces an independent output folder under tokenization_outputs/.
Point run_train.py at the folder you want to train on.

─── Tokenization flags ──────────────────────────────────────────────────────────
  insert_att               ATT tokens (W0-W3, M1-M11, LT) between visits — needed
                           for C1/C2 embedding ablations
  insert_visit_delimiters  [V_START]/[V_END] around each visit block
  bucket_labs              Append per-itemid quantile bucket (_Q1–_Q4) to abnormal
                           lab tokens — changes vocab, requires re-tokenization
  bucket_medications       Append per-drug dose-tier bucket (_Q1–_Q4) to medication
                           tokens — changes vocab, requires re-tokenization
────────────────────────────────────────────────────────────────────────────────
"""

from pathlib import Path

from configs import TokenizationConfig
import tokenization_src.tokenize_cycle_sequences as tok_module
import tokenization_src.summarize_tokenization as summary_module

REPO_ROOT = Path(__file__).resolve().parent

# ── shared settings ───────────────────────────────────────────────────────────
_BASE = dict(
    data_dir    = REPO_ROOT.parent / "MIMIC_IV_raw_data",
    cohort_name = "cycle_modeling_v3",
    max_seq_len = 512,
    run_split   = False,
    run_summarize = True,
)

# ── tokenization variants ─────────────────────────────────────────────────────
RUNS = [
    # Base: no ATT tokens, no bucketing
    TokenizationConfig(**_BASE,
        output_name = "Jul1_512",
    ),

    # ATT tokens (needed for C1/C2 embedding ablations)
    TokenizationConfig(**_BASE,
        output_name = "Jul1_512_att",
        insert_att  = True,
    ),

    # Bucketed labs
    TokenizationConfig(**_BASE,
        output_name = "Jul1_512_bucketed_labs",
        bucket_labs = True,
    ),

    # Bucketed medications
    TokenizationConfig(**_BASE,
        output_name = "Jul1_512_bucketed_meds",
        bucket_medications = True,
    ),

    # Bucketed labs + medications
    TokenizationConfig(**_BASE,
        output_name            = "Jul1_512_bucketed_all",
        bucket_labs            = True,
        bucket_medications     = True,
    ),
]

# ── run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    for i, cfg in enumerate(RUNS, 1):
        print(f"\n{'=' * 55}")
        print(f"  Tokenization {i}/{len(RUNS)}  →  {cfg.output_dir.name}")
        print(f"    insert_att             : {cfg.insert_att}")
        print(f"    insert_visit_delimiters: {cfg.insert_visit_delimiters}")
        print(f"    bucket_labs            : {cfg.bucket_labs}")
        print(f"    bucket_medications     : {cfg.bucket_medications}")
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
        )

        if cfg.run_summarize:
            print("\n── summarize ───────────────────────────────────────────────────────")
            summary_module.main(cfg.output_dir)
