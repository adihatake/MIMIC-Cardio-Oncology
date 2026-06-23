"""
run_tokenization.py

Run this once per cohort variant to tokenize, split, and summarize.

    python run_tokenization.py
"""

from pathlib import Path

from configs import TokenizationConfig
import tokenization_src.tokenize_cycle_sequences as tok_module
import tokenization_src.split_dataset as split_module
import tokenization_src.summarize_tokenization as summary_module

# ── config ────────────────────────────────────────────────────────────────────

cfg = TokenizationConfig(
    data_dir      = Path("/path/to/MIMIC_IV_raw_data"),
    cohort_name   = "cycle_modeling_ver2",
    output_name   = "ver1",
    max_seq_len   = 600,
    run_split     = True,
    run_summarize = True,
)

# ── run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"cohort_dir  : {cfg.cohort_dir}")
    print(f"output_dir  : {cfg.output_dir}")
    print()

    print("── tokenize ────────────────────────────────────────────────────────")
    tok_module.main(
        data_dir    = cfg.data_dir,
        cohort_name = cfg.cohort_name,
        output_name = cfg.output_name,
        max_seq_len = cfg.max_seq_len,
    )

    if cfg.run_split:
        print("\n── split ───────────────────────────────────────────────────────────")
        split_module.main(cfg.output_dir)

    if cfg.run_summarize:
        print("\n── summarize ───────────────────────────────────────────────────────")
        summary_module.main(cfg.output_dir)
