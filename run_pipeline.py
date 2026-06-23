"""
run_pipeline.py

Full end-to-end run. Delegates to the three stage scripts.
Use this for reproducibility / first-time setup.

For day-to-day work, edit and run the individual stage scripts:
    python run_cohort.py
    python run_tokenization.py
    python run_train.py
"""

import run_cohort
import run_tokenization
import run_train

import cohort_src.generate_cycle_modeling_table as cohort_module
import tokenization_src.tokenize_cycle_sequences as tok_module
import tokenization_src.split_dataset as split_module
import tokenization_src.summarize_tokenization as summary_module
import model_src.train as train_module

# ── toggle stages ─────────────────────────────────────────────────────────────

RUN_COHORT      = True
RUN_TOKENIZE    = True
RUN_SPLIT       = True
RUN_SUMMARIZE   = True
RUN_TRAIN       = True

# ── run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cohort_cfg = run_cohort.cfg
    tok_cfg    = run_tokenization.cfg
    train_cfgs = run_train.RUNS

    if RUN_COHORT:
        print("=" * 55 + "\n  COHORT\n" + "=" * 55)
        cohort_module.main(
            data_location = cohort_cfg.data_dir,
            output_name   = cohort_cfg.output_name,
        )

    if RUN_TOKENIZE:
        print("=" * 55 + "\n  TOKENIZE\n" + "=" * 55)
        tok_module.main(
            data_dir    = tok_cfg.data_dir,
            cohort_name = tok_cfg.cohort_name,
            output_name = tok_cfg.output_name,
            max_seq_len = tok_cfg.max_seq_len,
        )

    if RUN_SPLIT:
        print("=" * 55 + "\n  SPLIT\n" + "=" * 55)
        split_module.main(tok_cfg.output_dir)

    if RUN_SUMMARIZE:
        print("=" * 55 + "\n  SUMMARIZE\n" + "=" * 55)
        summary_module.main(tok_cfg.output_dir)

    if RUN_TRAIN:
        for i, cfg in enumerate(train_cfgs, 1):
            print(f"\n{'=' * 55}\n  TRAIN {i}/{len(train_cfgs)}  →  {cfg.output_dir}\n{'=' * 55}")
            cfg.save(cfg.output_dir / "config.json")
            train_module.train(cfg)
