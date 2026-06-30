"""
run_train.py

The file you edit constantly. Define one or more TrainConfigs and run them.

Single run:
    python run_train.py

─── Seeds and splits ───────────────────────────────────────────────────────────
Each TrainConfig.seed controls TWO things independently:

  1. Patient split  — which patients land in train/val/test (computed at runtime
                      from samples.parquet; tokenization produces no splits.json)
  2. Model init     — weight initialisation and dropout randomness

To estimate variance, repeat each ablation across multiple seeds:
    SEEDS = [42, 43, 44]
    RUNS  = [TrainConfig(**_BASE, seed=s, output_dir=f".../{name}_seed{s}", ...)
             for s in SEEDS for name, kwargs in ABLATIONS]
────────────────────────────────────────────────────────────────────────────────

─── Embedding ablations ────────────────────────────────────────────────────────
Three ablation dimensions, each independently toggleable:

  use_time_embedding   — additive sinusoidal time per token (CEHR-BERT)
  use_concat_embedding — concat [concept, time, age_sin, pos] → FC → GELU
                         + residual type/visit/segment  (CEHR-BERT / EHRMamba)

Tokenisation flags (set when running run_tokenization.py):
  insert_att               ATT tokens (W0-W3, M1-M11, LT) between visits
  insert_visit_delimiters  [V_START]/[V_END] around each visit block

All model ablations require re-running tokenization to generate:
  dates.pt      — needed by use_time_embedding and use_concat_embedding
  age_years.pt  — needed by use_concat_embedding
────────────────────────────────────────────────────────────────────────────────
"""

from pathlib import Path

from configs import TrainConfig
import model_src.train as train_module

# ── shared hyperparameters ────────────────────────────────────────────────────
_BASE = dict(
    data_dir      = Path("tokenization_outputs/Jun26_1000"),
    epochs        = 1,
    batch_size    = 8,
    lr            = 1e-4,
    d_model       = 768,
    num_heads     = 12,
    num_layers    = 12,
    ff_dim        = 3072,
    dropout       = 0.1,
    device        = "auto",
    seed          = 42,       # controls both patient split AND model init
    num_workers   = 6,
    use_wandb     = False,
    wandb_project = "mimic-cardio-oncology",
)

# ── define runs ───────────────────────────────────────────────────────────────
# To run multi-seed experiments, expand like:
#   SEEDS = [42, 43, 44]
#   RUNS  = [TrainConfig(**_BASE, seed=s,
#                        output_dir=Path(f"experiment_outputs/baseline_seed{s}"),
#                        run_name=f"baseline-seed{s}")
#            for s in SEEDS]

RUNS = [
    # 1. Baseline: additive sum, decade-bucket age, sequential position only
    TrainConfig(
        **_BASE,
        output_dir           = Path("experiment_outputs/ablation_baseline"),
        use_time_embedding   = False,
        use_concat_embedding = False,
        run_name             = "baseline",
    ),

    # 2. + additive CEHR-BERT time embedding (requires dates.pt)
    TrainConfig(
        **_BASE,
        output_dir           = Path("experiment_outputs/ablation_time_emb"),
        use_time_embedding   = True,
        use_concat_embedding = False,
        run_name             = "additive-time-emb",
    ),

    # 3. Concat embedding — CEHR-BERT / EHRMamba style
    #    (requires dates.pt + age_years.pt; implies time always on)
    TrainConfig(
        **_BASE,
        output_dir           = Path("experiment_outputs/ablation_concat_emb"),
        use_time_embedding   = False,   # time is part of concat; this flag unused
        use_concat_embedding = True,
        run_name             = "concat-emb-cehrbert",
    ),
]

# ── run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    for i, cfg in enumerate(RUNS, 1):
        print(f"\n{'=' * 55}")
        print(f"  Run {i}/{len(RUNS)}  →  {cfg.output_dir}")
        print(f"{'=' * 55}")
        cfg.save(cfg.output_dir / "config.json")
        train_module.train(cfg)
