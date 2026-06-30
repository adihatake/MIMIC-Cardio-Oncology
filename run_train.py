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

─── Embedding modes (embedding_mode) ───────────────────────────────────────────
  "additive"      BEHRT-style: additive sum of all embedding tables. No time signal.
  "additive+time" Additive sum + sinusoidal time per token (CEHR-BERT formula).
                  Requires dates.pt.
  "concat"        CEHR-BERT / EHRMamba: cat([concept, time, age, position]) →
                  Linear(4d→d) → GELU, then + type + visit + segment residuals.
                  Time and continuous age are always active in this mode.
                  Requires dates.pt and age_years.pt.

Tokenisation flags (set when running run_tokenization.py):
  insert_att               ATT tokens (W0-W3, M1-M11, LT) between visits
  insert_visit_delimiters  [V_START]/[V_END] around each visit block
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
    # 1. Additive — BEHRT-style, no time signal
    TrainConfig(
        **_BASE,
        output_dir     = Path("experiment_outputs/ablation_additive"),
        embedding_mode = "additive",
        run_name       = "additive",
    ),

    # 2. Additive + time — adds sinusoidal time per token (requires dates.pt)
    TrainConfig(
        **_BASE,
        output_dir     = Path("experiment_outputs/ablation_additive_time"),
        embedding_mode = "additive+time",
        run_name       = "additive+time",
    ),

    # 3. Concat — CEHR-BERT / EHRMamba style (requires dates.pt + age_years.pt)
    #    Time and continuous age are always active inside the projection
    TrainConfig(
        **_BASE,
        output_dir     = Path("experiment_outputs/ablation_concat"),
        embedding_mode = "concat",
        run_name       = "concat",
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
