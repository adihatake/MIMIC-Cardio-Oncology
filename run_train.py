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
    RUNS  = [TrainConfig(**_BASE, **emb, seed=s, output_dir=..., run_name=...)
             for name, emb in ABLATIONS for s in SEEDS]
────────────────────────────────────────────────────────────────────────────────

─── Embedding ablation flags ────────────────────────────────────────────────────
  fusion    "add"    BEHRT-style element-wise sum of all embedding tables.
            "concat" CEHR-BERT: cat([concept, time*, age*, position]) →
                     Linear(4d→d) → GELU, then type/visit/segment as residuals.
                     Components zeroed when disabled — same weight shape across B0-B2.
  use_time  bool     Sinusoidal time-gap embedding per token (requires dates.pt).
  use_age   bool     Continuous-age sinusoidal embedding (requires age_years.pt).

  Ablation grid:
    A0  add,    use_time=F, use_age=F  — baseline (no temporal info)
    A1  add,    use_time=T, use_age=F  — + relative time gaps
    A2  add,    use_time=F, use_age=T  — + patient age
    A3  add,    use_time=T, use_age=T  — + time + age (best additive)
    B0  concat, use_time=F, use_age=F  — concat fusion only
    B1  concat, use_time=T, use_age=F  — concat + time
    B2  concat, use_time=T, use_age=T  — CEHR-BERT style
    C1  same flags as best A/B, data_dir built with insert_att=True
    C2  concat, use_time=T, use_age=T, data_dir built with insert_att=True

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
    epochs           = 100,
    batch_size       = 16,
    lr               = 1e-4,
    weight_decay     = 5e-2,
    label_smoothing  = 0.1,
    d_model          = 64,
    num_heads        = 4,
    num_layers       = 1,
    ff_dim           = 128,
    dropout          = 0.4,
    device           = "auto",
    num_workers      = 2,
    use_wandb        = False,
)

# ── define runs ───────────────────────────────────────────────────────────────

SEEDS = [42, 43, 44]
RUNS  = []

for s in SEEDS:
    # Unbucketed baseline
    RUNS.append(TrainConfig(
        **_BASE,
        data_dir   = Path("tokenization_outputs/Jul1_512"),
        seed       = s,
        output_dir = Path(f"experiment_outputs/Jul1_dropout_comparison/unbucketed/seed{s}"),
    ))
    # Bucketed baseline (labs + medications)
    RUNS.append(TrainConfig(
        **_BASE,
        data_dir   = Path("tokenization_outputs/Jul1_512_bucketed_all"),
        seed       = s,
        output_dir = Path(f"experiment_outputs/Jul1_dropout_comparison/bucketed/seed{s}_b"),
    ))

# ── stage 2: full embedding ablation sweep (uncomment after picking tokenization) ──
# Replace <CHOSEN_TOKENIZATION> with either "Jul1_512" or "Jul1_512_bucketed_all".
#
# ABLATIONS = [
#     ("A0", dict(fusion="add",    use_time=False, use_age=False)),
#     ("A1", dict(fusion="add",    use_time=True,  use_age=False)),
#     ("A2", dict(fusion="add",    use_time=False, use_age=True)),
#     ("A3", dict(fusion="add",    use_time=True,  use_age=True)),
#     ("B0", dict(fusion="concat", use_time=False, use_age=False)),
#     ("B1", dict(fusion="concat", use_time=True,  use_age=False)),
#     ("B2", dict(fusion="concat", use_time=True,  use_age=True)),
#     # C1/C2: requires Jul1_512_att tokenization
#     # ("C2", dict(fusion="concat", use_time=True, use_age=True)),
# ]
# for ablation_id, emb_kwargs in ABLATIONS:
#     for s in SEEDS:
#         RUNS.append(TrainConfig(
#             **_BASE,
#             **emb_kwargs,
#             data_dir   = Path("tokenization_outputs/<CHOSEN_TOKENIZATION>"),
#             seed       = s,
#             output_dir = Path(f"experiment_outputs/Jul1_ablations/{ablation_id}/seed{s}"),
#             run_name   = f"{ablation_id}-seed{s}",
#         ))

# ── run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    for i, cfg in enumerate(RUNS, 1):
        print(f"\n{'=' * 55}")
        print(f"  Run {i}/{len(RUNS)}  →  {cfg.output_dir}")
        print(f"{'=' * 55}")
        cfg.save(cfg.output_dir / "config.json")
        train_module.train(cfg)
