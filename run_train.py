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
    epochs          = 100,
    batch_size      = 16,
    lr              = 1e-4,
    weight_decay    = 5e-2,
    label_smoothing = 0.1,
    dropout         = 0.3,
    device          = "auto",
    num_workers     = 2,
    use_wandb       = False,

    # Embedding config
    fusion          = "add",
    use_time        = False,
    use_age         = False,
    data_dir        = Path("tokenization_outputs/Jul17_512_all_labs"),
)

# ── architecture sweep ────────────────────────────────────────────────────────
# num_heads must divide d_model. ff_dim is kept at 2× d_model throughout.
ARCH = [
    ("S", dict(d_model=64,  num_heads=4, num_layers=1, ff_dim=128)),
    ("M", dict(d_model=128, num_heads=4, num_layers=2, ff_dim=256)),
    ("L", dict(d_model=128, num_heads=8, num_layers=4, ff_dim=512)),
]

# ── define runs ───────────────────────────────────────────────────────────────

SEEDS = [42, 52, 62, 72, 82]
RUNS  = []

for arch_id, arch_kwargs in ARCH:
    for s in SEEDS:
        RUNS.append(TrainConfig(
            **_BASE,
            **arch_kwargs,
            seed       = s,
            output_dir = Path(f"experiment_outputs/arch_sweep/{arch_id}/seed{s}"),
            run_name   = f"arch-{arch_id}-seed{s}",
        ))

# ── run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    for i, cfg in enumerate(RUNS, 1):
        print(f"\n{'=' * 55}")
        print(f"  Run {i}/{len(RUNS)}  →  {cfg.output_dir}")
        print(f"{'=' * 55}")
        cfg.save(cfg.output_dir / "config.json")
        train_module.train(cfg)
