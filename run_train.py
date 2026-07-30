"""
run_train.py

Dataset sweep for the pan_cancer_ctrcd cohort using the small model (arch S).
Runs all four tokenization variants × 3 seeds.

Tokenization variants (produced by run_tokenization.py):
    Jul30_pan_cancer_ctrcd_v3_all_labs            — all labs, no bucketing
    Jul30_pan_cancer_ctrcd_v3_cardiac_labs        — cardiac-only labs, no bucketing
    Jul30_pan_cancer_ctrcd_v3_bucketed_all_labs   — all labs, bucketed values
    Jul30_pan_cancer_ctrcd_v3_bucketed_cardiac_labs — cardiac-only labs, bucketed values

Architecture S (safest prior for small N):
    d_model=64, num_heads=4, num_layers=1, ff_dim=128  (~940K params)

Hyperparameters fixed from Jul24 sweep findings:
    lr=1e-4, weight_decay=5e-2, dropout=0.4, label_smoothing=0.1

Run:
    python run_train.py
"""

from pathlib import Path

from configs import TrainConfig
import model_src.train as train_module

# ── output root ───────────────────────────────────────────────────────────────
OUT_ROOT = Path("experiment_outputs/pan_cancer_ctrcd/small_model")

# ── small model (arch S) ──────────────────────────────────────────────────────
_ARCH_S = dict(
    d_model    = 64,
    num_heads  = 4,
    num_layers = 1,
    ff_dim     = 128,
)

# ── fixed hyperparameters (from Jul24 sweep) ──────────────────────────────────
_BASE = dict(
    **_ARCH_S,
    epochs          = 100,
    batch_size      = 16,
    lr              = 1e-4,
    weight_decay    = 5e-2,
    dropout         = 0.4,
    label_smoothing = 0.1,
    fusion          = "add",
    use_time        = False,
    use_age         = False,
    device          = "auto",
    num_workers     = 2,
    use_wandb       = False,
)

SEEDS = [42, 52, 62]

# ── tokenization dataset variants ─────────────────────────────────────────────
DATASETS = [
    ("all_labs",              "Jul30_pan_cancer_ctrcd_v3_all_labs"),
    ("cardiac_labs",          "Jul30_pan_cancer_ctrcd_v3_cardiac_labs"),
    ("bucketed_all_labs",     "Jul30_pan_cancer_ctrcd_v3_bucketed_all_labs"),
    ("bucketed_cardiac_labs", "Jul30_pan_cancer_ctrcd_v3_bucketed_cardiac_labs"),
]

# ── build run list ────────────────────────────────────────────────────────────
RUNS = [
    TrainConfig(
        **_BASE,
        data_dir   = Path("tokenization_outputs") / tok_dir,
        seed       = s,
        output_dir = OUT_ROOT / dataset_id / f"seed{s}",
        run_name   = f"small-{dataset_id}-seed{s}",
    )
    for dataset_id, tok_dir in DATASETS
    for s in SEEDS
]

# ── run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"Total runs : {len(RUNS)}")
    print(f"  datasets : {len(DATASETS)}")
    print(f"  seeds    : {SEEDS}")
    print(f"  model    : arch S  (d_model=64, num_heads=4, num_layers=1, ff_dim=128)")
    print(f"Output root: {OUT_ROOT}\n")

    for i, cfg in enumerate(RUNS, 1):
        print(f"\n{'=' * 55}")
        print(f"  Run {i}/{len(RUNS)}  →  {cfg.output_dir}")
        print(f"  data   : {cfg.data_dir.name}")
        print(f"  seed   : {cfg.seed}")
        print(f"{'=' * 55}")
        cfg.save(cfg.output_dir / "config.json")
        train_module.train(cfg)
