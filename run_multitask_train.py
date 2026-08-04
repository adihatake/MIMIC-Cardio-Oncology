"""
run_multitask_train.py

Architecture sweep for multi-task cardiotoxicity prediction at 90 / 180 / 365 days.

Before running this script, produce the multi-task tokenization:

    python tokenization_src/tokenize_cli.py \\
        --cohort pan_cancer_uniform_365_v1 \\
        --name <output_name> \\
        --multitask \\
        --all

Then point DATASETS below to those tokenization output directories.

Output layout:
    experiment_outputs/multitask/
        arch_s_plus_wide/
            all_labs/seed42/  seed52/  ...
            cardiac_labs/...
        arch_s_plus_deep/
            ...

Run:
    python run_multitask_train.py
"""

from pathlib import Path

from configs import TrainConfig
import model_src.multitask_train as train_module

# ── output root ───────────────────────────────────────────────────────────────
OUT_ROOT = Path("experiment_outputs/multitask/arch_s_plus")

# ── architectures ─────────────────────────────────────────────────────────────
_ARCH_S_PLUS_WIDE = dict(d_model=96, num_heads=4, num_layers=1, ff_dim=192)
_ARCH_S_PLUS_DEEP = dict(d_model=64, num_heads=4, num_layers=2, ff_dim=256)

ARCHS = [
    ("arch_s_plus_wide", _ARCH_S_PLUS_WIDE),
    ("arch_s_plus_deep", _ARCH_S_PLUS_DEEP),
]

# ── fixed hyperparameters ─────────────────────────────────────────────────────
_BASE = dict(
    epochs          = 200,
    batch_size      = 64,
    lr              = 1e-4,
    weight_decay    = 5e-2,
    dropout         = 0.5,
    label_smoothing = 0.1,
    eval_threshold  = 0.5,
    fusion          = "add",
    use_time        = False,
    use_age         = False,
    device          = "auto",
    num_workers     = 0,
    use_wandb       = False,
    num_tasks       = 3,
)

# ── sweep axes ────────────────────────────────────────────────────────────────
SEEDS = [42, 52, 62, 72, 82]

# Point these to tokenization outputs produced with --multitask.
# Replace the directory names below with your actual multitask tokenization names.
DATASETS = [
    ("all_labs",     "mt_pan_cancer_uniform_365_v1_all_labs"),
    ("cardiac_labs", "mt_pan_cancer_uniform_365_v1_cardiac_labs"),
]

# ── build run list ─────────────────────────────────────────────────────────────
RUNS = [
    TrainConfig(
        **_BASE,
        **arch_kwargs,
        data_dir   = Path("tokenization_outputs") / tok_dir,
        seed       = s,
        output_dir = OUT_ROOT / arch_name / dataset_id / f"seed{s}",
        run_name   = f"mt-{arch_name}-{dataset_id}-seed{s}",
    )
    for arch_name, arch_kwargs in ARCHS
    for dataset_id, tok_dir in DATASETS
    for s in SEEDS
]

# ── run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"Total runs : {len(RUNS)}")
    print(f"  archs    : {[a for a, _ in ARCHS]}")
    print(f"  datasets : {len(DATASETS)}")
    print(f"  seeds    : {SEEDS}")
    print(f"Output root: {OUT_ROOT}\n")

    for i, cfg in enumerate(RUNS, 1):
        print(f"\n{'=' * 60}")
        print(f"  Run {i}/{len(RUNS)}  →  {cfg.output_dir}")
        print(f"  data   : {cfg.data_dir.name}")
        print(f"  seed   : {cfg.seed}")
        print(f"  arch   : d_model={cfg.d_model}, heads={cfg.num_heads}, layers={cfg.num_layers}, ff={cfg.ff_dim}")
        print(f"  tasks  : {cfg.num_tasks}")
        print(f"{'=' * 60}")
        cfg.save(cfg.output_dir / "config.json")
        train_module.train(cfg)
