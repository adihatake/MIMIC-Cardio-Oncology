"""
run_train.py

The file you edit constantly. Define one or more TrainConfigs and run them.

Single run:
    python run_train.py

─── Sweep structure ────────────────────────────────────────────────────────────
Four independent axes, all using A0 embedding (add, no time, no age):

  experiment_outputs/July23/
    arch_sweep/    S / M / L        ×  seeds   (which model size?)
    lr_sweep/      LR1–LR5          ×  seeds   (learning rate)
    wd_sweep/      WD1–WD5          ×  seeds   (weight decay / L2)
    dropout_sweep/ D1–D5            ×  seeds   (dropout)

Hyperparameter sweeps (lr, wd, dropout) fix architecture M and A0 embedding.
One factor at a time — once you know the best value for each axis, combine them.

─── Seeds ───────────────────────────────────────────────────────────────────────
Each seed independently controls:
  1. Patient split  — which patients land in train/val/test
  2. Model init     — weight initialisation and dropout randomness
────────────────────────────────────────────────────────────────────────────────
"""

from pathlib import Path

from configs import TrainConfig
import model_src.train as train_module

# ── output root ───────────────────────────────────────────────────────────────
OUT_ROOT = Path("experiment_outputs/July23")

# ── shared base ───────────────────────────────────────────────────────────────
# A0 embedding throughout: simplest config, no temporal/age info.
_BASE = dict(
    epochs          = 100,
    batch_size      = 16,
    device          = "auto",
    num_workers     = 2,
    use_wandb       = False,
    data_dir        = Path("tokenization_outputs/Jul17_512_all_labs"),
    # A0 embedding
    fusion          = "add",
    use_time        = False,
    use_age         = False,
)

SEEDS = [42, 52, 62, 72, 82]

# ── architecture S fixed for all hyperparameter sweeps ────────────────────────
# S is the safer prior for ~1800 training samples (~940K params vs ~3.7M for M).
# Optimal lr/wd/dropout values found here transfer well to M if you later scale up.
_ARCH_SWEEP = dict(d_model=64, num_heads=4, num_layers=1, ff_dim=128)

# ── default hyperparameters fixed while sweeping other axes ───────────────────
_DEFAULT_LR  = 1e-4
_DEFAULT_WD  = 5e-2
_DEFAULT_DO  = 0.3
_DEFAULT_LS  = 0.1

# ── 1. Architecture sweep ─────────────────────────────────────────────────────
# Fix default lr/wd/dropout; vary model size.
ARCH = [
    ("S", dict(d_model=64,  num_heads=4, num_layers=1, ff_dim=128)),
    ("M", dict(d_model=128, num_heads=4, num_layers=2, ff_dim=256)),
    ("L", dict(d_model=128, num_heads=8, num_layers=4, ff_dim=512)),
]

_ARCH_BASE = dict(
    **_BASE,
    lr              = _DEFAULT_LR,
    weight_decay    = _DEFAULT_WD,
    dropout         = _DEFAULT_DO,
    label_smoothing = _DEFAULT_LS,
)

ARCH_RUNS = [
    TrainConfig(
        **_ARCH_BASE,
        **arch_kwargs,
        seed       = s,
        output_dir = OUT_ROOT / "arch_sweep" / arch_id / f"seed{s}",
        run_name   = f"arch-{arch_id}-seed{s}",
    )
    for arch_id, arch_kwargs in ARCH
    for s in SEEDS
]

# ── 2. Learning rate sweep ────────────────────────────────────────────────────
# Fix arch M, default wd/dropout; vary lr.
LR_SWEEP = [
    ("LR1", dict(lr=5e-5)),
    ("LR2", dict(lr=1e-4)),
    ("LR3", dict(lr=2e-4)),
    ("LR4", dict(lr=5e-4)),
    ("LR5", dict(lr=1e-3)),
]

_LR_BASE = dict(
    **_BASE,
    **_ARCH_SWEEP,
    weight_decay    = _DEFAULT_WD,
    dropout         = _DEFAULT_DO,
    label_smoothing = _DEFAULT_LS,
)

LR_RUNS = [
    TrainConfig(
        **_LR_BASE,
        **lr_kwargs,
        seed       = s,
        output_dir = OUT_ROOT / "lr_sweep" / lr_id / f"seed{s}",
        run_name   = f"lr-{lr_id}-seed{s}",
    )
    for lr_id, lr_kwargs in LR_SWEEP
    for s in SEEDS
]

# ── 3. Weight decay sweep ─────────────────────────────────────────────────────
# Fix arch M, default lr/dropout; vary weight decay.
WD_SWEEP = [
    ("WD1", dict(weight_decay=0.0)),
    ("WD2", dict(weight_decay=1e-2)),
    ("WD3", dict(weight_decay=5e-2)),
    ("WD4", dict(weight_decay=1e-1)),
    ("WD5", dict(weight_decay=2e-1)),
]

_WD_BASE = dict(
    **_BASE,
    **_ARCH_SWEEP,
    lr              = _DEFAULT_LR,
    dropout         = _DEFAULT_DO,
    label_smoothing = _DEFAULT_LS,
)

WD_RUNS = [
    TrainConfig(
        **_WD_BASE,
        **wd_kwargs,
        seed       = s,
        output_dir = OUT_ROOT / "wd_sweep" / wd_id / f"seed{s}",
        run_name   = f"wd-{wd_id}-seed{s}",
    )
    for wd_id, wd_kwargs in WD_SWEEP
    for s in SEEDS
]

# ── 4. Dropout sweep ──────────────────────────────────────────────────────────
# Fix arch M, default lr/wd; vary dropout.
DROPOUT_SWEEP = [
    ("D1", dict(dropout=0.1)),
    ("D2", dict(dropout=0.2)),
    ("D3", dict(dropout=0.3)),
    ("D4", dict(dropout=0.4)),
    ("D5", dict(dropout=0.5)),
]

_DROPOUT_BASE = dict(
    **_BASE,
    **_ARCH_SWEEP,
    lr              = _DEFAULT_LR,
    weight_decay    = _DEFAULT_WD,
    label_smoothing = _DEFAULT_LS,
)

DROPOUT_RUNS = [
    TrainConfig(
        **_DROPOUT_BASE,
        **d_kwargs,
        seed       = s,
        output_dir = OUT_ROOT / "dropout_sweep" / d_id / f"seed{s}",
        run_name   = f"dropout-{d_id}-seed{s}",
    )
    for d_id, d_kwargs in DROPOUT_SWEEP
    for s in SEEDS
]

# ── combined run list ─────────────────────────────────────────────────────────
# Ordered so arch runs first — pick the best size before the hyperparameter sweeps.
RUNS = ARCH_RUNS + LR_RUNS + WD_RUNS + DROPOUT_RUNS

# ── run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Total runs : {len(RUNS)}")
    print(f"  arch     : {len(ARCH_RUNS)}  ({len(ARCH)} variants × {len(SEEDS)} seeds)")
    print(f"  lr       : {len(LR_RUNS)}  ({len(LR_SWEEP)} variants × {len(SEEDS)} seeds)")
    print(f"  wd       : {len(WD_RUNS)}  ({len(WD_SWEEP)} variants × {len(SEEDS)} seeds)")
    print(f"  dropout  : {len(DROPOUT_RUNS)}  ({len(DROPOUT_SWEEP)} variants × {len(SEEDS)} seeds)")
    print(f"Output root: {OUT_ROOT}\n")

    for i, cfg in enumerate(RUNS, 1):
        print(f"\n{'=' * 55}")
        print(f"  Run {i}/{len(RUNS)}  →  {cfg.output_dir}")
        print(f"{'=' * 55}")
        cfg.save(cfg.output_dir / "config.json")
        train_module.train(cfg)
