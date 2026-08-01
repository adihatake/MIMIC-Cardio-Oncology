"""
run_train.py

Architecture sweep for the pan_cancer_uniform_365 cohort.
Runs arch S and arch M × 4 dataset variants × 5 seeds.

Tokenization variants (produced by run_tokenization.py):
    Jul31_pan_cancer_uniform_365_v1_all_labs
    Jul31_pan_cancer_uniform_365_v1_cardiac_labs
    Jul31_pan_cancer_uniform_365_v1_bucketed_all_labs
    Jul31_pan_cancer_uniform_365_v1_bucketed_cardiac_labs

Architecture S (baseline):
    d_model=64,  num_heads=4, num_layers=1, ff_dim=128   (~940K params)

Architecture M (capacity upgrade):
    d_model=128, num_heads=4, num_layers=2, ff_dim=256   (~3.6M params)

Rationale for arch M: arch S plateaus at AUROC ~0.63 with high seed variance,
consistent with underfitting on 415-token average sequences. Arch M doubles
d_model and adds a second layer to capture longer-range temporal dependencies.

Hyperparameters fixed:
    lr=1e-4, weight_decay=5e-2, dropout=0.3, label_smoothing=0.1

Run:
    python run_train.py
"""

from pathlib import Path

from configs import TrainConfig
import model_src.train as train_module

# ── output root ───────────────────────────────────────────────────────────────
OUT_ROOT = Path("experiment_outputs/July31/pan_cancer_uniform_365/")

# ── architectures ─────────────────────────────────────────────────────────────
_ARCH_S = dict(d_model=64,  num_heads=4, num_layers=1, ff_dim=128)
_ARCH_M = dict(d_model=128, num_heads=4, num_layers=2, ff_dim=256)

# ── fixed hyperparameters ─────────────────────────────────────────────────────
_BASE = dict(
    epochs          = 200,
    batch_size      = 16,
    lr              = 1e-4,
    weight_decay    = 5e-2,
    dropout         = 0.5,
    label_smoothing = 0.1,
    fusion          = "add",
    use_time        = False,
    use_age         = False,
    device          = "auto",
    num_workers     = 4,
    use_wandb       = False,
)

SEEDS = [42, 52, 62, 72, 82]

# ── dataset variants ──────────────────────────────────────────────────────────
DATASETS = [
    ("all_labs",              "Jul31_pan_cancer_uniform_365_v1_all_labs"),
    ("cardiac_labs",          "Jul31_pan_cancer_uniform_365_v1_cardiac_labs"),
    ("bucketed_all_labs",     "Jul31_pan_cancer_uniform_365_v1_bucketed_all_labs"),
    ("bucketed_cardiac_labs", "Jul31_pan_cancer_uniform_365_v1_bucketed_cardiac_labs"),
]

# ── build run list ─────────────────────────────────────────────────────────────
ARCHS = [
    ("arch_s", _ARCH_S),
    ("arch_m", _ARCH_M),
]

RUNS = [
    TrainConfig(
        **_BASE,
        **arch_kwargs,
        data_dir   = Path("tokenization_outputs") / tok_dir,
        seed       = s,
        output_dir = OUT_ROOT / arch_name / dataset_id / f"seed{s}",
        run_name   = f"{arch_name}-{dataset_id}-seed{s}",
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
        print(f"{'=' * 60}")
        cfg.save(cfg.output_dir / "config.json")
        train_module.train(cfg)
