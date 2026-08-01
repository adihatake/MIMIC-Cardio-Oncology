"""
run_train.py

Threshold sweep for arch_s / all_labs (pan_cancer_uniform_365 cohort).

Trains the same arch_s model 5 seeds × N thresholds, varying only the
eval_threshold used to compute sensitivity/specificity/F1 in history.json.
This lets you compare training curves at different decision thresholds using
plot_history.py, with each threshold as a separate variant directory.

Output layout:
    experiment_outputs/threshold_sweep/arch_s/all_labs/
        thresh_0.20/seed42/  seed52/  ...
        thresh_0.30/seed42/  ...
        thresh_0.40/seed42/  ...
        thresh_0.50/seed42/  ...

Plot the comparison:
    python evaluation/plot_history.py \\
        --model-dir experiment_outputs/threshold_sweep/arch_s/all_labs/thresh_0.20 \\
                    experiment_outputs/threshold_sweep/arch_s/all_labs/thresh_0.30 \\
                    experiment_outputs/threshold_sweep/arch_s/all_labs/thresh_0.40 \\
                    experiment_outputs/threshold_sweep/arch_s/all_labs/thresh_0.50 \\
        --metrics loss auroc auprc f1 sensitivity \\
        --save experiment_outputs/threshold_sweep/arch_s/all_labs/threshold_comparison.png

Run:
    python run_train.py
"""

from pathlib import Path

from configs import TrainConfig
import model_src.train as train_module

# ── output root ───────────────────────────────────────────────────────────────
OUT_ROOT = Path("experiment_outputs/threshold_sweep/arch_s/all_labs")

# ── fixed: arch_s / all_labs ──────────────────────────────────────────────────
_ARCH_S  = dict(d_model=64, num_heads=4, num_layers=1, ff_dim=128)
TOK_DIR  = Path("tokenization_outputs/Jul31_pan_cancer_uniform_365_v1_all_labs")

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

# ── sweep axes ────────────────────────────────────────────────────────────────
SEEDS      = [42, 52, 62, 72, 82]
THRESHOLDS = [0.20, 0.30, 0.40, 0.50]

# ── build run list ─────────────────────────────────────────────────────────────
RUNS = [
    TrainConfig(
        **_BASE,
        **_ARCH_S,
        data_dir       = TOK_DIR,
        seed           = s,
        eval_threshold = t,
        output_dir     = OUT_ROOT / f"thresh_{t:.2f}" / f"seed{s}",
        run_name       = f"arch_s-all_labs-thresh{t:.2f}-seed{s}",
    )
    for t in THRESHOLDS
    for s in SEEDS
]

# ── run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"Total runs : {len(RUNS)}")
    print(f"  thresholds : {THRESHOLDS}")
    print(f"  seeds      : {SEEDS}")
    print(f"Output root  : {OUT_ROOT}\n")

    for i, cfg in enumerate(RUNS, 1):
        print(f"\n{'=' * 60}")
        print(f"  Run {i}/{len(RUNS)}  →  {cfg.output_dir}")
        print(f"  eval_threshold : {cfg.eval_threshold}")
        print(f"  seed           : {cfg.seed}")
        print(f"{'=' * 60}")
        cfg.save(cfg.output_dir / "config.json")
        train_module.train(cfg)
