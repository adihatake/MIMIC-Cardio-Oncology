"""
run_train.py

The file you edit constantly. Define one or more TrainConfigs and run them.

Single run:
    python run_train.py

Multiple runs (grid search, ablations) are defined in the RUNS list below.
"""

from pathlib import Path

from configs import TrainConfig
import model_src.train as train_module

# ── define runs ───────────────────────────────────────────────────────────────
# Add as many configs as you want. Each gets its own output_dir.

RUNS = [
    TrainConfig(
        data_dir   = Path("tokenization_outputs/ver1"),
        output_dir = Path("experiment_outputs/test1"),
        epochs     = 1,
        batch_size = 8,
        lr         = 1e-4,

        d_model    = 768,
        num_heads  = 12,
        num_layers = 12,
        ff_dim     = 3072,
        dropout    = 0.1,

        device      = "auto",
        seed        = 42,
        num_workers = 4,

        # ── W&B tracking ─────────────────────────────────────────────────────
        use_wandb     = False,              # flip to True to enable tracking
        wandb_project = "mimic-cardio-oncology",
        run_name      = None,               # None → W&B auto-generates a name
    ),

    # Example: deeper model
    # TrainConfig(
    #     data_dir   = Path("tokenization_outputs/ver1"),
    #     output_dir = Path("model_outputs/run2_deep"),
    #     d_model    = 256,
    #     num_layers = 8,
    #     ff_dim     = 1024,
    #     use_wandb  = True,
    #     run_name   = "deep-ablation",
    # ),
]

# ── run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    for i, cfg in enumerate(RUNS, 1):
        print(f"\n{'=' * 55}")
        print(f"  Run {i}/{len(RUNS)}  →  {cfg.output_dir}")
        print(f"{'=' * 55}")
        cfg.save(cfg.output_dir / "config.json")
        train_module.train(cfg)
