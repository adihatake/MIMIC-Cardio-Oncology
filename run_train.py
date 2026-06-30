"""
run_train.py

The file you edit constantly. Define one or more TrainConfigs and run them.

Single run:
    python run_train.py

Multiple runs (grid search, ablations) are defined in the RUNS list below.

─── Time-embedding ablation ────────────────────────────────────────────────────
CEHR-BERT uses two mechanisms for temporal information:
  1. TimeEmbeddingLayer: sin((days_since_2000 / 365.25) × w + φ) where w, φ are
     learned — added to the token embedding sum at each position.
  2. Artificial Time Tokens (ATT): discrete vocabulary tokens (W0–W3, M1–M11, LT)
     inserted between consecutive visits during tokenization.

To run the ablation:
  Step 1 — Regenerate tokenization with dates.pt (required for time embedding):
      python run_tokenization.py   # or tokenize_cli.py with your data-dir

  Step 2 — Run both configs below.  The two output_dirs give side-by-side results.
────────────────────────────────────────────────────────────────────────────────
"""

from pathlib import Path

from configs import TrainConfig
import model_src.train as train_module

# ── shared hyperparameters ────────────────────────────────────────────────────
_BASE = dict(
    data_dir    = Path("tokenization_outputs/ver1"),
    epochs      = 1,
    batch_size  = 8,
    lr          = 1e-4,
    d_model     = 768,
    num_heads   = 12,
    num_layers  = 12,
    ff_dim      = 3072,
    dropout     = 0.1,
    device      = "auto",
    seed        = 42,
    num_workers = 6,
    use_wandb   = False,
    wandb_project = "mimic-cardio-oncology",
)

# ── define runs ───────────────────────────────────────────────────────────────

RUNS = [
    # ── Baseline: no time embedding (sequential position only) ────────────────
    TrainConfig(
        **_BASE,
        output_dir         = Path("experiment_outputs/ablation_no_time_emb"),
        use_time_embedding = False,
        run_name           = "baseline-no-time-emb",
    ),

    # ── CEHR-BERT time embedding ablation ─────────────────────────────────────
    # Adds sin((days_since_2000 / 365.25) × w + φ) per token.
    # Requires dates.pt — re-run tokenization first if it is missing.
    TrainConfig(
        **_BASE,
        output_dir         = Path("experiment_outputs/ablation_time_emb"),
        use_time_embedding = True,
        run_name           = "cehrbert-time-emb",
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
