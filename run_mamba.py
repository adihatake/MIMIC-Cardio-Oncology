"""
run_mamba.py

The file you edit to run EHR_Mamba experiments. Works the same way as
run_train.py — define configs and run them — but defaults to model_type="mamba".

Single run:
    python run_mamba.py

Requires CUDA + mamba-ssm (install once on your cluster before running):
    pip install causal-conv1d mamba-ssm

─── Mamba hyperparameter guide ──────────────────────────────────────────────────
  d_model       Embedding + hidden dim. 128 with Mamba is roughly comparable to
                256 with a Transformer for EHR sequence tasks.
  num_layers    Number of BiMamba blocks. 4–6 is a good starting range.
  d_state       SSM latent state size N. Default 16. Larger → more capacity
                but more memory and slightly slower.
  d_conv        Depthwise Conv1d kernel width (local context before the SSM). Default 4.
  d_expand      d_inner = d_expand × d_model. Default 2.
  bidirectional True  → BiMambaBlock: two Mamba instances (fwd + bwd) giving
                       full left- and right-context at every position. Recommended
                       for EHR encoding (equivalent of BiLSTM / BERT encoder).
                False → causal MambaBlock: left-context only (GPT-style).
─────────────────────────────────────────────────────────────────────────────────

─── Embedding ablation flags ────────────────────────────────────────────────────
  Same as run_train.py — fusion, use_time, use_age are independent of model_type.

  Ablation grid:
    A0  add,    use_time=F, use_age=F  — baseline (no temporal info)
    A1  add,    use_time=T, use_age=F  — + relative time gaps
    A2  add,    use_time=F, use_age=T  — + patient age
    A3  add,    use_time=T, use_age=T  — + time + age
    B0  concat, use_time=F, use_age=F  — concat fusion only
    B1  concat, use_time=T, use_age=F  — concat + time
    B2  concat, use_time=T, use_age=T  — concat + time + age
─────────────────────────────────────────────────────────────────────────────────
"""

from pathlib import Path

from configs import TrainConfig
import model_src.mamba_train as train_module

# ── shared hyperparameters ────────────────────────────────────────────────────
_BASE = dict(
    model_type      = "mamba",
    data_dir        = Path("tokenization_outputs/Jul1_512"),
    epochs          = 100,
    batch_size      = 16,
    lr              = 2e-4,     # Mamba generally trains well with a slightly higher LR
    weight_decay    = 1e-2,
    label_smoothing = 0.1,
    # Mamba architecture
    d_model         = 128,      # parameter-efficient; ~comparable to 256-dim Transformer
    num_layers      = 4,
    d_state         = 16,       # SSM state size
    d_conv          = 4,        # depthwise conv kernel width
    d_expand        = 2,        # d_inner = 2 × d_model
    bidirectional   = True,     # bidirectional SSM scan
    dropout         = 0.1,
    device          = "auto",
    num_workers     = 2,
    use_wandb       = False,
)

# ── active run: A3 (add + time + age) ────────────────────────────────────────
# Starting point recommended for Mamba: additive fusion with explicit time-gap
# and age embeddings. use_time matters more for SSMs than Transformers because
# the recurrence treats all steps as equally spaced without it.
#
# Compare results with:
#   python evaluation/compare_ablations.py experiment_outputs/Jul3/mamba_A3/ --sort auroc

SEEDS = [42, 43, 44, 45, 46]
RUNS  = []

for s in SEEDS:
    RUNS.append(TrainConfig(
        **_BASE,
        fusion     = "add",
        use_time   = True,
        use_age    = True,
        data_dir   = Path("tokenization_outputs/Jul1_512"),
        seed       = s,
        output_dir = Path(f"experiment_outputs/Jul3/mamba_A3/seed{s}"),
        run_name   = f"mamba-A3-seed{s}",
    ))

# ── full embedding ablation sweep (uncomment to run all variants) ─────────────
# Run after A3 to see whether other embedding combinations do better or worse.
# Replace <CHOSEN_TOKENIZATION> with the tokenization that worked best for the
# Transformer (either "Jul1_512" or "Jul1_512_bucketed_all").
#
# ABLATIONS = [
#     ("A0", dict(fusion="add",    use_time=False, use_age=False)),  # baseline
#     ("A1", dict(fusion="add",    use_time=True,  use_age=False)),
#     ("A2", dict(fusion="add",    use_time=False, use_age=True)),
#     ("A3", dict(fusion="add",    use_time=True,  use_age=True)),   # ← recommended start
#     ("B0", dict(fusion="concat", use_time=False, use_age=False)),
#     ("B1", dict(fusion="concat", use_time=True,  use_age=False)),
#     ("B2", dict(fusion="concat", use_time=True,  use_age=True)),
# ]
# for ablation_id, emb_kwargs in ABLATIONS:
#     for s in SEEDS:
#         RUNS.append(TrainConfig(
#             **_BASE,
#             **emb_kwargs,
#             data_dir   = Path("tokenization_outputs/<CHOSEN_TOKENIZATION>"),
#             seed       = s,
#             output_dir = Path(f"experiment_outputs/Jul3/mamba_ablations/{ablation_id}/seed{s}"),
#             run_name   = f"mamba-{ablation_id}-seed{s}",
#         ))

# ── run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    for i, cfg in enumerate(RUNS, 1):
        print(f"\n{'=' * 55}")
        print(f"  Run {i}/{len(RUNS)}  →  {cfg.output_dir}")
        print(f"{'=' * 55}")
        cfg.save(cfg.output_dir / "config.json")
        train_module.train(cfg)
