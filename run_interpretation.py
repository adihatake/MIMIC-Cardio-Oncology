"""
run_interpretation.py

Define samples to explain in RUNS and execute them all with:
    python run_interpretation.py

Each entry produces an independent output folder under
interpretation/outputs/ containing attention heatmaps, rollout and IG
bar charts, and attributions.csv.  The visualizer then adds three
summary plots to the same folder.

─── Sample selection ────────────────────────────────────────────────────────
Each InterpretationConfig must identify exactly one sample via either:

    sample_idx=42                              — row index in the dataset
    subject_id=12345, cycle_number=2           — patient + cycle lookup

─── IG baseline ─────────────────────────────────────────────────────────────
The Integrated Gradients baseline is always a zero-embedding sequence
(see interpretation/interpret.py).  Increase ig_steps for more precise
attribution at the cost of compute time (convergence delta < 0.01 is good).

─── Skipping IG ─────────────────────────────────────────────────────────────
Set skip_ig=True in a config (or globally in _BASE) if captum is not
installed.  Attention heatmaps, rollout, and the visualizer will still run.
Install captum with:  pip install captum
────────────────────────────────────────────────────────────────────────────
"""

from pathlib import Path

from configs import InterpretationConfig
import interpretation.interpret as interp_module
import interpretation.visualize_attributions as viz_module

REPO_ROOT = Path(__file__).resolve().parent

# ── shared settings ───────────────────────────────────────────────────────────
_BASE = dict(
    model_dir     = REPO_ROOT / "experiment_outputs" / "run1",
    data_dir      = REPO_ROOT / "tokenization_outputs" / "Jul17_512_all_labs",
    ig_steps      = 100,
    top_k         = 30,
    skip_ig       = False,
    run_visualize = True,
    device        = "auto",
)

# ── samples to explain ────────────────────────────────────────────────────────
# Add entries here — one per patient+cycle you want to inspect.
# Use subject_id + cycle_number for named outputs, or sample_idx for quick runs.

RUNS: list[InterpretationConfig] = [
    # Example: explain by subject_id and cycle_number
    # InterpretationConfig(**_BASE, subject_id=10006008, cycle_number=1),
    # InterpretationConfig(**_BASE, subject_id=10006008, cycle_number=2),

    # Example: explain by dataset row index
    # InterpretationConfig(**_BASE, sample_idx=0),
    # InterpretationConfig(**_BASE, sample_idx=1),
]

# ── run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if not RUNS:
        print("RUNS is empty — add InterpretationConfig entries to run_interpretation.py")
        raise SystemExit(0)

    for i, cfg in enumerate(RUNS, 1):
        cfg.validate()
        output_dir = cfg.resolved_output_dir

        print(f"\n{'=' * 55}")
        print(f"  Interpretation {i}/{len(RUNS)}  →  {output_dir.name}")
        print(f"    model_dir    : {cfg.model_dir}")
        print(f"    data_dir     : {cfg.data_dir}")
        print(f"    sample       : {cfg.label}")
        print(f"    ig_steps     : {cfg.ig_steps}  |  skip_ig : {cfg.skip_ig}")
        print(f"    top_k        : {cfg.top_k}  |  run_visualize: {cfg.run_visualize}")
        print(f"{'=' * 55}\n")

        print("── interpret ───────────────────────────────────────────────────────")
        interp_module.explain(
            model_dir         = cfg.model_dir,
            data_dir          = cfg.data_dir,
            sample_idx        = cfg.sample_idx,
            subject_id        = cfg.subject_id,
            cycle_number      = cfg.cycle_number,
            output_dir        = output_dir,
            ig_steps          = cfg.ig_steps,
            top_k             = cfg.top_k,
            device            = cfg.device,
            skip_ig           = cfg.skip_ig,
            checkpoint_metric = cfg.checkpoint_metric,
        )

        if cfg.run_visualize:
            print("\n── visualize ───────────────────────────────────────────────────────")
            viz_module.visualize(
                csv_path     = output_dir / "attributions.csv",
                output_dir   = output_dir,
                top_k        = cfg.top_k,
                title_prefix = cfg.label,
            )
