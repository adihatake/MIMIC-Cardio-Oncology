"""
run_xai.py

Runner for population and patient-level xAI visualizations.
Edit the configuration block below and run:

    python run_xai.py

─── Outputs ──────────────────────────────────────────────────────────────────
Dataset-level   (OUTPUT_DIR/dataset/):
  cls_space_by_label.png      CLS embedding space coloured by true label
  cls_space_by_prob.png       CLS embedding space coloured by P(cardiotoxic)
  cls_space_by_cycle.png      CLS embedding space coloured by cycle number
  cls_coords_2d.npy           Raw 2-D coordinates (reusable)
  population_ig.png           Aggregate signed IG with ±1 SEM
  population_ig_cache.csv     Cached per-feature IG scores

Per-patient     (OUTPUT_DIR/patient_{ID}/):
  cls_trajectory.png          CLS trajectory across cycles
  rollout_per_cycle.png       Side-by-side rollout bars
  rollout_heatmap.png         Feature × cycle rollout heatmap
  perturbation_trajectory.png CLS shift under token edits (if PERTURBATION_SPECS set)
  perturbation_delta.png      ΔP(cardiotoxic) per perturbation per cycle
─────────────────────────────────────────────────────────────────────────────
"""

from pathlib import Path
import torch

# ════════════════════════════════════════════════════════════════════
#  CONFIGURATION — edit these before running
# ════════════════════════════════════════════════════════════════════

REPO_ROOT  = Path(__file__).resolve().parent
MODEL_DIR  = REPO_ROOT / "experiment_outputs" / "run1"
DATA_DIR   = REPO_ROOT / "tokenization_outputs" / "Jul17_512_all_labs"
OUTPUT_DIR = REPO_ROOT / "interpretation" / "xai_outputs"

DEVICE            = "auto"    # "auto" | "cpu" | "cuda" | "mps"
CHECKPOINT_METRIC = "auroc"   # loads best_model_{CHECKPOINT_METRIC}.pt

# ── Dataset-level settings ────────────────────────────────────────
# Which samples appear in population plots (UMAP, aggregate IG).
#   "test"  → test set only  (recommended)
#   "all"   → train + val + test
DATASET_SPLIT = "test"

# Dimensionality reduction method.
#   "umap"  → pip install umap-learn  (best cluster separation)
#   "pca"   → always available, fastest
#   "tsne"  → pip install scikit-learn
DIM_REDUCTION = "umap"

# ── Population IG settings ────────────────────────────────────────
RUN_POPULATION_IG   = True
MAX_IG_SAMPLES      = 50    # randomly sub-sampled from DATASET_SPLIT
IG_STEPS_POPULATION = 30    # fewer steps = faster; 30 is usually sufficient
POP_IG_TOP_K        = 25    # features shown in the population IG plot

# ── Per-patient settings ──────────────────────────────────────────
# Set to a MIMIC subject_id to run all per-patient plots.
# Set to None to skip.
PATIENT_SUBJECT_ID = None   # e.g. 10006008

RUN_PATIENT_IG   = True
IG_STEPS_PATIENT = 100      # more steps = more accurate
ROLLOUT_TOP_K    = 20       # features shown in rollout plots

# ── Perturbation specs ────────────────────────────────────────────
# Each dict describes one token edit to test. Leave as [] to skip.
#
# Types:
#   "zero_feature"  — PAD out all tokens matching feature_pattern prefix
#   "replace_token" — swap from_token → to_token in the vocabulary
#   "remove_visit"  — PAD out all tokens from a visit (by relative offset)
#
# Examples (uncomment to use):
PERTURBATION_SPECS: list[dict] = [
    # {"name": "Remove NTproBNP",
    #  "type": "zero_feature",
    #  "feature_pattern": "lab::50963"},
    #
    # {"name": "NTproBNP Q4 → Q1",
    #  "type": "replace_token",
    #  "from_token": "lab::50963_Q4",
    #  "to_token":   "lab::50963_Q1"},
    #
    # {"name": "Remove last visit",
    #  "type": "remove_visit",
    #  "visit_offset": -1},
]

# ════════════════════════════════════════════════════════════════════
#  END OF CONFIGURATION
# ════════════════════════════════════════════════════════════════════

import sys
sys.path.insert(0, str(REPO_ROOT))
import interpretation.xai as xai


def resolve_device(s: str) -> torch.device:
    if s == "auto":
        return torch.device(
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
    return torch.device(s)


def run_dataset_level(model, cfg, samples_df, tensors, inv_vocab,
                      split_indices, device):
    print(f"\n{'=' * 58}")
    print(f"  Dataset-level  ({DATASET_SPLIT} split)")
    print(f"{'=' * 58}\n")

    ds_out = OUTPUT_DIR / "dataset"
    indices, _ = xai.get_indices(split_indices, DATASET_SPLIT)
    labels_arr = tensors["labels"][indices].numpy()

    print(f"Extracting CLS embeddings for {len(indices)} samples...")
    embeddings = xai.extract_cls_embeddings(model, tensors, indices, device)

    print("Computing prediction probabilities...")
    probs = xai.get_predictions(model, tensors, indices, device)

    coords_2d = xai.plot_cls_embedding_space(
        embeddings   = embeddings,
        labels       = labels_arr,
        probs        = probs,
        samples_df   = samples_df,
        indices      = indices,
        output_dir   = ds_out,
        method       = DIM_REDUCTION,
        split_filter = DATASET_SPLIT,
    )

    if RUN_POPULATION_IG:
        print(f"\nPopulation IG  (max {MAX_IG_SAMPLES} samples, "
              f"{IG_STEPS_POPULATION} steps)...")
        ig_df = xai.compute_population_ig(
            model       = model,
            tensors     = tensors,
            inv_vocab   = inv_vocab,
            indices     = indices,
            device      = device,
            ig_steps    = IG_STEPS_POPULATION,
            max_samples = MAX_IG_SAMPLES,
            cache_path  = ds_out / "population_ig_cache.csv",
        )
        xai.plot_population_ig(
            ig_df        = ig_df,
            output_path  = ds_out / "population_ig.png",
            top_k        = POP_IG_TOP_K,
            split_filter = DATASET_SPLIT,
            n_samples    = min(MAX_IG_SAMPLES, len(indices)),
        )

    return embeddings, coords_2d, labels_arr


def run_patient_level(model, cfg, samples_df, tensors, vocab, inv_vocab,
                      split_indices, device,
                      bg_embeddings, bg_labels):
    if PATIENT_SUBJECT_ID is None:
        print("\n  PATIENT_SUBJECT_ID not set — skipping per-patient analysis")
        return

    print(f"\n{'=' * 58}")
    print(f"  Per-patient  (subject_id={PATIENT_SUBJECT_ID})")
    print(f"{'=' * 58}\n")

    pat_out = OUTPUT_DIR / f"patient_{PATIENT_SUBJECT_ID}"
    pat_out.mkdir(parents=True, exist_ok=True)

    patient_indices = xai.find_patient_indices(samples_df, PATIENT_SUBJECT_ID)
    if not patient_indices:
        print(f"  No samples found for subject_id={PATIENT_SUBJECT_ID}")
        return

    cycles = [int(samples_df.iloc[i]["cycle_number"]) for i in patient_indices]
    print(f"  Cycles found: {cycles}")

    print(f"  Running per-cycle forward passes"
          f"{' + IG' if RUN_PATIENT_IG else ''}...")
    cycle_data = xai.get_patient_cycle_data(
        model           = model,
        tensors         = tensors,
        inv_vocab       = inv_vocab,
        samples_df      = samples_df,
        patient_indices = patient_indices,
        device          = device,
        ig_steps        = IG_STEPS_PATIENT,
        skip_ig         = not RUN_PATIENT_IG,
    )

    patient_cls = torch.stack([cd["cls_embedding"] for cd in cycle_data])

    # Joint projection so patient lives in the same 2-D space as the background
    print(f"  Projecting background + patient jointly ({DIM_REDUCTION.upper()})...")
    bg_coords, pat_coords, _, method_name = xai.project_together(
        background_embeddings = bg_embeddings,
        patient_embeddings    = patient_cls,
        method                = DIM_REDUCTION,
    )

    xai.plot_cls_trajectory(
        cycle_data        = cycle_data,
        patient_coords_2d = pat_coords,
        subject_id        = PATIENT_SUBJECT_ID,
        coords_background = bg_coords,
        background_labels = bg_labels,
        output_path       = pat_out / "cls_trajectory.png",
        method_name       = method_name,
    )

    xai.plot_rollout_per_cycle(
        cycle_data  = cycle_data,
        subject_id  = PATIENT_SUBJECT_ID,
        output_path = pat_out / "rollout_per_cycle.png",
        top_k       = ROLLOUT_TOP_K,
    )

    xai.plot_rollout_heatmap(
        cycle_data  = cycle_data,
        subject_id  = PATIENT_SUBJECT_ID,
        output_path = pat_out / "rollout_heatmap.png",
        top_k       = ROLLOUT_TOP_K,
    )

    if PERTURBATION_SPECS:
        print(f"\n  Running {len(PERTURBATION_SPECS)} perturbation(s)...")
        perturb_results = xai.run_perturbation_analysis(
            model              = model,
            tensors            = tensors,
            inv_vocab          = inv_vocab,
            vocab              = vocab,
            patient_indices    = patient_indices,
            perturbation_specs = PERTURBATION_SPECS,
            device             = device,
        )

        perturb_embs = {
            name: torch.stack([r["cls_emb"] for r in results])
            for name, results in perturb_results.items()
        }

        bg2, pat2, perturb_coords, mname2 = xai.project_together(
            background_embeddings = bg_embeddings,
            patient_embeddings    = patient_cls,
            perturb_embeddings    = perturb_embs,
            method                = DIM_REDUCTION,
        )

        xai.plot_perturbation_trajectory(
            original_cycle_data = cycle_data,
            perturb_results     = perturb_results,
            subject_id          = PATIENT_SUBJECT_ID,
            patient_coords_2d   = pat2,
            perturb_coords_2d   = perturb_coords,
            coords_background   = bg2,
            background_labels   = bg_labels,
            output_path         = pat_out / "perturbation_trajectory.png",
            method_name         = mname2,
        )

        xai.plot_perturbation_delta(
            original_cycle_data = cycle_data,
            perturb_results     = perturb_results,
            subject_id          = PATIENT_SUBJECT_ID,
            output_path         = pat_out / "perturbation_delta.png",
        )
    else:
        print("  No perturbation specs defined — skipping")


if __name__ == "__main__":
    device = resolve_device(DEVICE)
    print(f"Device: {device}")

    if not (MODEL_DIR / "config.json").exists():
        print(f"ERROR: config.json not found in {MODEL_DIR}")
        raise SystemExit(1)
    if not DATA_DIR.exists():
        print(f"ERROR: DATA_DIR not found: {DATA_DIR}")
        raise SystemExit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\nLoading model and data...")
    model, cfg, samples_df, tensors, vocab, inv_vocab, split_indices = xai.load_setup(
        model_dir         = MODEL_DIR,
        data_dir          = DATA_DIR,
        device            = device,
        checkpoint_metric = CHECKPOINT_METRIC,
    )

    bg_embeddings, _, bg_labels = run_dataset_level(
        model, cfg, samples_df, tensors, inv_vocab, split_indices, device
    )

    run_patient_level(
        model, cfg, samples_df, tensors, vocab, inv_vocab,
        split_indices, device, bg_embeddings, bg_labels,
    )

    print(f"\n{'=' * 58}")
    print(f"  All outputs → {OUTPUT_DIR}")
    print(f"{'=' * 58}")
