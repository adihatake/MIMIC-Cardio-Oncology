"""
run_xai.py

Runner for population and patient-level xAI visualizations.
Edit the CONFIGURATION block below and run:

    python run_xai.py

─── Toggles ──────────────────────────────────────────────────────────────────
RUN_DATASET_PLOTS       CLS embedding space (all colour schemes) + population IG
RUN_PATIENT_DEEP_DIVE   Per-patient trajectory, rollout, perturbation

Patients listed in PATIENT_SUBJECT_IDS are:
  • overlaid as ★ markers on every dataset CLS plot (SHOW_PATIENTS_ON_DATASET_PLOTS)
  • given a full per-patient deep-dive (RUN_PATIENT_DEEP_DIVE)
Set to [] to skip all patient analysis.

─── Outputs ──────────────────────────────────────────────────────────────────
Dataset-level   (OUTPUT_DIR/dataset/):
  cls_space_by_label.png
  cls_space_by_prob.png
  cls_space_by_cycle.png
  cls_space_by_drug_class.png          (if drug info available)
  cls_space_by_primary_regimen.png     (if drug info available)
  cls_space_by_rx_count.png            (if drug info available)
  cls_coords_2d.npy
  population_ig.png
  population_ig_cache.csv

Per-patient     (OUTPUT_DIR/patient_{ID}/):
  cls_trajectory.png
  rollout_per_cycle.png
  rollout_heatmap.png
  perturbation_trajectory.png          (if PERTURBATION_SPECS set)
  perturbation_delta.png               (if PERTURBATION_SPECS set)
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

# ── What to run ───────────────────────────────────────────────────
RUN_DATASET_PLOTS    = True
RUN_PATIENT_DEEP_DIVE = True

# ── Patients ──────────────────────────────────────────────────────
# All listed patients are overlaid on dataset CLS plots and/or given
# a per-patient deep-dive, depending on the toggles above.
# Set to [] to skip all patient analysis.
PATIENT_SUBJECT_IDS: list[int] = [19169531, 18865367, 11885477, 13082017]

# Overlay patients as ★ markers on every dataset CLS space plot.
# Requires RUN_DATASET_PLOTS = True.
SHOW_PATIENTS_ON_DATASET_PLOTS = True

# ── Dataset-level settings ────────────────────────────────────────
DATASET_SPLIT = "test"   # "test" | "val" | "train" | "all"
DIM_REDUCTION = "umap"   # "umap" | "pca" | "tsne"

# ── Drug info enrichment ──────────────────────────────────────────
# Adds drug class / dose-proxy columns, enabling extra CLS space plots.
# Set to None to auto-discover via metadata.json (recommended).
# Set ENRICH_DRUG_INFO = False to skip entirely.
ENRICH_DRUG_INFO  = True
COHORT_TABLE_PATH = None  # e.g. Path("cohort_outputs/cycle_modeling_v4/final_cycle_binary_modeling_table.parquet")

# ── Population IG settings ────────────────────────────────────────
RUN_POPULATION_IG   = True
MAX_IG_SAMPLES      = 50    # randomly sub-sampled from DATASET_SPLIT
IG_STEPS_POPULATION = 30    # fewer steps = faster; 30 is usually sufficient
POP_IG_TOP_K        = 25    # features shown in the population IG plot

# ── Per-patient deep-dive settings ───────────────────────────────
RUN_PATIENT_IG   = True
IG_STEPS_PATIENT = 100     # more steps = more accurate
ROLLOUT_TOP_K    = 20      # features shown in rollout plots

# ── Perturbation specs ────────────────────────────────────────────
# Each dict describes one token edit. Leave as [] to skip.
# Applied to every patient listed in PATIENT_SUBJECT_IDS.
#
# Types:
#   "zero_feature"  — PAD all tokens matching feature_pattern prefix
#   "replace_token" — swap from_token → to_token in the vocabulary
#   "remove_visit"  — PAD all tokens from a visit (by relative offset)
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


def run_dataset_level(model, samples_df, tensors, inv_vocab, split_indices, device,
                      patient_overlay: dict[int, torch.Tensor]) -> tuple:
    print(f"\n{'=' * 58}")
    print(f"  Dataset-level  ({DATASET_SPLIT} split)")
    print(f"{'=' * 58}\n")

    ds_out     = OUTPUT_DIR / "dataset"
    indices, _ = xai.get_indices(split_indices, DATASET_SPLIT)
    labels_arr = tensors["labels"][indices].numpy()

    print(f"Extracting CLS embeddings for {len(indices)} samples...")
    embeddings = xai.extract_cls_embeddings(model, tensors, indices, device)

    print("Computing prediction probabilities...")
    probs = xai.get_predictions(model, tensors, indices, device)

    xai.plot_cls_embedding_space(
        embeddings      = embeddings,
        labels          = labels_arr,
        probs           = probs,
        samples_df      = samples_df,
        indices         = indices,
        output_dir      = ds_out,
        method          = DIM_REDUCTION,
        split_filter    = DATASET_SPLIT,
        patient_overlay = patient_overlay if SHOW_PATIENTS_ON_DATASET_PLOTS else {},
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

    return embeddings, labels_arr


def run_patient_level(subject_id: int, patient_cls: torch.Tensor,
                      model, samples_df, tensors, vocab, inv_vocab,
                      bg_embeddings, bg_labels, device) -> None:
    print(f"\n{'=' * 58}")
    print(f"  Per-patient  (subject_id={subject_id})")
    print(f"{'=' * 58}\n")

    pat_out         = OUTPUT_DIR / f"patient_{subject_id}"
    pat_out.mkdir(parents=True, exist_ok=True)
    patient_indices = xai.find_patient_indices(samples_df, subject_id)
    cycles          = [int(samples_df.iloc[i]["cycle_number"]) for i in patient_indices]
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

    print(f"  Projecting background + patient jointly ({DIM_REDUCTION.upper()})...")
    bg_coords, pat_coords, _, method_name = xai.project_together(
        background_embeddings = bg_embeddings,
        patient_embeddings    = patient_cls,
        method                = DIM_REDUCTION,
    )

    xai.plot_cls_trajectory(
        cycle_data        = cycle_data,
        patient_coords_2d = pat_coords,
        subject_id        = subject_id,
        coords_background = bg_coords,
        background_labels = bg_labels,
        output_path       = pat_out / "cls_trajectory.png",
        method_name       = method_name,
    )

    xai.plot_rollout_per_cycle(
        cycle_data  = cycle_data,
        subject_id  = subject_id,
        output_path = pat_out / "rollout_per_cycle.png",
        top_k       = ROLLOUT_TOP_K,
    )

    xai.plot_rollout_heatmap(
        cycle_data  = cycle_data,
        subject_id  = subject_id,
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
            subject_id          = subject_id,
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
            subject_id          = subject_id,
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

    # ── Load model and data ───────────────────────────────────────────────────
    print("\nLoading model and data...")
    model, cfg, samples_df, tensors, vocab, inv_vocab, split_indices = xai.load_setup(
        model_dir         = MODEL_DIR,
        data_dir          = DATA_DIR,
        device            = device,
        checkpoint_metric = CHECKPOINT_METRIC,
    )

    if ENRICH_DRUG_INFO:
        samples_df = xai.enrich_samples_with_drug_info(
            samples_df,
            data_dir          = DATA_DIR,
            cohort_table_path = COHORT_TABLE_PATH,
        )

    # ── Extract patient CLS embeddings once (reused for overlay + deep dive) ─
    patient_overlay: dict[int, torch.Tensor] = {}
    if PATIENT_SUBJECT_IDS:
        print(f"\nExtracting CLS embeddings for {len(PATIENT_SUBJECT_IDS)} patient(s)...")
        for sid in PATIENT_SUBJECT_IDS:
            try:
                pat_idx = xai.find_patient_indices(samples_df, sid)
                pat_emb = xai.extract_cls_embeddings(model, tensors, pat_idx, device)
                patient_overlay[sid] = pat_emb
                cycles = [int(samples_df.iloc[i]["cycle_number"]) for i in pat_idx]
                print(f"  subject_id={sid}: cycles {cycles}")
            except ValueError as e:
                print(f"  WARNING: {e} — skipping")

    # ── Dataset-level plots ───────────────────────────────────────────────────
    bg_embeddings, bg_labels = None, None
    if RUN_DATASET_PLOTS:
        bg_embeddings, bg_labels = run_dataset_level(
            model, samples_df, tensors, inv_vocab, split_indices, device,
            patient_overlay=patient_overlay,
        )

    # ── Per-patient deep dives ────────────────────────────────────────────────
    if RUN_PATIENT_DEEP_DIVE and patient_overlay:
        if bg_embeddings is None:
            print("\nWARNING: RUN_DATASET_PLOTS=False — trajectory background will be empty.")
            bg_embeddings = torch.empty(0, model.embedding.d_model
                                        if hasattr(model.embedding, "d_model")
                                        else cfg["d_model"])
            bg_labels = []

        for sid, patient_cls in patient_overlay.items():
            run_patient_level(
                subject_id    = sid,
                patient_cls   = patient_cls,
                model         = model,
                samples_df    = samples_df,
                tensors       = tensors,
                vocab         = vocab,
                inv_vocab     = inv_vocab,
                bg_embeddings = bg_embeddings,
                bg_labels     = bg_labels,
                device        = device,
            )

    print(f"\n{'=' * 58}")
    print(f"  All outputs → {OUTPUT_DIR}")
    print(f"{'=' * 58}")
