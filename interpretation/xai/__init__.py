"""
interpretation/xai/  — population and patient-level xAI visualizations.

Submodules
----------
utils        shared constants, data loading, split helpers
embeddings   CLS extraction, prediction probabilities, dimensionality reduction
population   dataset-level plots: CLS embedding space, aggregate IG
patient      per-patient plots: CLS trajectory, attention rollout
perturbation perturbation analysis: token edits, CLS shift, ΔP plots
"""

from .utils import (
    LAB_NAMES,
    EVENT_TYPE_COLORS,
    LABEL_COLORS,
    LABEL_NAMES,
    PERTURB_COLORS,
    DRUG_CLASS_COLORS,
    load_vocab,
    load_samples_df,
    load_setup,
    get_indices,
    find_patient_indices,
    enrich_samples_with_drug_info,
)

from .embeddings import (
    extract_cls_embeddings,
    get_predictions,
    reduce_dim,
    project_together,
)

from .population import (
    plot_cls_embedding_space,
    compute_population_ig,
    plot_population_ig,
)

from .patient import (
    get_patient_cycle_data,
    plot_cls_trajectory,
    plot_rollout_per_cycle,
    plot_rollout_heatmap,
)

from .perturbation import (
    apply_perturbation,
    run_perturbation_analysis,
    plot_perturbation_trajectory,
    plot_perturbation_delta,
)

__all__ = [
    # utils
    "LAB_NAMES", "EVENT_TYPE_COLORS", "LABEL_COLORS", "LABEL_NAMES", "PERTURB_COLORS",
    "DRUG_CLASS_COLORS", "PATIENT_OVERLAY_COLORS",
    "load_vocab", "load_samples_df", "load_setup", "get_indices", "find_patient_indices",
    "enrich_samples_with_drug_info",
    # embeddings
    "extract_cls_embeddings", "get_predictions", "reduce_dim", "project_together",
    # population
    "plot_cls_embedding_space", "compute_population_ig", "plot_population_ig",
    # patient
    "get_patient_cycle_data", "plot_cls_trajectory",
    "plot_rollout_per_cycle", "plot_rollout_heatmap",
    # perturbation
    "apply_perturbation", "run_perturbation_analysis",
    "plot_perturbation_trajectory", "plot_perturbation_delta",
]
