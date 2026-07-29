"""
cohort_pipeline.py

Defines what varies between cohort SQL pipelines.

To add a new pipeline endpoint:
  1. Write the SQL files in sql_files/drug_cycles_sql/<name>/ and
     sql_files/prescriptions_sql/<name>/.
  2. Define a PipelineConfig below and add it to PIPELINE_REGISTRY.
  3. Set PIPELINE = "<name>" in run_cohort.py — no other Python changes needed.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PipelineConfig:
    """
    Describes one cohort SQL pipeline. All SQL filenames are relative to the
    directory passed to generate_cohort.main() at runtime.
    """
    name: str

    # ── SQL files ─────────────────────────────────────────────────────────────
    # Prescriptions pre-filter filename (within prescriptions_sql_dir)
    prescriptions_sql_file: str

    # Ordered cycle SQL filenames to execute (within cycle_sql_dir)
    cycle_sql_files: list[str]

    # Optional SQL files — allowed to fail (e.g., echo endpoint needs MIMIC-IV-Echo).
    # Primary pipeline output is unaffected if these fail.
    optional_sql_files: list[str] = field(default_factory=list)

    # ── Progress reporting ────────────────────────────────────────────────────
    # Views to count and print after the required SQL runs.
    # List of (view_name, display_label).
    progress_views: list[tuple[str, str]] = field(default_factory=list)

    # ── Output tables ─────────────────────────────────────────────────────────
    # Full cycle table view name (all labels, all rows).
    full_table_view: str = "final_cycle_modeling_table"

    # Required binary tables: [(view_name, output_file_stem), ...].
    # The FIRST entry is also written as "final_cycle_binary_modeling_table"
    # so the tokenizer can find it without knowing which pipeline was used.
    binary_table_views: list[tuple[str, str]] = field(default_factory=list)

    # Optional binary tables — written only if the optional SQL succeeded.
    optional_binary_table_views: list[tuple[str, str]] = field(default_factory=list)

    # ── Cohort accounting ─────────────────────────────────────────────────────
    # View whose DISTINCT subject_id count goes into the accounting table.
    anchor_view: str = "cancer_first_drug"

    # ── Summary breakdown config ──────────────────────────────────────────────
    # Column ORDER BY when fetching results from DuckDB.
    sort_cols: list[str] = field(default_factory=lambda: ["subject_id", "cycle_number"])

    # Column for drug-class grouping in breakdowns.
    drug_class_col: str = "drug_classes_in_cycle"

    # Column whose .sum() gives positive row counts in the drug-class breakdown.
    # None = omit the column.
    positive_col: str | None = None

    # Columns for a pre-existing condition breakdown (empty list = skip).
    preexisting_cols: list[str] = field(default_factory=list)

    # Label value that maps to "unknown_patient" in the patient-level summary.
    unknown_followup_label: str = "unknown_no_followup_evidence"


# ── Built-in pipeline configs ─────────────────────────────────────────────────

MAIN_PIPELINE = PipelineConfig(
    name="main",
    prescriptions_sql_file="prescriptions_count_regex.sql",
    cycle_sql_files=[
        "00_parameters_and_windows.sql",
        "01_drug_classification_and_first_drug.sql",
        "02_cycle_exposures.sql",
        "03_lvef_toxicity_events.sql",
        "04_cv_toxicity_events.sql",
        "05_first_toxicity_and_observation.sql",
        "06_final_modeling_table.sql",
    ],
    progress_views=[
        ("oncology_cycle_exposures",          "oncology_cycle_exposures"),
        ("lvef_toxicity_events",              "lvef_toxicity_events"),
        ("cv_toxicity_events",                "cv_toxicity_events"),
        ("first_cardiotoxicity_event",        "first_cardiotoxicity_event"),
        ("final_cycle_modeling_table",        "final_cycle_modeling_table"),
        ("final_cycle_binary_modeling_table", "final_cycle_binary_modeling_table"),
    ],
    full_table_view="final_cycle_modeling_table",
    binary_table_views=[
        ("final_cycle_binary_modeling_table", "final_cycle_binary_modeling_table"),
    ],
    anchor_view="cancer_first_drug",
    sort_cols=["subject_id", "cycle_number"],
    drug_class_col="drug_classes_in_cycle",
    positive_col="toxicity_in_window",
    unknown_followup_label="unknown_no_followup_evidence",
)

HF_CARDIOTOX_PIPELINE = PipelineConfig(
    name="hf_cardiotox",
    prescriptions_sql_file="prescriptions_hf_cardiotox.sql",
    cycle_sql_files=[
        "00_parameters.sql",
        "01_drug_classification.sql",
        "02_cycle_exposures.sql",
        "03_heart_failure_events.sql",
        "04_observation_and_death.sql",
        "05_final_modeling_table.sql",
    ],
    optional_sql_files=["06_lvef_gls_events.sql"],
    progress_views=[
        ("hf_cycle_exposures",                       "hf_cycle_exposures"),
        ("incident_hf_events",                       "incident_hf_events"),
        ("hf_final_cycle_modeling_table",            "hf_final_cycle_modeling_table"),
        ("hf_final_binary_modeling_table_strict",    "hf_final_binary_modeling_table_strict"),
        ("hf_final_binary_modeling_table_inclusive", "hf_final_binary_modeling_table_inclusive"),
    ],
    full_table_view="hf_final_cycle_modeling_table",
    binary_table_views=[
        ("hf_final_binary_modeling_table_strict",    "hf_final_binary_modeling_table_strict"),
        ("hf_final_binary_modeling_table_inclusive", "hf_final_binary_modeling_table_inclusive"),
    ],
    optional_binary_table_views=[
        ("echo_cardiotox_binary_modeling_table_strict",    "echo_cardiotox_binary_modeling_table_strict"),
        ("echo_cardiotox_binary_modeling_table_inclusive", "echo_cardiotox_binary_modeling_table_inclusive"),
    ],
    anchor_view="hf_patient_first_drug",
    sort_cols=["subject_id", "drug_class", "cycle_number"],
    drug_class_col="drug_class",
    positive_col="binary_label",
    preexisting_cols=["has_pre_existing_hf", "has_pre_existing_cmp"],
    unknown_followup_label="unknown_insufficient_followup",
)

HF_CARDIOTOX_V2_PIPELINE = PipelineConfig(
    name="hf_cardiotox_v2",
    prescriptions_sql_file="prescriptions_hf_cardiotox.sql",
    cycle_sql_files=[
        "00_parameters.sql",
        "01_drug_classification.sql",
        "02_cycle_exposures.sql",
        "03_heart_failure_events.sql",
        "04_echo_events.sql",
        "05_combined_cardiotox_event.sql",
        "06_observation_and_death.sql",
        "07_final_modeling_table.sql",
    ],
    progress_views=[
        ("hf_cycle_exposures",                           "hf_cycle_exposures"),
        ("incident_hf_events",                           "incident_hf_events"),
        ("first_echo_cardiotox_event",                   "first_echo_cardiotox_event"),
        ("first_combined_cardiotox_event",               "first_combined_cardiotox_event"),
        ("ctrcd_final_cycle_modeling_table",             "ctrcd_final_cycle_modeling_table"),
        ("ctrcd_final_binary_modeling_table_strict",     "ctrcd_final_binary_modeling_table_strict"),
        ("ctrcd_final_binary_modeling_table_inclusive",  "ctrcd_final_binary_modeling_table_inclusive"),
    ],
    full_table_view="ctrcd_final_cycle_modeling_table",
    binary_table_views=[
        ("ctrcd_final_binary_modeling_table_strict",    "ctrcd_final_binary_modeling_table_strict"),
        ("ctrcd_final_binary_modeling_table_inclusive", "ctrcd_final_binary_modeling_table_inclusive"),
    ],
    anchor_view="hf_patient_first_drug",
    sort_cols=["subject_id", "drug_class", "cycle_number"],
    drug_class_col="drug_class",
    positive_col="binary_label",
    preexisting_cols=["has_pre_existing_hf", "has_pre_existing_cmp"],
    unknown_followup_label="unknown_insufficient_followup",
)

HF_CARDIOTOX_V2_365D_PIPELINE = PipelineConfig(
    name="hf_cardiotox_v2_365d",
    prescriptions_sql_file="prescriptions_hf_cardiotox.sql",
    cycle_sql_files=[
        "00_parameters.sql",
        "01_drug_classification.sql",
        "02_cycle_exposures.sql",
        "03_heart_failure_events.sql",
        "04_echo_events.sql",
        "05_combined_cardiotox_event.sql",
        "06_observation_and_death.sql",
        "07_final_modeling_table.sql",
    ],
    progress_views=[
        ("hf_cycle_exposures",                           "hf_cycle_exposures"),
        ("incident_hf_events",                           "incident_hf_events"),
        ("first_echo_cardiotox_event",                   "first_echo_cardiotox_event"),
        ("first_combined_cardiotox_event",               "first_combined_cardiotox_event"),
        ("ctrcd_final_cycle_modeling_table",             "ctrcd_final_cycle_modeling_table"),
        ("ctrcd_final_binary_modeling_table_strict",     "ctrcd_final_binary_modeling_table_strict"),
        ("ctrcd_final_binary_modeling_table_inclusive",  "ctrcd_final_binary_modeling_table_inclusive"),
    ],
    full_table_view="ctrcd_final_cycle_modeling_table",
    binary_table_views=[
        ("ctrcd_final_binary_modeling_table_strict",    "ctrcd_final_binary_modeling_table_strict"),
        ("ctrcd_final_binary_modeling_table_inclusive", "ctrcd_final_binary_modeling_table_inclusive"),
    ],
    anchor_view="hf_patient_first_drug",
    sort_cols=["subject_id", "drug_class", "cycle_number"],
    drug_class_col="drug_class",
    positive_col="binary_label",
    preexisting_cols=["has_pre_existing_hf", "has_pre_existing_cmp"],
    unknown_followup_label="unknown_insufficient_followup",
)

ANTHRACYCLINE_ONLY_PIPELINE = PipelineConfig(
    name="anthracycline_only_exposure",
    prescriptions_sql_file="prescriptions_hf_cardiotox.sql",
    cycle_sql_files=[
        "00_parameters.sql",
        "01_drug_classification.sql",
        "02_cycle_exposures.sql",
        "03_heart_failure_events.sql",
        "04_echo_events.sql",
        "05_combined_cardiotox_event.sql",
        "06_observation_and_death.sql",
        "07_final_modeling_table.sql",
    ],
    progress_views=[
        ("hf_cycle_exposures",                                  "hf_cycle_exposures (all classes)"),
        ("anthracycline_first_patients",                        "anthracycline_first_patients"),
        ("anthracycline_cycle_exposures",                       "anthracycline_cycle_exposures"),
        ("first_combined_cardiotox_event",                      "first_combined_cardiotox_event"),
        ("anthracycline_final_cycle_modeling_table",            "anthracycline_final_cycle_modeling_table"),
        ("anthracycline_final_binary_modeling_table_strict",    "anthracycline_final_binary_modeling_table_strict"),
        ("anthracycline_final_binary_modeling_table_inclusive", "anthracycline_final_binary_modeling_table_inclusive"),
    ],
    full_table_view="anthracycline_final_cycle_modeling_table",
    binary_table_views=[
        ("anthracycline_final_binary_modeling_table_strict",    "anthracycline_final_binary_modeling_table_strict"),
        ("anthracycline_final_binary_modeling_table_inclusive", "anthracycline_final_binary_modeling_table_inclusive"),
    ],
    anchor_view="anthracycline_first_patients",
    sort_cols=["subject_id", "cycle_number"],
    drug_class_col="drug_class",
    positive_col="binary_label",
    preexisting_cols=["has_pre_existing_hf", "has_pre_existing_cmp"],
    unknown_followup_label="unknown_insufficient_followup",
)

# Registry: maps pipeline name string → config object.
# Add new pipelines here.
PIPELINE_REGISTRY: dict[str, PipelineConfig] = {
    "main":                       MAIN_PIPELINE,
    "hf_cardiotox":               HF_CARDIOTOX_PIPELINE,
    "hf_cardiotox_v2":            HF_CARDIOTOX_V2_PIPELINE,
    "hf_cardiotox_v2_365d":       HF_CARDIOTOX_V2_365D_PIPELINE,
    "anthracycline_only_exposure": ANTHRACYCLINE_ONLY_PIPELINE,
}
