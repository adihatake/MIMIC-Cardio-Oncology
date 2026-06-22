"""
generate_cycle_modeling_table.py

Builds the cycle-level cardiotoxicity modelling table from MIMIC-IV raw data
by executing a chain of DuckDB SQL files.

Outputs (cohort_outputs/cycle_modeling_ver2/):
    final_cycle_modeling_table.csv / .parquet
    final_cycle_binary_modeling_table.csv / .parquet
    row_level_label_breakdown.csv
    row_level_binary_label_breakdown.csv
    row_level_drug_class_breakdown.csv
    patient_level_labels.csv
    patient_level_summary.csv
    cohort_accounting.csv
"""

from __future__ import annotations

import os
from pathlib import Path

import duckdb
import pandas as pd

# ── paths ──────────────────────────────────────────────────────────────────────
REPO_ROOT     = Path(__file__).resolve().parent.parent
DATA_LOCATION = Path("/Users/catherinebalajadia/Downloads/2026_Summer_Research/MIMIC_IV_raw_data").resolve()

SQL_ROOT              = REPO_ROOT / "sql_files"
DIAGNOSES_SQL_DIR     = SQL_ROOT / "diagnoses_sql"
PRESCRIPTIONS_SQL_DIR = SQL_ROOT / "prescriptions_sql"
DRUG_CYCLES_SQL_DIR   = SQL_ROOT / "drug_cycles_sql"

OUTPUT_DIR = REPO_ROOT / "cohort_outputs" / "cycle_modeling_ver2"

BASE_SQL_PATHS = [
    DIAGNOSES_SQL_DIR     / "active_cancer.sql",
    DIAGNOSES_SQL_DIR     / "personal_history_cancer.sql",
    DIAGNOSES_SQL_DIR     / "history_and_active.sql",
    PRESCRIPTIONS_SQL_DIR / "prescriptions_count_regex.sql",
]

CYCLE_SQL_PATHS = [
    DRUG_CYCLES_SQL_DIR / "00_parameters_and_windows.sql",
    DRUG_CYCLES_SQL_DIR / "01_drug_classification_and_first_drug.sql",
    DRUG_CYCLES_SQL_DIR / "02_cycle_exposures.sql",
    DRUG_CYCLES_SQL_DIR / "03_lvef_toxicity_events.sql",
    DRUG_CYCLES_SQL_DIR / "04_cv_toxicity_events.sql",
    DRUG_CYCLES_SQL_DIR / "05_first_toxicity_and_observation.sql",
    DRUG_CYCLES_SQL_DIR / "06_final_modeling_table.sql",
]
# ──────────────────────────────────────────────────────────────────────────────


def _execute_sql_file(con: duckdb.DuckDBPyConnection, path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing SQL file: {path}")
    sql = path.read_text()
    sql = sql.replace("CREATE VIEW active_cancer",   "CREATE OR REPLACE VIEW active_cancer")
    sql = sql.replace("CREATE VIEW oncology_drugs",  "CREATE OR REPLACE VIEW oncology_drugs")
    con.execute(sql)


def _count_rows(con: duckdb.DuckDBPyConnection, name: str) -> int:
    return con.execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0]


def _write_dataframe(df: pd.DataFrame, output_dir: Path, stem: str) -> None:
    df.to_csv(output_dir / f"{stem}.csv", index=False)
    print(f"  wrote {stem}.csv")
    try:
        df.to_parquet(output_dir / f"{stem}.parquet", index=False)
        print(f"  wrote {stem}.parquet")
    except Exception as exc:
        print(f"  skipped parquet for {stem}: {exc}")


def _assign_patient_status(labels) -> str:
    labels = set(labels)
    if "positive" in labels:
        return "positive_patient"
    elif "negative_observed" in labels:
        return "negative_observed_patient"
    elif "unknown_no_followup_evidence" in labels:
        return "unknown_patient"
    elif "exclude_already_toxic" in labels:
        return "only_excluded_rows_review"
    return "unclassified_review"


def main(output_name: str | None = None, data_location: Path | None = None) -> None:
    out_dir  = REPO_ROOT / "cohort_outputs" / output_name if output_name else OUTPUT_DIR
    data_loc = data_location or DATA_LOCATION
    out_dir.mkdir(parents=True, exist_ok=True)

    # Recompute SQL roots if data_location changed
    mimic_hosp_dir = data_loc / "mimic-iv-3.1/hosp"
    mimic_echo_dir = data_loc / "mimic-iv-echo"

    print("REPO_ROOT:    ", REPO_ROOT)
    print("DATA_LOCATION:", data_loc)
    print("OUTPUT_DIR:   ", out_dir)

    # Verify SQL files exist before connecting
    missing = [p for p in BASE_SQL_PATHS + CYCLE_SQL_PATHS if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing SQL files:\n" + "\n".join(str(p) for p in missing))

    # DuckDB SQL uses relative CSV paths rooted at DATA_LOCATION
    os.chdir(data_loc)
    con = duckdb.connect(":memory:")

    # ── base cohort ───────────────────────────────────────────────────────────
    print("\nRunning base cohort SQL...")
    for path in BASE_SQL_PATHS:
        print(f"  {path.relative_to(REPO_ROOT)}")
        _execute_sql_file(con, path)
    print(f"  all_cancer_patients: {_count_rows(con, 'all_cancer_patients'):,} rows")
    print(f"  oncology_drugs:      {_count_rows(con, 'oncology_drugs'):,} rows")

    # ── cycle modelling SQL ───────────────────────────────────────────────────
    print("\nRunning cycle modelling SQL...")
    for path in CYCLE_SQL_PATHS:
        print(f"  {path.relative_to(REPO_ROOT)}")
        _execute_sql_file(con, path)

    print(f"  oncology_cycle_exposures:         {_count_rows(con, 'oncology_cycle_exposures'):,} rows")
    print(f"  lvef_toxicity_events:             {_count_rows(con, 'lvef_toxicity_events'):,} rows")
    print(f"  cv_toxicity_events:               {_count_rows(con, 'cv_toxicity_events'):,} rows")
    print(f"  first_cardiotoxicity_event:       {_count_rows(con, 'first_cardiotoxicity_event'):,} rows")
    print(f"  final_cycle_modeling_table:       {_count_rows(con, 'final_cycle_modeling_table'):,} rows")
    print(f"  final_cycle_binary_modeling_table:{_count_rows(con, 'final_cycle_binary_modeling_table'):,} rows")

    # ── load into pandas ──────────────────────────────────────────────────────
    final_df = con.execute(
        "SELECT * FROM final_cycle_modeling_table ORDER BY subject_id, cycle_number"
    ).df()
    binary_df = con.execute(
        "SELECT * FROM final_cycle_binary_modeling_table ORDER BY subject_id, cycle_number"
    ).df()

    # ── summary breakdowns ────────────────────────────────────────────────────
    label_breakdown = (
        final_df.groupby("label", dropna=False)
        .agg(
            n_cycle_rows=("subject_id", "size"),
            n_patients_with_at_least_one_row=("subject_id", "nunique"),
        )
        .reset_index()
        .sort_values("n_cycle_rows", ascending=False)
    )

    binary_label_breakdown = (
        binary_df.groupby(["label", "binary_label"], dropna=False)
        .agg(
            n_cycle_rows=("subject_id", "size"),
            n_patients_with_at_least_one_row=("subject_id", "nunique"),
        )
        .reset_index()
        .sort_values("n_cycle_rows", ascending=False)
    )

    drug_class_breakdown = (
        final_df.groupby("drug_classes_in_cycle", dropna=False)
        .agg(
            n_cycle_rows=("subject_id", "size"),
            n_patients_with_at_least_one_row=("subject_id", "nunique"),
            n_positive_cycle_rows=("toxicity_in_window", "sum"),
        )
        .reset_index()
        .sort_values("n_cycle_rows", ascending=False)
    )

    patient_level_labels = (
        final_df.groupby("subject_id")["label"]
        .apply(_assign_patient_status)
        .reset_index(name="patient_status")
    )

    patient_level_summary = (
        patient_level_labels.groupby("patient_status")
        .agg(n_patients=("subject_id", "nunique"))
        .reset_index()
        .sort_values("n_patients", ascending=False)
    )

    cohort_accounting = pd.DataFrame({
        "metric": [
            "patients_in_cancer_first_drug",
            "patients_in_final_cycle_modeling_table",
            "patients_in_binary_modeling_table",
        ],
        "n_patients": [
            con.execute("SELECT COUNT(DISTINCT subject_id) FROM cancer_first_drug").fetchone()[0],
            final_df["subject_id"].nunique(),
            binary_df["subject_id"].nunique(),
        ],
    })

    # ── print cohort summary ──────────────────────────────────────────────────
    print("\n── Label breakdown ─────────────────────────────────────────────────")
    print(label_breakdown.to_string(index=False))
    print("\n── Patient-level summary ───────────────────────────────────────────")
    print(patient_level_summary.to_string(index=False))
    print("\n── Cohort accounting ───────────────────────────────────────────────")
    print(cohort_accounting.to_string(index=False))

    # ── write outputs ─────────────────────────────────────────────────────────
    print(f"\nWriting outputs to {out_dir}")
    _write_dataframe(final_df,   out_dir, "final_cycle_modeling_table")
    _write_dataframe(binary_df,  out_dir, "final_cycle_binary_modeling_table")

    label_breakdown.to_csv(       out_dir / "row_level_label_breakdown.csv",        index=False)
    binary_label_breakdown.to_csv(out_dir / "row_level_binary_label_breakdown.csv", index=False)
    drug_class_breakdown.to_csv(  out_dir / "row_level_drug_class_breakdown.csv",   index=False)
    patient_level_labels.to_csv(  out_dir / "patient_level_labels.csv",             index=False)
    patient_level_summary.to_csv( out_dir / "patient_level_summary.csv",            index=False)
    cohort_accounting.to_csv(     out_dir / "cohort_accounting.csv",                index=False)

    print("Done.")


if __name__ == "__main__":
    main()
