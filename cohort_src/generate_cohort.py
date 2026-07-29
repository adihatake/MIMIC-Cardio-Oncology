"""
generate_cohort.py

Generic cohort builder — executes any SQL pipeline defined by a PipelineConfig
and exports the results. All clinical logic lives in the SQL files.

Usage:
    import cohort_src.generate_cohort as cohort_module
    cohort_module.main(
        data_location         = Path("/path/to/MIMIC_IV_raw_data"),
        pipeline              = "hf_cardiotox",   # or a PipelineConfig object
        output_name           = "hf_cardiotox_v1",
        cycle_sql_dir         = ...,   # optional override
        prescriptions_sql_dir = ...,   # optional override
    )

SQL directory defaults:
    cycle_sql_dir         → sql_files/drug_cycles_sql/<pipeline.name>/
    prescriptions_sql_dir → sql_files/prescriptions_sql/<pipeline.name>/

Override these in run_cohort.py only when using a versioned subdirectory
(e.g., "jul28_alt") instead of the canonical pipeline directory.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Union

import duckdb
import pandas as pd

from cohort_src.cohort_pipeline import PipelineConfig, PIPELINE_REGISTRY

REPO_ROOT         = Path(__file__).resolve().parent.parent
SQL_ROOT          = REPO_ROOT / "sql_files"
DIAGNOSES_SQL_DIR = SQL_ROOT / "diagnoses_sql"

_BASE_SQL_FILES = [
    "active_cancer.sql",
    "personal_history_cancer.sql",
    "history_and_active.sql",
]


# ── helpers ───────────────────────────────────────────────────────────────────

def _execute_sql_file(con: duckdb.DuckDBPyConnection, path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing SQL file: {path}")
    con.execute(path.read_text())


def _count_rows(con: duckdb.DuckDBPyConnection, view: str) -> int:
    return con.execute(f"SELECT COUNT(*) FROM {view}").fetchone()[0]


def _view_exists(con: duckdb.DuckDBPyConnection, view: str) -> bool:
    try:
        con.execute(f"SELECT 1 FROM {view} LIMIT 1")
        return True
    except Exception:
        return False


def _materialize_view(con: duckdb.DuckDBPyConnection, view: str) -> None:
    """Replace a VIEW with an in-memory TABLE of the same name."""
    tmp = f"{view}__mat"
    con.execute(f"CREATE TABLE {tmp} AS SELECT * FROM {view}")
    con.execute(f"DROP VIEW IF EXISTS {view}")
    con.execute(f"ALTER TABLE {tmp} RENAME TO {view}")


def _write_dataframe(df: pd.DataFrame, output_dir: Path, stem: str) -> None:
    df.to_csv(output_dir / f"{stem}.csv", index=False)
    print(f"  wrote {stem}.csv")
    try:
        df.to_parquet(output_dir / f"{stem}.parquet", index=False)
        print(f"  wrote {stem}.parquet")
    except Exception as exc:
        print(f"  skipped parquet for {stem}: {exc}")


def _patient_status(labels, unknown_label: str) -> str:
    s = set(labels)
    if "positive"            in s: return "positive_patient"
    if "negative_observed"   in s: return "negative_observed_patient"
    if unknown_label         in s: return "unknown_patient"
    if "exclude_already_toxic" in s: return "only_excluded_rows_review"
    return "unclassified_review"


# ── main ──────────────────────────────────────────────────────────────────────

def main(
    data_location: Path,
    pipeline: Union[str, PipelineConfig] = "hf_cardiotox",
    output_name: str | None = None,
    cycle_sql_dir: Path | None = None,
    prescriptions_sql_dir: Path | None = None,
) -> None:
    cfg: PipelineConfig = (
        PIPELINE_REGISTRY[pipeline] if isinstance(pipeline, str) else pipeline
    )

    out_dir = REPO_ROOT / "cohort_outputs" / (output_name or cfg.name)
    out_dir.mkdir(parents=True, exist_ok=True)

    _cycle_dir = cycle_sql_dir         or (SQL_ROOT / "drug_cycles_sql"   / cfg.name)
    _presc_dir = prescriptions_sql_dir or (SQL_ROOT / "prescriptions_sql" / cfg.name)

    base_paths     = [DIAGNOSES_SQL_DIR / f for f in _BASE_SQL_FILES]
    presc_path     = _presc_dir / cfg.prescriptions_sql_file
    cycle_paths    = [_cycle_dir / f for f in cfg.cycle_sql_files]
    optional_paths = [_cycle_dir / f for f in cfg.optional_sql_files]

    print(f"pipeline      : {cfg.name}")
    print(f"REPO_ROOT     : {REPO_ROOT}")
    print(f"DATA_LOCATION : {data_location}")
    print(f"OUTPUT_DIR    : {out_dir}")

    missing = [p for p in base_paths + [presc_path] + cycle_paths if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing SQL files:\n" + "\n".join(str(p) for p in missing))

    os.chdir(data_location)
    con = duckdb.connect(":memory:")

    # ── base cohort (shared across all pipelines) ─────────────────────────────
    print("\nRunning base cohort SQL...")
    for path in base_paths:
        print(f"  {path.relative_to(REPO_ROOT)}")
        _execute_sql_file(con, path)
    print(f"  {presc_path.relative_to(REPO_ROOT)}")
    _execute_sql_file(con, presc_path)
    # Materialize oncology_drugs as a table so the prescriptions CSV is scanned
    # only once — every downstream view otherwise re-reads it from disk.
    _materialize_view(con, "oncology_drugs")
    print(f"  all_cancer_patients : {_count_rows(con, 'all_cancer_patients'):,} rows")
    print(f"  oncology_drugs      : {_count_rows(con, 'oncology_drugs'):,} rows")

    # ── required pipeline SQL ─────────────────────────────────────────────────
    print(f"\nRunning {cfg.name} SQL...")
    for path in cycle_paths:
        print(f"  {path.relative_to(REPO_ROOT)}")
        _execute_sql_file(con, path)

    # Materialize heavy intermediate views and final output views so that
    # progress counts, full table fetch, and binary table fetches don't each
    # retrigger the full computation chain (window functions, joins, etc.).
    _to_materialize = [
        "hf_cohort_drug_starts",  # cuts recomputation early in all hf_* chains
        "hf_cycle_exposures",     # expensive window functions used by all downstream views
        cfg.full_table_view,
    ] + [v for v, _ in cfg.binary_table_views]
    for view in _to_materialize:
        if _view_exists(con, view):
            _materialize_view(con, view)

    for view, label in cfg.progress_views:
        print(f"  {label:<47}: {_count_rows(con, view):,} rows")

    # ── optional pipeline SQL ─────────────────────────────────────────────────
    optional_ok = False
    for path in optional_paths:
        if not path.exists():
            print(f"\n  optional SQL not found, skipping: {path.name}")
            continue
        print(f"\n  Running optional: {path.relative_to(REPO_ROOT)}")
        try:
            _execute_sql_file(con, path)
            optional_ok = True
        except Exception as exc:
            print(f"  WARNING: optional SQL failed (missing data?): {exc}")
            print("  Primary pipeline output is unaffected.")

    # ── load DataFrames ───────────────────────────────────────────────────────
    order = "ORDER BY " + ", ".join(cfg.sort_cols)
    full_df = con.execute(f"SELECT * FROM {cfg.full_table_view} {order}").df()

    binary_dfs: list[tuple[pd.DataFrame, str]] = [
        (con.execute(f"SELECT * FROM {view} {order}").df(), stem)
        for view, stem in cfg.binary_table_views
    ]

    # ── summary breakdowns ────────────────────────────────────────────────────
    dc = cfg.drug_class_col

    label_breakdown = (
        full_df.groupby([dc, "label"], dropna=False)
        .agg(n_cycle_rows=("subject_id", "size"), n_patients=("subject_id", "nunique"))
        .reset_index()
        .sort_values([dc, "n_cycle_rows"], ascending=[True, False])
    )

    canonical_df = binary_dfs[0][0] if binary_dfs else full_df
    binary_label_breakdown = (
        canonical_df.groupby([dc, "label", "binary_label"], dropna=False)
        .agg(n_cycle_rows=("subject_id", "size"), n_patients=("subject_id", "nunique"))
        .reset_index()
        .sort_values([dc, "n_cycle_rows"], ascending=[True, False])
    )

    agg: dict = {"n_cycle_rows": ("subject_id", "size"), "n_patients": ("subject_id", "nunique")}
    if cfg.positive_col and cfg.positive_col in full_df.columns:
        agg["n_positive_rows"] = (cfg.positive_col, "sum")
    drug_class_breakdown = (
        full_df.groupby(dc, dropna=False).agg(**agg)
        .reset_index().sort_values("n_cycle_rows", ascending=False)
    )

    preexisting_breakdown = None
    if cfg.preexisting_cols and all(c in full_df.columns for c in cfg.preexisting_cols):
        preexisting_breakdown = (
            full_df.drop_duplicates("subject_id")[["subject_id"] + cfg.preexisting_cols]
            .groupby(cfg.preexisting_cols, dropna=False)
            .agg(n_patients=("subject_id", "nunique"))
            .reset_index()
        )

    patient_level_labels = (
        full_df.groupby("subject_id")["label"]
        .apply(lambda s: _patient_status(s, cfg.unknown_followup_label))
        .reset_index(name="patient_status")
    )
    patient_level_summary = (
        patient_level_labels.groupby("patient_status")
        .agg(n_patients=("subject_id", "nunique"))
        .reset_index().sort_values("n_patients", ascending=False)
    )

    accounting_rows = [
        (f"patients_in_{cfg.anchor_view}", con.execute(
            f"SELECT COUNT(DISTINCT subject_id) FROM {cfg.anchor_view}"
        ).fetchone()[0]),
        (f"patients_in_{cfg.full_table_view}", full_df["subject_id"].nunique()),
    ] + [(f"patients_in_{stem}", df["subject_id"].nunique()) for df, stem in binary_dfs]
    cohort_accounting = pd.DataFrame(accounting_rows, columns=["metric", "n_patients"])

    # ── print summary ─────────────────────────────────────────────────────────
    print("\n── Label breakdown ─────────────────────────────────────────────────")
    print(label_breakdown.to_string(index=False))
    print("\n── Drug class summary ──────────────────────────────────────────────")
    print(drug_class_breakdown.to_string(index=False))
    if preexisting_breakdown is not None:
        print("\n── Pre-existing cardiac history ────────────────────────────────────")
        print(preexisting_breakdown.to_string(index=False))
    print("\n── Patient-level summary ───────────────────────────────────────────")
    print(patient_level_summary.to_string(index=False))
    print("\n── Cohort accounting ───────────────────────────────────────────────")
    print(cohort_accounting.to_string(index=False))

    # ── write outputs ─────────────────────────────────────────────────────────
    print(f"\nWriting outputs to {out_dir}")
    _write_dataframe(full_df, out_dir, cfg.full_table_view)

    for i, (df, stem) in enumerate(binary_dfs):
        _write_dataframe(df, out_dir, stem)
        if i == 0 and stem != "final_cycle_binary_modeling_table":
            # Canonical tokenizer input — always written under this fixed name.
            _write_dataframe(df, out_dir, "final_cycle_binary_modeling_table")

    if optional_ok:
        for view, stem in cfg.optional_binary_table_views:
            if _view_exists(con, view):
                opt_df = con.execute(f"SELECT * FROM {view} {order}").df()
                _write_dataframe(opt_df, out_dir, stem)

    label_breakdown.to_csv(       out_dir / "row_level_label_breakdown.csv",        index=False)
    binary_label_breakdown.to_csv(out_dir / "row_level_binary_label_breakdown.csv", index=False)
    drug_class_breakdown.to_csv(  out_dir / "row_level_drug_class_breakdown.csv",   index=False)
    if preexisting_breakdown is not None:
        preexisting_breakdown.to_csv(out_dir / "row_level_preexisting_breakdown.csv", index=False)
    patient_level_labels.to_csv(  out_dir / "patient_level_labels.csv",             index=False)
    patient_level_summary.to_csv( out_dir / "patient_level_summary.csv",            index=False)
    cohort_accounting.to_csv(     out_dir / "cohort_accounting.csv",                index=False)

    print("Done.")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir",  required=True, type=Path, help="Path to MIMIC_IV_raw_data/")
    p.add_argument("--pipeline",  default="hf_cardiotox", choices=list(PIPELINE_REGISTRY),
                   help="Pipeline to run")
    p.add_argument("--name",      default=None, help="Output name under cohort_outputs/")
    a = p.parse_args()
    main(data_location=a.data_dir, pipeline=a.pipeline, output_name=a.name)
