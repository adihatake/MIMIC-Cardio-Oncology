"""
tokenize_cycle_sequences.py

Builds a tokenized EHR dataset for cycle-level cardiotoxicity prediction.

─────────────────────────────────────────────────────────────────────────────
USAGE
─────────────────────────────────────────────────────────────────────────────
Preferred — use the CLI wrapper which supports naming and optional split /
summary steps in one command:

    python tokenization_src/tokenize_cli.py --name ver1 --all
    python tokenization_src/tokenize_cli.py --cohort cycle_modeling_v3 \\
        --name ver2 --max-seq-len 512 --all

Run this script directly only if you want tokenization alone:

    python tokenization_src/tokenize_cycle_sequences.py

Programmatic (e.g. from a notebook or another script):

    from tokenization_src import tokenize_cycle_sequences as tok
    tok.main(
        cohort_name  = "cycle_modeling_v3",   # reads from cohort_outputs/<cohort_name>/
        output_name  = "ver2",                # writes to tokenization_outputs/<output_name>/
        max_seq_len  = 512,                   # truncate to this many tokens per sample
        data_dir     = Path("/path/to/MIMIC_IV_raw_data"),  # optional override
    )
    # All args are optional; omitting any keeps the module-level default.

─────────────────────────────────────────────────────────────────────────────
WHAT IT DOES
─────────────────────────────────────────────────────────────────────────────
Reads final_cycle_binary_modeling_table from cohort_outputs/<cohort_name>/.
For every eligible row (one per patient cycle), one sample is created whose
token sequence contains all EHR events that occurred strictly before that
cycle's prediction_time.  Cycles for the same patient are cumulative windows:

    sample 1 : events before prediction_time of cycle 1
    sample 2 : events before prediction_time of cycle 2  (larger window)
    sample 3 : …

Token format follows EHR_Event_Embedding in embedding_layers.py — five
parallel index sequences per sample:

    concept_ids   vocabulary index for each clinical event
    type_ids      event-type category (diagnosis / procedure / lab / medication)
    visit_ids     per-patient hospital-admission index (1-indexed, chronological)
    position_ids  0 … seq_len-1
    age_ids       decade bucket (0=0-9 yrs … 9=90+ yrs) at prediction_time

Sequences longer than MAX_SEQ_LEN are truncated to the most recent tokens
(oldest events dropped) before prepending [CLS].

─────────────────────────────────────────────────────────────────────────────
OUTPUTS  (tokenization_outputs/<output_name>/)
─────────────────────────────────────────────────────────────────────────────
    concept_ids.pt        (N, padded_len)  long tensor
    type_ids.pt           (N, padded_len)  long tensor
    visit_ids.pt          (N, padded_len)  long tensor
    position_ids.pt       (N, padded_len)  long tensor
    age_ids.pt            (N,)             long tensor
    attention_mask.pt     (N, padded_len)  bool tensor  (True = real token)
    labels.pt             (N,)             long tensor  (0 or 1)
    samples.parquet/.csv  per-sample metadata (subject_id, cycle_number, …)
    vocab.json            concept → token-id and event-type → type-id maps
    metadata.json         dataset-level config and statistics

─────────────────────────────────────────────────────────────────────────────
DEFAULTS  (override via main() kwargs or tokenize_cli.py args)
─────────────────────────────────────────────────────────────────────────────
    cohort_name  cycle_modeling_ver2       (cohort_outputs/<cohort_name>/)
    output_name  ver1                      (tokenization_outputs/<output_name>/)
    max_seq_len  600
    data_dir     MIMIC_IV_raw_data path hard-coded below
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

# ── configurable paths ────────────────────────────────────────────────────────
REPO_ROOT    = Path(__file__).resolve().parent.parent
MODELING_DIR = REPO_ROOT / "cohort_outputs" / "cycle_modeling_ver2"
OUTPUT_DIR   = REPO_ROOT / "tokenization_outputs" / "ver1"
MAX_SEQ_LEN  = 600
DATA_DIR: Path   # set by main() — no default; must be supplied via --data-dir
HOSP_DIR: Path   # set by main()
# ─────────────────────────────────────────────────────────────────────────────

TYPE_VOCAB: dict[str, int] = {
    "special":   0,
    "diagnosis": 1,
    "procedure": 2,
    "lab":       3,
    "medication": 4,
}

SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[CLS]", "[V_START]", "[V_END]"]

# ── CEHR-BERT time machinery ──────────────────────────────────────────────────
# Reference epoch for converting absolute event dates to integer day counts.
# Days since this date are passed to TimeEmbeddingLayer (see embedding_layers.py).
TIME_REFERENCE_DATE = pd.Timestamp("2000-01-01")

# Artificial Time Token (ATT) thresholds — CEHR-BERT CEHR_BERT mode:
#   < 28 days  → "W{floor(days/7)}"   (W0 … W3)
#   28–359 days → "M{floor(days/30)}"  (M1 … M11)
#   ≥ 360 days → "LT"
ATT_SPECIAL_TOKENS = (
    [f"W{i}" for i in range(4)]     # W0-W3  (0–3 weeks)
    + [f"M{i}" for i in range(1, 12)]  # M1-M11 (1–11 months)
    + ["LT"]                         # long-term (≥360 days)
)


def _days_to_att(days: int) -> str:
    """Map days-since-last-visit to an ATT token name (CEHR-BERT CEHR_BERT mode)."""
    if days < 28:
        return f"W{days // 7}"
    if days < 360:
        return f"M{days // 30}"
    return "LT"


# ── data loading ──────────────────────────────────────────────────────────────

def _load_modeling_table() -> pd.DataFrame:
    parquet = MODELING_DIR / "final_cycle_binary_modeling_table.parquet"
    df = pd.read_parquet(parquet) if parquet.exists() else pd.read_csv(
        MODELING_DIR / "final_cycle_binary_modeling_table.csv"
    )
    df["prediction_time"] = pd.to_datetime(df["prediction_time"], errors="coerce")
    df["cycle_number"]    = df["cycle_number"].astype(int)
    df["binary_label"]    = df["binary_label"].astype(int)
    df["subject_id"]      = df["subject_id"].astype(int)
    return df.sort_values(["subject_id", "cycle_number"]).reset_index(drop=True)


def _load_patients(subject_ids: set[int]) -> pd.DataFrame:
    t0 = time.time()
    df = pd.read_csv(HOSP_DIR / "patients.csv",
                     usecols=["subject_id", "anchor_age", "anchor_year"])
    df = df[df["subject_id"].isin(subject_ids)].copy()
    print(f"  patients.csv          → {len(df):,} rows ({time.time()-t0:.1f}s)")
    return df


def _load_admissions(subject_ids: set[int]) -> pd.DataFrame:
    t0 = time.time()
    df = pd.read_csv(
        HOSP_DIR / "admissions.csv",
        usecols=["subject_id", "hadm_id", "admittime", "dischtime"],
        parse_dates=["admittime", "dischtime"],
    )
    df = df[df["subject_id"].isin(subject_ids)].copy()
    df = df.sort_values(["subject_id", "admittime", "hadm_id"]).reset_index(drop=True)
    df["visit_id"] = df.groupby("subject_id").cumcount() + 1
    print(f"  admissions.csv        → {len(df):,} rows | {df['subject_id'].nunique():,} patients ({time.time()-t0:.1f}s)")
    return df


def _build_diagnoses(hadm_ids: set[int], admissions: pd.DataFrame) -> pd.DataFrame:
    t0 = time.time()
    dx = pd.read_csv(
        HOSP_DIR / "diagnoses_icd.csv",
        usecols=["subject_id", "hadm_id", "icd_code", "icd_version", "seq_num"],
    )
    dx = dx[dx["hadm_id"].isin(hadm_ids)].copy()
    print(f"  diagnoses_icd.csv     → {len(dx):,} rows ({time.time()-t0:.1f}s)")
    dx = dx.merge(
        admissions[["hadm_id", "admittime", "dischtime", "visit_id"]],
        on="hadm_id", how="left",
    )
    dx["event_time"]     = dx["dischtime"]   # anchored at discharge, end of visit
    dx["event_type"]     = "diagnosis"
    dx["concept_token"]  = (
        "diagnosis::" + dx["icd_code"].astype(str)
        + "_ICDver" + dx["icd_version"].astype(str)
    )
    dx["event_priority"] = dx["seq_num"]
    return dx[["subject_id", "hadm_id", "admittime", "event_time",
               "event_type", "concept_token", "event_priority", "visit_id"]]


def _build_procedures(hadm_ids: set[int], admissions: pd.DataFrame) -> pd.DataFrame:
    t0 = time.time()
    proc = pd.read_csv(
        HOSP_DIR / "procedures_icd.csv",
        usecols=["subject_id", "hadm_id", "icd_code", "icd_version", "seq_num", "chartdate"],
        parse_dates=["chartdate"],
    )
    proc = proc[proc["hadm_id"].isin(hadm_ids)].copy()
    print(f"  procedures_icd.csv    → {len(proc):,} rows ({time.time()-t0:.1f}s)")
    proc = proc.merge(
        admissions[["hadm_id", "admittime", "visit_id"]],
        on="hadm_id", how="left",
    )
    proc["event_time"]     = proc["chartdate"]
    proc["event_type"]     = "procedure"
    proc["concept_token"]  = (
        "procedure::" + proc["icd_code"].astype(str)
        + "_ICDver" + proc["icd_version"].astype(str)
    )
    proc["event_priority"] = proc["seq_num"]
    return proc[["subject_id", "hadm_id", "admittime", "event_time",
                 "event_type", "concept_token", "event_priority", "visit_id"]]


def _build_medications(hadm_ids: set[int], admissions: pd.DataFrame) -> pd.DataFrame:
    t0 = time.time()
    rx = pd.read_csv(
        HOSP_DIR / "prescriptions.csv",
        usecols=["subject_id", "hadm_id", "starttime", "drug"],
        parse_dates=["starttime"],
    )
    rx = rx[rx["hadm_id"].isin(hadm_ids)].dropna(subset=["starttime"]).copy()
    print(f"  prescriptions.csv     → {len(rx):,} rows ({time.time()-t0:.1f}s)")
    rx = rx.merge(
        admissions[["hadm_id", "admittime", "visit_id"]],
        on="hadm_id", how="left",
    )
    rx["event_time"]     = rx["starttime"]
    rx["event_type"]     = "medication"
    rx["concept_token"]  = "medication::" + rx["drug"].astype(str).str.lower().str.strip()
    rx["event_priority"] = rx.groupby(["subject_id", "hadm_id", "starttime"]).cumcount()
    return rx[["subject_id", "hadm_id", "admittime", "event_time",
               "event_type", "concept_token", "event_priority", "visit_id"]]


def _build_labs(subject_ids: set[int], hadm_ids: set[int],
                admissions: pd.DataFrame) -> pd.DataFrame:
    # Because of how massive (18GB) the lab events csv file is, DuckDB reads the 18 GB CSV in 
    # parallel and pushes the JOIN filters down into the scan, so only matching rows ever enter memory.
    t0 = time.time()
    file_path = str(HOSP_DIR / "labevents.csv")

    sid_df = pd.DataFrame({"subject_id": list(subject_ids)})
    hid_df = pd.DataFrame({"hadm_id":    list(hadm_ids)})

    con = duckdb.connect()
    con.register("sid_filter", sid_df)
    con.register("hid_filter", hid_df)
    lab = con.execute(f"""
        SELECT l.subject_id, l.hadm_id, l.itemid, l.storetime, l.flag
        FROM read_csv_auto('{file_path}', header = true) l
        INNER JOIN sid_filter USING (subject_id)
        INNER JOIN hid_filter USING (hadm_id)
        WHERE l.flag     IS NOT NULL
          AND l.storetime IS NOT NULL
    """).df()
    con.close()

    print(f"  labevents.csv         → {len(lab):,} rows ({time.time()-t0:.1f}s)")
    lab["storetime"] = pd.to_datetime(lab["storetime"], errors="coerce")
    lab = lab.dropna(subset=["storetime"])
    print(f"    {len(lab):,} abnormal inpatient lab rows retained for {lab['subject_id'].nunique():,} patients")

    lab = lab.merge(
        admissions[["hadm_id", "admittime", "visit_id"]],
        on="hadm_id", how="left",
    )
    lab["event_time"]     = lab["storetime"]
    lab["event_type"]     = "lab"
    lab["concept_token"]  = "lab::" + lab["itemid"].astype(str)
    lab["event_priority"] = 0
    return lab[["subject_id", "hadm_id", "admittime", "event_time",
                "event_type", "concept_token", "event_priority", "visit_id"]]


def build_master_events(subject_ids: set[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and merge all EHR event types into one chronologically sorted DataFrame."""
    print("Loading admissions...")
    admissions = _load_admissions(subject_ids)
    hadm_ids   = set(admissions["hadm_id"])

    print("Loading diagnoses...")
    dx   = _build_diagnoses(hadm_ids, admissions)
    print("Loading procedures...")
    proc = _build_procedures(hadm_ids, admissions)
    print("Loading medications (prescriptions)...")
    med  = _build_medications(hadm_ids, admissions)
    labs = _build_labs(subject_ids, hadm_ids, admissions)

    master = pd.concat([dx, proc, med, labs], ignore_index=True)
    master["admittime"]   = pd.to_datetime(master["admittime"], errors="coerce")
    master["event_time"]  = pd.to_datetime(master["event_time"], errors="coerce")
    master = master.dropna(subset=["event_time"])
    master["visit_id"]    = master["visit_id"].fillna(0).astype(int)
    master = master.sort_values(
        ["subject_id", "admittime", "hadm_id", "event_time", "event_type", "event_priority"],
        ascending=True, na_position="last",
    ).reset_index(drop=True)

    return master, admissions


# ── vocabulary ────────────────────────────────────────────────────────────────

def build_concept_vocab(all_tokens: list[str]) -> dict[str, int]:
    # ATT tokens are reserved slots right after SPECIAL_TOKENS so their indices
    # are stable regardless of which clinical concepts appear in a cohort.
    reserved = SPECIAL_TOKENS + ATT_SPECIAL_TOKENS
    unique = sorted(set(all_tokens) - set(reserved))
    return {tok: idx for idx, tok in enumerate(reserved + unique)}


# ── tokenization ──────────────────────────────────────────────────────────────

def _compute_age_years(subject_id: int, prediction_time: pd.Timestamp,
                       patients_df: pd.DataFrame) -> float:
    """Return continuous age in years at prediction_time (0.0 if patient unknown)."""
    row = patients_df.loc[patients_df["subject_id"] == subject_id]
    if row.empty:
        return 0.0
    anchor_age  = int(row.iloc[0]["anchor_age"])
    anchor_year = int(row.iloc[0]["anchor_year"])
    return float(max(0, anchor_age + (prediction_time.year - anchor_year)))


def _compute_age_id(subject_id: int, prediction_time: pd.Timestamp,
                    patients_df: pd.DataFrame) -> int:
    """Decade-bucket age for backward-compatible tokenizations (0–9)."""
    age = _compute_age_years(subject_id, prediction_time, patients_df)
    return max(0, min(9, int(age) // 10))


def tokenize_window(
    patient_events: pd.DataFrame,
    prediction_time: pd.Timestamp,
    concept_vocab: dict[str, int],
    insert_att: bool = False,
    insert_visit_delimiters: bool = False,
) -> dict:
    """
    Build token lists for one (patient, prediction_time) cumulative window.
    Events are filtered to strictly before prediction_time, then truncated to
    the most recent MAX_SEQ_LEN - 1 tokens before prepending [CLS].

    Always computes a `dates` field (days since TIME_REFERENCE_DATE per token)
    for use by the CEHR-BERT TimeEmbeddingLayer at training time.

    insert_att=True inserts CEHR-BERT ATT tokens (W0–W3, M1–M11, LT) between
    consecutive visits based on inter-visit gap in days.

    insert_visit_delimiters=True wraps each visit's events with [V_START]/[V_END],
    matching the BEHRT/CEHR-BERT sequence structure:
        [CLS] [V_START] e1 e2 [V_END] [ATT] [V_START] e3 [V_END] ...
    """
    window = patient_events[patient_events["event_time"] < prediction_time].copy()

    unk_id     = concept_vocab["[UNK]"]
    cls_id     = concept_vocab["[CLS]"]
    v_start_id = concept_vocab.get("[V_START]", unk_id)
    v_end_id   = concept_vocab.get("[V_END]",   unk_id)

    # Absolute date of each event as integer days since TIME_REFERENCE_DATE.
    raw_dates = (
        (window["event_time"] - TIME_REFERENCE_DATE)
        .dt.days
        .clip(lower=0)
        .astype(int)
        .tolist()
    )

    concept_ids = window["concept_token"].map(concept_vocab).fillna(unk_id).astype(int).tolist()
    type_ids    = window["event_type"].map(TYPE_VOCAB).fillna(TYPE_VOCAB["special"]).astype(int).tolist()
    visit_ids   = window["visit_id"].fillna(0).astype(int).tolist()

    if insert_att or insert_visit_delimiters:
        # Single pass: insert [V_START]/[V_END] around each visit block and
        # ATT tokens between visits.  Sequence per visit:
        #   [V_START] event... [V_END] [ATT]  (ATT after V_END, before next V_START)
        out_cids, out_tids, out_vids, out_dates = [], [], [], []
        prev_visit = None
        prev_date  = None

        for cid, tid, vid, date in zip(concept_ids, type_ids, visit_ids, raw_dates):
            if vid != prev_visit:
                if prev_visit is not None and prev_visit != 0:
                    # Close previous visit
                    if insert_visit_delimiters:
                        out_cids.append(v_end_id);  out_tids.append(TYPE_VOCAB["special"])
                        out_vids.append(prev_visit); out_dates.append(prev_date)
                    # ATT between visits
                    if insert_att and vid != 0:
                        gap_days = max(0, date - prev_date)
                        att_id   = concept_vocab.get(_days_to_att(gap_days), unk_id)
                        out_cids.append(att_id);  out_tids.append(TYPE_VOCAB["special"])
                        out_vids.append(vid);     out_dates.append(date)
                # Open new visit
                if insert_visit_delimiters and vid != 0:
                    out_cids.append(v_start_id); out_tids.append(TYPE_VOCAB["special"])
                    out_vids.append(vid);         out_dates.append(date)

            out_cids.append(cid);  out_tids.append(tid)
            out_vids.append(vid);  out_dates.append(date)
            prev_visit = vid
            prev_date  = date

        # Close the final visit
        if prev_visit is not None and prev_visit != 0 and insert_visit_delimiters:
            out_cids.append(v_end_id);  out_tids.append(TYPE_VOCAB["special"])
            out_vids.append(prev_visit); out_dates.append(prev_date)

        concept_ids = out_cids
        type_ids    = out_tids
        visit_ids   = out_vids
        raw_dates   = out_dates

    budget = MAX_SEQ_LEN - 1
    raw_seq_len = len(concept_ids) + 1  # +1 for CLS, before truncation
    concept_ids = concept_ids[-budget:]
    type_ids    = type_ids[-budget:]
    visit_ids   = visit_ids[-budget:]
    raw_dates   = raw_dates[-budget:]

    cls_date = max(0, (prediction_time - TIME_REFERENCE_DATE).days)
    concept_ids  = [cls_id] + concept_ids
    type_ids     = [TYPE_VOCAB["special"]] + type_ids
    visit_ids    = [0] + visit_ids
    dates        = [cls_date] + raw_dates
    position_ids = list(range(len(concept_ids)))

    return {
        "concept_ids":  concept_ids,
        "type_ids":     type_ids,
        "visit_ids":    visit_ids,
        "position_ids": position_ids,
        "dates":        dates,
        "seq_len":      len(concept_ids),
        "raw_seq_len":  raw_seq_len,
    }


# ── main ──────────────────────────────────────────────────────────────────────

def main(
    data_dir: Path,
    cohort_name: str | None = None,
    output_name: str | None = None,
    max_seq_len: int | None = None,
    insert_att: bool = False,
    insert_visit_delimiters: bool = False,
) -> None:
    global MAX_SEQ_LEN, MODELING_DIR, OUTPUT_DIR, DATA_DIR, HOSP_DIR
    DATA_DIR = data_dir
    HOSP_DIR = data_dir / "mimic-iv-3.1" / "hosp"
    if cohort_name is not None:
        MODELING_DIR = REPO_ROOT / "cohort_outputs" / cohort_name
    if output_name is not None:
        OUTPUT_DIR = REPO_ROOT / "tokenization_outputs" / output_name
    if max_seq_len is not None:
        MAX_SEQ_LEN = max_seq_len

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading binary modeling table...")
    modeling_df  = _load_modeling_table()
    subject_ids  = set(modeling_df["subject_id"])
    print(f"  {len(subject_ids):,} patients  |  {len(modeling_df):,} cycle samples")

    print("Loading patient demographics...")
    patients_df = _load_patients(subject_ids)

    print("Building master EHR event timeline...")
    master_df, _ = build_master_events(subject_ids)
    print(f"  {len(master_df):,} total events loaded")

    print("Building concept vocabulary...")
    all_tokens   = master_df["concept_token"].dropna().unique().tolist()
    concept_vocab = build_concept_vocab(all_tokens)
    print(f"  {len(concept_vocab):,} unique concepts (including special tokens)")

    # Index patient events for fast per-sample lookup
    patient_events_map: dict[int, pd.DataFrame] = {
        sid: grp for sid, grp in master_df.groupby("subject_id")
    }

    print("Tokenizing samples...")
    samples_meta:   list[dict] = []
    token_sequences: list[dict] = []

    for _, row in tqdm(modeling_df.iterrows(), total=len(modeling_df), unit="sample"):
        sid       = int(row["subject_id"])
        cycle_num = int(row["cycle_number"])
        pred_time = pd.Timestamp(row["prediction_time"])
        label     = int(row["binary_label"])

        events    = patient_events_map.get(sid, pd.DataFrame())
        tok       = tokenize_window(events, pred_time, concept_vocab,
                                    insert_att=insert_att,
                                    insert_visit_delimiters=insert_visit_delimiters)
        age_id    = _compute_age_id(sid, pred_time, patients_df)
        age_years = _compute_age_years(sid, pred_time, patients_df)

        samples_meta.append({
            "subject_id":      sid,
            "cycle_number":    cycle_num,
            "prediction_time": pred_time,
            "binary_label":    label,
            "age_id":          age_id,
            "age_years":       age_years,
            "raw_seq_len":     tok["raw_seq_len"],
            "seq_len":         tok["seq_len"],
        })
        token_sequences.append({**tok, "age_id": age_id, "age_years": age_years, "label": label})

    print("Padding sequences and building tensors...")
    pad_id = concept_vocab["[PAD]"]

    concept_ids_t   = pad_sequence(
        [torch.tensor(s["concept_ids"],  dtype=torch.long) for s in token_sequences],
        batch_first=True, padding_value=pad_id,
    )
    type_ids_t      = pad_sequence(
        [torch.tensor(s["type_ids"],     dtype=torch.long) for s in token_sequences],
        batch_first=True, padding_value=TYPE_VOCAB["special"],
    )
    visit_ids_t     = pad_sequence(
        [torch.tensor(s["visit_ids"],    dtype=torch.long) for s in token_sequences],
        batch_first=True, padding_value=0,
    )
    position_ids_t  = pad_sequence(
        [torch.tensor(s["position_ids"], dtype=torch.long) for s in token_sequences],
        batch_first=True, padding_value=0,
    )
    # dates: days since TIME_REFERENCE_DATE per token, for CEHR-BERT TimeEmbeddingLayer.
    # Padding positions receive 0 (same as CLS to be benign; masked out in attention anyway).
    dates_t = pad_sequence(
        [torch.tensor(s["dates"], dtype=torch.long) for s in token_sequences],
        batch_first=True, padding_value=0,
    )
    attention_mask  = concept_ids_t != pad_id
    age_ids_t       = torch.tensor([s["age_id"]    for s in token_sequences], dtype=torch.long)
    age_years_t     = torch.tensor([s["age_years"] for s in token_sequences], dtype=torch.float32)
    labels_t        = torch.tensor([s["label"]     for s in token_sequences], dtype=torch.long)

    print("Saving outputs...")
    torch.save(concept_ids_t,  OUTPUT_DIR / "concept_ids.pt")
    torch.save(type_ids_t,     OUTPUT_DIR / "type_ids.pt")
    torch.save(visit_ids_t,    OUTPUT_DIR / "visit_ids.pt")
    torch.save(position_ids_t, OUTPUT_DIR / "position_ids.pt")
    torch.save(dates_t,        OUTPUT_DIR / "dates.pt")
    torch.save(attention_mask, OUTPUT_DIR / "attention_mask.pt")
    torch.save(age_ids_t,      OUTPUT_DIR / "age_ids.pt")
    torch.save(age_years_t,    OUTPUT_DIR / "age_years.pt")
    torch.save(labels_t,       OUTPUT_DIR / "labels.pt")

    samples_df = pd.DataFrame(samples_meta)
    samples_df.to_parquet(OUTPUT_DIR / "samples.parquet", index=False)
    samples_df.to_csv(OUTPUT_DIR / "samples.csv", index=False)

    with open(OUTPUT_DIR / "vocab.json", "w") as f:
        json.dump({"concept_vocab": concept_vocab, "type_vocab": TYPE_VOCAB}, f, indent=2)

    seq_lens = [s["seq_len"] for s in token_sequences]
    metadata = {
        "version":              "ver1",
        "max_seq_len":          MAX_SEQ_LEN,
        "n_samples":            len(token_sequences),
        "n_patients":           len(subject_ids),
        "vocab_size":           len(concept_vocab),
        "n_positive":           int(labels_t.sum().item()),
        "n_negative":           int((labels_t == 0).sum().item()),
        "positive_rate":        float(labels_t.float().mean().item()),
        "mean_seq_len":         float(np.mean(seq_lens)),
        "median_seq_len":       float(np.median(seq_lens)),
        "std_seq_len":          float(np.std(seq_lens)),
        "min_seq_len":          int(min(seq_lens)),
        "max_seq_len_observed": int(max(seq_lens)),
        "n_truncated":          int(sum(l == MAX_SEQ_LEN for l in seq_lens)),
        "tensor_shape":         list(concept_ids_t.shape),
        "modeling_dir":         str(MODELING_DIR),
        "data_dir":             str(DATA_DIR),
        # CEHR-BERT temporal fields
        "has_dates":              True,
        "has_age_years":          True,
        "time_reference_date":    str(TIME_REFERENCE_DATE.date()),
        "insert_att":             insert_att,
        "insert_visit_delimiters": insert_visit_delimiters,
    }
    with open(OUTPUT_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n{'─'*55}")
    print(f"  Saved to: {OUTPUT_DIR}")
    print(f"  Samples:       {metadata['n_samples']:>8,}")
    print(f"  Patients:      {metadata['n_patients']:>8,}")
    print(f"  Vocab size:    {metadata['vocab_size']:>8,}")
    print(f"  Tensor shape:  {metadata['tensor_shape']}")
    print(f"  Positive:      {metadata['n_positive']:>8,}  ({metadata['positive_rate']:.1%})")
    print(f"  Negative:      {metadata['n_negative']:>8,}")
    print(f"  Mean seq len:  {metadata['mean_seq_len']:>8.1f}")
    print(f"  Truncated:     {metadata['n_truncated']:>8,}")
    print(f"{'─'*55}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir",   required=True, type=Path, help="Path to MIMIC_IV_raw_data/")
    p.add_argument("--cohort",     default=None,             help="Cohort name under cohort_outputs/")
    p.add_argument("--name",       default=None,             help="Output name under tokenization_outputs/")
    p.add_argument("--max-seq-len",default=None, type=int,   help="Max token sequence length")
    p.add_argument("--insert-att",              action="store_true", help="Insert CEHR-BERT ATT tokens between visits")
    p.add_argument("--insert-visit-delimiters", action="store_true", help="Wrap each visit with [V_START]/[V_END] tokens")
    a = p.parse_args()
    main(data_dir=a.data_dir, cohort_name=a.cohort, output_name=a.name,
         max_seq_len=a.max_seq_len, insert_att=a.insert_att,
         insert_visit_delimiters=a.insert_visit_delimiters)
