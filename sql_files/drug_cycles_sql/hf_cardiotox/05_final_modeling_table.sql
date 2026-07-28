-- 05_final_modeling_table.sql
--
-- Cycle-level modeling table for HF-specific cardiotoxicity prediction.
-- Outcome: incident heart failure (ICD-10 I50%, ICD-9 428%) within monitoring window.
--
-- Two binary modeling tables are produced:
--
--   hf_final_binary_modeling_table_strict
--     Excludes patients with pre-existing HF (I50%/428%) OR cardiomyopathy (I42%/425%)
--     before cohort entry. Cleaner drug-attribution; smaller N.
--
--   hf_final_binary_modeling_table_inclusive
--     No pre-existing cardiac exclusion. Includes patients with prior HF or CMP.
--     Larger N; HF events may reflect worsening of pre-existing disease rather than
--     de novo drug-induced cardiotoxicity.
--
-- Main outputs:
--   hf_final_cycle_modeling_table
--   hf_final_binary_modeling_table_strict
--   hf_final_binary_modeling_table_inclusive

CREATE OR REPLACE VIEW hf_final_cycle_modeling_table AS
SELECT
    e.subject_id,
    e.drug_class,
    e.cycle_number,
    e.prediction_time,
    e.cycle_start_date,
    e.cycle_end_date,
    e.n_exposure_start_days_in_cycle,
    e.n_prescription_rows_in_cycle,
    e.drugs_in_cycle,
    e.toxicity_window_days,
    e.prediction_time + e.toxicity_window_days * INTERVAL '1 day' AS prediction_window_end,
    e.window_rationale,

    -- Incident HF details (NULL if no HF event occurred after cohort entry)
    hf.hf_event_time,
    hf.hf_hadm_id,
    hf.hf_icd_codes,

    -- Pre-existing cardiac history flags (from pre_existing_hf_or_cmp view in file 03)
    COALESCE(pre.has_pre_existing_hf,      0) AS has_pre_existing_hf,
    COALESCE(pre.has_pre_existing_cmp,     0) AS has_pre_existing_cmp,
    COALESCE(pre.has_pre_existing_hf_or_cmp, 0) AS has_pre_existing_hf_or_cmp,

    -- Death information (see 04_observation_and_death.sql for handling options)
    d.death_time,

    -- Last observation time (for follow-up completeness)
    obs.last_observation_time,

    -- Was the full monitoring window covered by available data?
    CASE
        WHEN obs.last_observation_time >= e.prediction_time + e.toxicity_window_days * INTERVAL '1 day'
        THEN 1 ELSE 0
    END AS observed_through_prediction_window,

    -- Did the patient die inside the monitoring window before any HF event?
    CASE
        WHEN d.death_time IS NOT NULL
         AND d.death_time >  e.prediction_time
         AND d.death_time <= e.prediction_time + e.toxicity_window_days * INTERVAL '1 day'
         AND (hf.hf_event_time IS NULL OR hf.hf_event_time > d.death_time)
        THEN 1 ELSE 0
    END AS died_in_window_before_hf,

    -- ── Label ──────────────────────────────────────────────────────────────────
    CASE
        -- HF already occurred before this cycle's prediction time — exclude from binary table
        WHEN hf.hf_event_time IS NOT NULL
         AND hf.hf_event_time <= e.prediction_time
        THEN 'exclude_already_toxic'

        WHEN hf.hf_event_time >  e.prediction_time
         AND hf.hf_event_time <= e.prediction_time + e.toxicity_window_days * INTERVAL '1 day'
        THEN 'positive'

        WHEN d.death_time IS NOT NULL
         AND d.death_time >  e.prediction_time
         AND d.death_time <= e.prediction_time + e.toxicity_window_days * INTERVAL '1 day'
         AND (hf.hf_event_time IS NULL OR hf.hf_event_time > d.death_time)
        THEN 'censored_death'

        WHEN obs.last_observation_time >= e.prediction_time + e.toxicity_window_days * INTERVAL '1 day'
        THEN 'negative_observed'

        ELSE 'unknown_insufficient_followup'
    END AS label,

    -- ── Binary label ───────────────────────────────────────────────────────────
    -- 1 = incident HF in window, 0 = survived window without HF, NULL = indeterminate
    -- 'censored_death' and 'exclude_already_toxic' rows receive NULL.
    CASE
        WHEN hf.hf_event_time IS NOT NULL
         AND hf.hf_event_time <= e.prediction_time
        THEN NULL
        WHEN hf.hf_event_time >  e.prediction_time
         AND hf.hf_event_time <= e.prediction_time + e.toxicity_window_days * INTERVAL '1 day'
        THEN 1
        WHEN d.death_time IS NOT NULL
         AND d.death_time >  e.prediction_time
         AND d.death_time <= e.prediction_time + e.toxicity_window_days * INTERVAL '1 day'
         AND (hf.hf_event_time IS NULL OR hf.hf_event_time > d.death_time)
        THEN NULL
        WHEN obs.last_observation_time >= e.prediction_time + e.toxicity_window_days * INTERVAL '1 day'
        THEN 0
        ELSE NULL
    END AS binary_label,

    -- ── Eligibility: strict ────────────────────────────────────────────────────
    -- Requires: determinable label AND no pre-existing HF or CMP AND not already toxic
    CASE
        WHEN hf.hf_event_time IS NOT NULL
         AND hf.hf_event_time <= e.prediction_time                                          THEN 0
        WHEN COALESCE(pre.has_pre_existing_hf_or_cmp, 0) = 1                               THEN 0
        WHEN hf.hf_event_time >  e.prediction_time
         AND hf.hf_event_time <= e.prediction_time + e.toxicity_window_days * INTERVAL '1 day' THEN 1
        WHEN d.death_time IS NOT NULL
         AND d.death_time >  e.prediction_time
         AND d.death_time <= e.prediction_time + e.toxicity_window_days * INTERVAL '1 day'
         AND (hf.hf_event_time IS NULL OR hf.hf_event_time > d.death_time)                 THEN 0
        WHEN obs.last_observation_time >= e.prediction_time + e.toxicity_window_days * INTERVAL '1 day' THEN 1
        ELSE 0
    END AS eligible_strict,

    -- ── Eligibility: inclusive ─────────────────────────────────────────────────
    -- Requires: determinable label AND not already toxic; pre-existing HF/CMP kept
    CASE
        WHEN hf.hf_event_time IS NOT NULL
         AND hf.hf_event_time <= e.prediction_time                                          THEN 0
        WHEN hf.hf_event_time >  e.prediction_time
         AND hf.hf_event_time <= e.prediction_time + e.toxicity_window_days * INTERVAL '1 day' THEN 1
        WHEN d.death_time IS NOT NULL
         AND d.death_time >  e.prediction_time
         AND d.death_time <= e.prediction_time + e.toxicity_window_days * INTERVAL '1 day'
         AND (hf.hf_event_time IS NULL OR hf.hf_event_time > d.death_time)                 THEN 0
        WHEN obs.last_observation_time >= e.prediction_time + e.toxicity_window_days * INTERVAL '1 day' THEN 1
        ELSE 0
    END AS eligible_inclusive

FROM hf_cycle_exposures e
LEFT JOIN incident_hf_events        hf  ON e.subject_id = hf.subject_id
LEFT JOIN pre_existing_hf_or_cmp    pre ON e.subject_id = pre.subject_id
LEFT JOIN patient_death_hf          d   ON e.subject_id = d.subject_id
LEFT JOIN hf_patient_last_observation obs ON e.subject_id = obs.subject_id;

-- ── Dataset 1: Strict ──────────────────────────────────────────────────────────
-- Excludes patients with pre-existing HF (I50%/428%) or cardiomyopathy (I42%/425%).
-- Use this for cleaner drug-attribution analysis.
CREATE OR REPLACE VIEW hf_final_binary_modeling_table_strict AS
SELECT *
FROM hf_final_cycle_modeling_table
WHERE eligible_strict = 1
  AND label IN ('positive', 'negative_observed');

-- ── Dataset 2: Inclusive ───────────────────────────────────────────────────────
-- Retains patients with pre-existing HF or CMP; no cardiac history exclusion.
-- Use this for analyses that treat prior cardiac disease as a covariate,
-- or to assess drug effects on top of existing cardiac vulnerability.
CREATE OR REPLACE VIEW hf_final_binary_modeling_table_inclusive AS
SELECT *
FROM hf_final_cycle_modeling_table
WHERE eligible_inclusive = 1
  AND label IN ('positive', 'negative_observed');
