-- 07_final_modeling_table.sql
--
-- Cycle-level modeling table for the combined CTRCD endpoint.
--
-- Outcome (positive): any cardiotoxicity event within the per-class monitoring window:
--   (1) Heart failure admission  — ICD-10 I50%, ICD-9 428% (file 03)
--   (2) LVEF CTRCD               — decrease >=10 pp from normal baseline (>=50%) to <50% (file 04)
--   (3) GLS subclinical CTRCD    — >15% relative decrease from baseline GLS (file 04)
--
-- Event timing comes from first_combined_cardiotox_event (file 05).
-- For patients without echo data, the endpoint reduces to ICD HF admission only.
--
-- Two binary datasets:
--
--   ctrcd_final_binary_modeling_table_strict
--     Excludes patients with pre-existing HF (I50%/428%) OR cardiomyopathy (I42%/425%)
--     before cohort entry. Cleanest drug-attribution; smallest N.
--
--   ctrcd_final_binary_modeling_table_inclusive
--     No pre-existing cardiac exclusion. Larger N; HF/echo events may reflect worsening
--     of pre-existing disease rather than de novo drug-induced cardiotoxicity.
--
-- Main outputs:
--   ctrcd_final_cycle_modeling_table
--   ctrcd_final_binary_modeling_table_strict
--   ctrcd_final_binary_modeling_table_inclusive

CREATE OR REPLACE VIEW ctrcd_final_cycle_modeling_table AS
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

    -- ── Combined CTRCD event (earliest of ICD HF, LVEF CTRCD, GLS subclinical) ──
    ev.first_event_time,
    ev.first_event_type,

    -- ICD HF details (NULL if no HF admission)
    ev.hf_event_time,
    ev.hf_hadm_id,
    ev.hf_icd_codes,

    -- Echo CTRCD details (NULL if no echo event or no echo data)
    ev.first_echo_event_time,
    ev.first_echo_event_type,
    ev.baseline_lvef,
    ev.event_lvef,
    ev.lvef_drop_pp,
    ev.baseline_gls,
    ev.event_gls,
    ev.gls_relative_decrease,

    -- ── Pre-existing cardiac history (from file 03) ────────────────────────────
    COALESCE(pre.has_pre_existing_hf,        0) AS has_pre_existing_hf,
    COALESCE(pre.has_pre_existing_cmp,       0) AS has_pre_existing_cmp,
    COALESCE(pre.has_pre_existing_hf_or_cmp, 0) AS has_pre_existing_hf_or_cmp,

    -- ── Death and observation (from file 06) ───────────────────────────────────
    d.death_time,
    obs.last_observation_time,

    CASE
        WHEN obs.last_observation_time >= e.prediction_time + e.toxicity_window_days * INTERVAL '1 day'
        THEN 1 ELSE 0
    END AS observed_through_prediction_window,

    CASE
        WHEN d.death_time IS NOT NULL
         AND d.death_time >  e.prediction_time
         AND d.death_time <= e.prediction_time + e.toxicity_window_days * INTERVAL '1 day'
         AND (ev.first_event_time IS NULL OR ev.first_event_time > d.death_time)
        THEN 1 ELSE 0
    END AS died_in_window_before_event,

    -- ── Label ──────────────────────────────────────────────────────────────────
    -- exclude_already_toxic : any CTRCD event occurred at or before prediction_time
    -- positive              : first CTRCD event falls within the monitoring window
    -- censored_death        : patient died inside the window before any CTRCD event
    -- negative_observed     : patient observed through full window, no CTRCD event
    -- unknown_insufficient_followup : follow-up ended before the window closed
    CASE
        WHEN ev.first_event_time IS NOT NULL
         AND ev.first_event_time <= e.prediction_time
        THEN 'exclude_already_toxic'

        WHEN ev.first_event_time >  e.prediction_time
         AND ev.first_event_time <= e.prediction_time + e.toxicity_window_days * INTERVAL '1 day'
        THEN 'positive'

        WHEN d.death_time IS NOT NULL
         AND d.death_time >  e.prediction_time
         AND d.death_time <= e.prediction_time + e.toxicity_window_days * INTERVAL '1 day'
         AND (ev.first_event_time IS NULL OR ev.first_event_time > d.death_time)
        THEN 'censored_death'

        WHEN obs.last_observation_time >= e.prediction_time + e.toxicity_window_days * INTERVAL '1 day'
        THEN 'negative_observed'

        ELSE 'unknown_insufficient_followup'
    END AS label,

    -- ── Binary label ───────────────────────────────────────────────────────────
    -- 1 = CTRCD in window, 0 = survived window without CTRCD, NULL = indeterminate
    CASE
        WHEN ev.first_event_time IS NOT NULL
         AND ev.first_event_time <= e.prediction_time
        THEN NULL
        WHEN ev.first_event_time >  e.prediction_time
         AND ev.first_event_time <= e.prediction_time + e.toxicity_window_days * INTERVAL '1 day'
        THEN 1
        WHEN d.death_time IS NOT NULL
         AND d.death_time >  e.prediction_time
         AND d.death_time <= e.prediction_time + e.toxicity_window_days * INTERVAL '1 day'
         AND (ev.first_event_time IS NULL OR ev.first_event_time > d.death_time)
        THEN NULL
        WHEN obs.last_observation_time >= e.prediction_time + e.toxicity_window_days * INTERVAL '1 day'
        THEN 0
        ELSE NULL
    END AS binary_label,

    -- ── Eligibility: strict ────────────────────────────────────────────────────
    -- Requires determinable label AND no pre-existing HF or CMP AND not already toxic
    CASE
        WHEN ev.first_event_time IS NOT NULL
         AND ev.first_event_time <= e.prediction_time                                              THEN 0
        WHEN COALESCE(pre.has_pre_existing_hf_or_cmp, 0) = 1                                      THEN 0
        WHEN ev.first_event_time >  e.prediction_time
         AND ev.first_event_time <= e.prediction_time + e.toxicity_window_days * INTERVAL '1 day' THEN 1
        WHEN d.death_time IS NOT NULL
         AND d.death_time >  e.prediction_time
         AND d.death_time <= e.prediction_time + e.toxicity_window_days * INTERVAL '1 day'
         AND (ev.first_event_time IS NULL OR ev.first_event_time > d.death_time)                   THEN 0
        WHEN obs.last_observation_time >= e.prediction_time + e.toxicity_window_days * INTERVAL '1 day' THEN 1
        ELSE 0
    END AS eligible_strict,

    -- ── Eligibility: inclusive ─────────────────────────────────────────────────
    -- Requires determinable label AND not already toxic; pre-existing HF/CMP kept
    CASE
        WHEN ev.first_event_time IS NOT NULL
         AND ev.first_event_time <= e.prediction_time                                              THEN 0
        WHEN ev.first_event_time >  e.prediction_time
         AND ev.first_event_time <= e.prediction_time + e.toxicity_window_days * INTERVAL '1 day' THEN 1
        WHEN d.death_time IS NOT NULL
         AND d.death_time >  e.prediction_time
         AND d.death_time <= e.prediction_time + e.toxicity_window_days * INTERVAL '1 day'
         AND (ev.first_event_time IS NULL OR ev.first_event_time > d.death_time)                   THEN 0
        WHEN obs.last_observation_time >= e.prediction_time + e.toxicity_window_days * INTERVAL '1 day' THEN 1
        ELSE 0
    END AS eligible_inclusive

FROM hf_cycle_exposures e
LEFT JOIN first_combined_cardiotox_event  ev  ON e.subject_id = ev.subject_id
LEFT JOIN pre_existing_hf_or_cmp          pre ON e.subject_id = pre.subject_id
LEFT JOIN patient_death_hf                d   ON e.subject_id = d.subject_id
LEFT JOIN hf_patient_last_observation     obs ON e.subject_id = obs.subject_id;

-- ── Dataset 1: Strict ──────────────────────────────────────────────────────────
-- Excludes patients with pre-existing HF (I50%/428%) or cardiomyopathy (I42%/425%).
-- Use for cleaner drug-attribution and de novo cardiotoxicity analysis.
CREATE OR REPLACE VIEW ctrcd_final_binary_modeling_table_strict AS
SELECT *
FROM ctrcd_final_cycle_modeling_table
WHERE eligible_strict = 1
  AND label IN ('positive', 'negative_observed');

-- ── Dataset 2: Inclusive ───────────────────────────────────────────────────────
-- Retains patients with pre-existing HF or CMP; no cardiac history exclusion.
-- Use for analyses treating prior cardiac disease as a covariate, or to assess
-- drug effects on top of existing cardiac vulnerability.
CREATE OR REPLACE VIEW ctrcd_final_binary_modeling_table_inclusive AS
SELECT *
FROM ctrcd_final_cycle_modeling_table
WHERE eligible_inclusive = 1
  AND label IN ('positive', 'negative_observed');
