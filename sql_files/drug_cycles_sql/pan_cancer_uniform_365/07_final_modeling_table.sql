-- 07_final_modeling_table.sql  (pan_cancer_uniform_365)
--
-- Final cycle-level modeling table — one row per (patient, cycle_number).
--
-- CTRCD endpoint (positive = any of):
--   (1) ICD HF admission     — I50% / 428%                       (file 03)
--   (2) LVEF CTRCD           — drop >=10pp from >=50% baseline   (file 04)
--   (3) GLS subclinical      — >15% relative decrease            (file 04)
--
-- Toxicity window: uniform 365 days from prediction_time for ALL cycles.
-- Unlike pan_cancer_per_cycle (class-specific ESC 2022 windows), this pipeline
-- uses a single horizon appropriate for risk stratification — the question being
-- "will this patient have a cardiac event in the next year?" rather than
-- attributing events to specific drug classes.
--
-- Two binary datasets:
--
--   pcu_final_binary_modeling_table_inclusive
--     No pre-existing cardiac exclusion. Largest N; events may reflect worsening
--     of pre-existing HF/CMP rather than de novo drug-induced cardiotoxicity.
--     USE FIRST (canonical training table written as final_cycle_binary_modeling_table).
--
--   pcu_final_binary_modeling_table_strict
--     Excludes patients with pre-existing HF (I50%/428%) or CMP (I42%/425%).
--     Cleanest drug-attribution; smallest N. Use for etiological analyses.
--
-- Label values:
--   positive                     : CTRCD event within 365-day monitoring window
--   negative_observed            : observed through full 365-day window, no CTRCD
--   censored_death               : died in window before CTRCD (competing event)
--   exclude_already_toxic        : CTRCD event at or before prediction_time
--   unknown_insufficient_followup: follow-up ended before window closed
--
-- Main outputs:
--   pcu_final_cycle_modeling_table
--   pcu_final_binary_modeling_table_inclusive   (written first = canonical)
--   pcu_final_binary_modeling_table_strict

CREATE OR REPLACE VIEW pcu_final_cycle_modeling_table AS
SELECT
    e.subject_id,
    e.cycle_number,
    e.prediction_time,
    e.cycle_start_date,
    e.cycle_end_date,
    e.n_exposure_start_days_in_cycle,
    e.n_prescription_rows_in_cycle,
    e.drugs_in_cycle,
    e.drug_classes_in_cycle,
    e.toxicity_window_days,
    e.prediction_time + e.toxicity_window_days * INTERVAL '1 day' AS prediction_window_end,

    -- Drug-class binary exposure flags
    e.exposed_anthracycline,
    e.exposed_her2_targeted,
    e.exposed_immune_checkpoint_inhibitor,
    e.exposed_taxane,
    e.exposed_fluoropyrimidine,
    e.exposed_vegf_inhibitor,
    e.exposed_egfr_inhibitor,
    e.exposed_tyrosine_kinase_inhibitor,
    e.exposed_proteasome_inhibitor,
    e.exposed_immunomodulatory_agent,

    -- Combined CTRCD event details
    ev.first_event_time,
    ev.first_event_type,
    ev.hf_event_time,
    ev.hf_hadm_id,
    ev.hf_icd_codes,
    ev.first_echo_event_time,
    ev.first_echo_event_type,
    ev.baseline_lvef,
    ev.event_lvef,
    ev.lvef_drop_pp,
    ev.baseline_gls,
    ev.event_gls,
    ev.gls_relative_decrease,

    -- Pre-existing cardiac history
    COALESCE(pre.has_pre_existing_hf,        0) AS has_pre_existing_hf,
    COALESCE(pre.has_pre_existing_cmp,       0) AS has_pre_existing_cmp,
    COALESCE(pre.has_pre_existing_hf_or_cmp, 0) AS has_pre_existing_hf_or_cmp,

    -- Death and observation
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

    -- Label
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

    -- Binary label
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

    -- Eligibility: strict (exclude pre-existing HF/CMP)
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

    -- Eligibility: inclusive (retain pre-existing HF/CMP patients)
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

FROM pcu_cycle_exposures                      e
LEFT JOIN pcu_first_combined_cardiotox_event  ev  ON e.subject_id = ev.subject_id
LEFT JOIN pcu_pre_existing_hf_or_cmp         pre ON e.subject_id = pre.subject_id
LEFT JOIN pcu_patient_death                  d   ON e.subject_id = d.subject_id
LEFT JOIN pcu_patient_last_observation       obs ON e.subject_id = obs.subject_id;

-- ── Dataset 1: Inclusive ──────────────────────────────────────────────────────
-- Written FIRST so the pipeline runner copies it as the canonical
-- final_cycle_binary_modeling_table (largest usable N for prediction models).
CREATE OR REPLACE VIEW pcu_final_binary_modeling_table_inclusive AS
SELECT *
FROM pcu_final_cycle_modeling_table
WHERE eligible_inclusive = 1
  AND label IN ('positive', 'negative_observed');

-- ── Dataset 2: Strict ─────────────────────────────────────────────────────────
-- Excludes pre-existing HF / CMP. Use for de novo cardiotoxicity analyses.
CREATE OR REPLACE VIEW pcu_final_binary_modeling_table_strict AS
SELECT *
FROM pcu_final_cycle_modeling_table
WHERE eligible_strict = 1
  AND label IN ('positive', 'negative_observed');
