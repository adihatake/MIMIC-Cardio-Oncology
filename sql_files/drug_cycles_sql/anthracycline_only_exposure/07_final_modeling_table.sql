-- 07_final_modeling_table.sql  (anthracycline_only_exposure)
--
-- Anthracycline-only cycle modeling table.
--
-- Two-stage restriction:
--
--   (1) anthracycline_first_patients
--       Keeps only patients whose FIRST cardiotoxic drug class was an anthracycline.
--       Any patient who received HER2-targeted or ICI drugs before their first
--       anthracycline dose is excluded — prior cardiotoxic exposure is a confounder
--       for attributing HF/CTRCD to anthracyclines.
--       NOTE: patients who started another class on the SAME DAY as their first
--       anthracycline dose are retained (strictly-prior filter). Add AND <= instead
--       of AND < in the subquery below for a stricter same-day exclusion.
--
--   (2) drug_class = 'anthracycline' rows only
--       The cycle backbone (hf_cycle_exposures) tracks all three drug classes.
--       Only anthracycline cycles are carried forward into the final table.
--
-- The has_concurrent_other_class flag marks cycles where another cardiotoxic
-- drug class was also active (same calendar window). Use this for sensitivity
-- analyses that further restrict to purely isolated anthracycline exposure.
--
-- Outcome: combined CTRCD endpoint from first_combined_cardiotox_event (file 05)
--   = earliest of ICD HF admission, LVEF CTRCD, or GLS subclinical event.
--   Monitoring window: 365 days (anthracycline ESC 2022 window from 00_parameters.sql).
--
-- Main outputs:
--   anthracycline_first_patients
--   anthracycline_cycle_exposures
--   anthracycline_final_cycle_modeling_table
--   anthracycline_final_binary_modeling_table_strict
--   anthracycline_final_binary_modeling_table_inclusive

-- ── Stage 1: patients whose first cardiotoxic drug was an anthracycline ────────
-- Excludes anyone who received HER2-targeted or ICI before their first anthracycline.
CREATE OR REPLACE VIEW anthracycline_first_patients AS
SELECT a.subject_id
FROM hf_patient_first_drug_per_class a
WHERE a.drug_class = 'anthracycline'
  AND NOT EXISTS (
      SELECT 1
      FROM hf_patient_first_drug_per_class o
      WHERE o.subject_id          =  a.subject_id
        AND o.drug_class          != 'anthracycline'
        AND o.first_class_drug_time < a.first_class_drug_time   -- strictly prior
  );

-- ── Stage 2: anthracycline cycle exposures for qualifying patients ─────────────
CREATE OR REPLACE VIEW anthracycline_cycle_exposures AS
SELECT e.*
FROM hf_cycle_exposures e
INNER JOIN anthracycline_first_patients af ON e.subject_id = af.subject_id
WHERE e.drug_class = 'anthracycline';

-- ── Full cycle modeling table ──────────────────────────────────────────────────
CREATE OR REPLACE VIEW anthracycline_final_cycle_modeling_table AS
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

    -- ── Combined CTRCD event ───────────────────────────────────────────────────
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

    -- ── Pre-existing cardiac history ───────────────────────────────────────────
    COALESCE(pre.has_pre_existing_hf,        0) AS has_pre_existing_hf,
    COALESCE(pre.has_pre_existing_cmp,       0) AS has_pre_existing_cmp,
    COALESCE(pre.has_pre_existing_hf_or_cmp, 0) AS has_pre_existing_hf_or_cmp,

    -- ── Concurrent exposure flag ───────────────────────────────────────────────
    -- 1 if another cardiotoxic drug class was administered during this cycle window.
    -- Use WHERE has_concurrent_other_class = 0 for the purest anthracycline-only analysis.
    CASE WHEN EXISTS (
        SELECT 1
        FROM hf_cohort_drug_starts d
        WHERE d.subject_id = e.subject_id
          AND d.drug_class != 'anthracycline'
          AND CAST(d.starttime AS DATE) BETWEEN e.cycle_start_date AND e.cycle_end_date
    ) THEN 1 ELSE 0 END AS has_concurrent_other_class,

    -- ── Death and observation ──────────────────────────────────────────────────
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

FROM anthracycline_cycle_exposures e
LEFT JOIN first_combined_cardiotox_event  ev  ON e.subject_id = ev.subject_id
LEFT JOIN pre_existing_hf_or_cmp          pre ON e.subject_id = pre.subject_id
LEFT JOIN patient_death_hf                d   ON e.subject_id = d.subject_id
LEFT JOIN hf_patient_last_observation     obs ON e.subject_id = obs.subject_id;

-- ── Binary dataset 1: Strict ───────────────────────────────────────────────────
-- Excludes pre-existing HF or CMP. Use for de novo anthracycline cardiotoxicity.
-- Further restrict with WHERE has_concurrent_other_class = 0 for pure isolation.
CREATE OR REPLACE VIEW anthracycline_final_binary_modeling_table_strict AS
SELECT *
FROM anthracycline_final_cycle_modeling_table
WHERE eligible_strict = 1
  AND label IN ('positive', 'negative_observed');

-- ── Binary dataset 2: Inclusive ────────────────────────────────────────────────
-- Retains patients with pre-existing HF or CMP.
CREATE OR REPLACE VIEW anthracycline_final_binary_modeling_table_inclusive AS
SELECT *
FROM anthracycline_final_cycle_modeling_table
WHERE eligible_inclusive = 1
  AND label IN ('positive', 'negative_observed');
