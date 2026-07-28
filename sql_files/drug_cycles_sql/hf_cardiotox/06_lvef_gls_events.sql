-- 06_lvef_gls_events.sql
--
-- Echo-based cardiotoxicity definitions:
--   (1) LVEF: decrease >10 percentage points from baseline to a value <50%
--   (2) GLS:  >15% relative decrease from baseline
--
-- Both thresholds follow the ESC 2022 Cardio-Oncology Guidelines definition of
-- Cancer Therapy-Related Cardiac Dysfunction (CTRCD):
--   doi:10.1093/eurheartj/ehac244, Table 4
--
-- This file is an ALTERNATIVE outcome to the ICD-based HF endpoint in file 03.
-- Echo-based cardiotoxicity detects subclinical dysfunction earlier than HF admission;
-- GLS changes typically precede LVEF decline, which precedes symptomatic HF.
--
-- ── GLS SIGN CONVENTION NOTE ──────────────────────────────────────────────────
-- GLS is stored in mimic-iv-echo as a NEGATIVE percentage (e.g., -18.5 for -18.5%).
-- Normal GLS: typically between -16% and -22% (more negative = better function).
-- Worsening: GLS becomes less negative (e.g., -20% → -14%).
-- A >15% relative decrease means the strain magnitude falls by >15% of baseline:
--   relative_decrease = (follow_up_gls - baseline_gls) / ABS(baseline_gls)
--   e.g., (-14 - (-20)) / |-20| = 6/20 = 0.30 → 30% decrease (>15% threshold) ✓
--
-- REQUIRED DATA CHECK — run before any GLS analysis:
--   SELECT COUNT(*), MIN(TRY_CAST(result AS DOUBLE)), MAX(TRY_CAST(result AS DOUBLE))
--   FROM read_csv_auto('mimic-iv-echo/structured-measurement.csv')
--   WHERE LOWER(measurement) LIKE '%gls%' OR LOWER(measurement) LIKE '%longitudinal%';
--
-- If COUNT(*) = 0: GLS measurements absent in this MIMIC build — skip GLS analysis.
-- If MIN > 0 (all positive): GLS is stored as positive values in this build.
--   Flip the sign filter in all_gls_hf (remove "< 0") and update gls_cardiotox_events to:
--     (baseline_gls - follow_up_gls) / baseline_gls >= 0.15
-- If all_gls_hf returns 0 rows after running 02_cycle_exposures.sql, this is the likely cause.
-- ─────────────────────────────────────────────────────────────────────────────
--
-- Main outputs:
--   all_lvef_hf / all_gls_hf            (all echo measurements for cohort patients)
--   baseline_lvef_hf / baseline_gls_hf  (last pre-drug measurement within lookback window)
--   lvef_cardiotox_events               (LVEF drop >=10pp to <50% after drug start)
--   gls_cardiotox_events                (GLS relative decrease >15% after drug start)
--   first_echo_cardiotox_event          (earliest LVEF or GLS event per patient)
--   echo_cardiotox_final_binary_modeling_table_strict
--   echo_cardiotox_final_binary_modeling_table_inclusive

-- ── All LVEF measurements for cohort patients ─────────────────────────────────
CREATE OR REPLACE VIEW all_lvef_hf AS
SELECT
    subject_id,
    CAST(measurement_datetime AS TIMESTAMP) AS measurement_datetime,
    TRY_CAST(result AS DOUBLE) AS lvef_value
FROM read_csv_auto('mimic-iv-echo/structured-measurement.csv')
WHERE LOWER(measurement) = 'lvef'
  AND result IS NOT NULL
  AND TRY_CAST(result AS DOUBLE) IS NOT NULL
  AND subject_id IN (SELECT subject_id FROM hf_patient_first_drug);

-- ── All GLS measurements for cohort patients ──────────────────────────────────
-- GLS may appear under different measurement names depending on the echo system.
-- The LIKE patterns below are intentionally broad; narrow after verifying your data.
CREATE OR REPLACE VIEW all_gls_hf AS
SELECT
    subject_id,
    CAST(measurement_datetime AS TIMESTAMP) AS measurement_datetime,
    TRY_CAST(result AS DOUBLE) AS gls_value
FROM read_csv_auto('mimic-iv-echo/structured-measurement.csv')
WHERE (
    LOWER(measurement) LIKE '%gls%'
    OR LOWER(measurement) LIKE '%global longitudinal%'
    OR LOWER(measurement) LIKE '%global_longitudinal%'
)
  AND result IS NOT NULL
  AND TRY_CAST(result AS DOUBLE) IS NOT NULL
  AND TRY_CAST(result AS DOUBLE) < 0   -- enforce negative convention; remove if stored as positive
  AND subject_id IN (SELECT subject_id FROM hf_patient_first_drug);

-- ── Baseline LVEF ─────────────────────────────────────────────────────────────
-- Last LVEF within baseline_lookback_days before first drug.
-- ESC 2022 requires a NORMAL baseline LVEF (>=50%) to define CTRCD. Patients without
-- a normal baseline LVEF cannot be classified for LVEF-based CTRCD.
-- QUALIFY tiebreaker: latest datetime first; on tie, highest LVEF (most optimistic).
CREATE OR REPLACE VIEW baseline_lvef_hf AS
SELECT
    p.subject_id,
    p.first_drug_time,
    l.measurement_datetime AS baseline_lvef_time,
    l.lvef_value           AS baseline_lvef
FROM hf_patient_first_drug p
LEFT JOIN all_lvef_hf l
    ON p.subject_id = l.subject_id
   AND l.measurement_datetime <  p.first_drug_time
   AND l.measurement_datetime >= p.first_drug_time - (SELECT baseline_lookback_days FROM hf_cohort_params) * INTERVAL '1 day'
QUALIFY ROW_NUMBER() OVER (
    PARTITION BY p.subject_id
    ORDER BY l.measurement_datetime DESC NULLS LAST,
             l.lvef_value           DESC NULLS LAST  -- stable tiebreaker
) = 1;

-- ── Baseline GLS ──────────────────────────────────────────────────────────────
-- Last GLS within baseline_lookback_days before first drug.
-- No mandatory normal-baseline cutoff is defined for GLS in ESC 2022 (unlike LVEF >=50%).
-- QUALIFY tiebreaker: latest datetime first; on tie, most negative GLS (best function).
CREATE OR REPLACE VIEW baseline_gls_hf AS
SELECT
    p.subject_id,
    p.first_drug_time,
    g.measurement_datetime AS baseline_gls_time,
    g.gls_value            AS baseline_gls
FROM hf_patient_first_drug p
LEFT JOIN all_gls_hf g
    ON p.subject_id = g.subject_id
   AND g.measurement_datetime <  p.first_drug_time
   AND g.measurement_datetime >= p.first_drug_time - (SELECT baseline_lookback_days FROM hf_cohort_params) * INTERVAL '1 day'
QUALIFY ROW_NUMBER() OVER (
    PARTITION BY p.subject_id
    ORDER BY g.measurement_datetime DESC NULLS LAST,
             g.gls_value            ASC  NULLS LAST  -- most negative = best function on tie
) = 1;

-- ── LVEF cardiotoxicity events ────────────────────────────────────────────────
-- Criterion: LVEF drop >= 10 percentage points from a normal baseline (>=50%) to <50%.
-- ESC 2022 CTRCD definition: "new decrease in LVEF of >=10 percentage points to a
-- value of <50%" (symptomatic or asymptomatic).
-- All qualifying follow-up LVEF measurements after drug start are kept here;
-- the first_echo_cardiotox_event view below picks the earliest.
CREATE OR REPLACE VIEW lvef_cardiotox_events AS
SELECT
    b.subject_id,
    f.measurement_datetime AS event_time,
    'lvef_ctrcd'           AS event_type,
    b.baseline_lvef,
    f.lvef_value           AS event_lvef,
    b.baseline_lvef - f.lvef_value AS lvef_drop_pp,  -- absolute drop in percentage points
    NULL::DOUBLE           AS baseline_gls,
    NULL::DOUBLE           AS event_gls,
    NULL::DOUBLE           AS gls_relative_decrease
FROM baseline_lvef_hf b
JOIN all_lvef_hf f
    ON b.subject_id = f.subject_id
WHERE b.baseline_lvef  IS NOT NULL
  AND b.baseline_lvef  >= 50                                  -- normal baseline required
  AND f.measurement_datetime > b.first_drug_time              -- post-drug only
  AND (b.baseline_lvef - f.lvef_value) >= 10                  -- drop >= 10 pp
  AND f.lvef_value < 50;                                      -- below 50% threshold

-- ── GLS cardiotoxicity events ─────────────────────────────────────────────────
-- Criterion: >15% relative decrease from baseline GLS.
-- ESC 2022: "new relative percentage decrease of GLS >15% from baseline" is a marker
-- of subclinical cardiotoxicity (Stage 1 CTRCD when asymptomatic, no biomarker rise).
-- GLS worsening typically precedes LVEF decline; early detection enables intervention.
--
-- Formula (negative convention): relative_decrease = (follow_up - baseline) / |baseline|
-- e.g., baseline -20%, follow_up -14%: (−14 − (−20)) / |−20| = 0.30 → 30% decrease ✓
CREATE OR REPLACE VIEW gls_cardiotox_events AS
SELECT
    b.subject_id,
    f.measurement_datetime AS event_time,
    'gls_subclinical'      AS event_type,
    NULL::DOUBLE           AS baseline_lvef,
    NULL::DOUBLE           AS event_lvef,
    NULL::DOUBLE           AS lvef_drop_pp,
    b.baseline_gls,
    f.gls_value            AS event_gls,
    (f.gls_value - b.baseline_gls) / ABS(b.baseline_gls) AS gls_relative_decrease
FROM baseline_gls_hf b
JOIN all_gls_hf f
    ON b.subject_id = f.subject_id
WHERE b.baseline_gls IS NOT NULL
  AND b.baseline_gls < 0                                      -- confirm negative convention
  AND f.measurement_datetime > b.first_drug_time              -- post-drug only
  AND (f.gls_value - b.baseline_gls) / ABS(b.baseline_gls) > 0.15;  -- >15% relative decrease

-- ── First echo cardiotoxicity event per patient ───────────────────────────────
-- Combines LVEF CTRCD and GLS subclinical events; picks the earliest event.
-- QUALIFY tiebreaker: on same event_time, LVEF-based event takes priority (more severe),
-- then smallest gls_relative_decrease (most conservative GLS call on tie).
CREATE OR REPLACE VIEW first_echo_cardiotox_event AS
SELECT
    subject_id,
    event_time         AS first_echo_event_time,
    event_type         AS first_echo_event_type,
    baseline_lvef,
    event_lvef,
    lvef_drop_pp,
    baseline_gls,
    event_gls,
    gls_relative_decrease
FROM (
    SELECT * FROM lvef_cardiotox_events
    UNION ALL
    SELECT * FROM gls_cardiotox_events
)
QUALIFY ROW_NUMBER() OVER (
    PARTITION BY subject_id
    ORDER BY event_time                                            ASC,
             CASE event_type WHEN 'lvef_ctrcd' THEN 0 ELSE 1 END  ASC,  -- LVEF takes priority over GLS
             lvef_drop_pp                                          DESC NULLS LAST,
             gls_relative_decrease                                 DESC NULLS LAST
) = 1;

-- ── Echo cardiotoxicity final modeling tables ─────────────────────────────────
-- These reuse the same cycle backbone (hf_cycle_exposures) and observation/death
-- views from files 02 and 04, but swap the outcome to echo-based CTRCD.

CREATE OR REPLACE VIEW echo_cardiotox_final_cycle_modeling_table AS
SELECT
    e.subject_id,
    e.drug_class,
    e.cycle_number,
    e.prediction_time,
    e.cycle_start_date,
    e.cycle_end_date,
    e.drugs_in_cycle,
    e.toxicity_window_days,
    e.prediction_time + e.toxicity_window_days * INTERVAL '1 day' AS prediction_window_end,

    -- Echo cardiotoxicity event details
    ev.first_echo_event_time,
    ev.first_echo_event_type,
    ev.baseline_lvef,
    ev.event_lvef,
    ev.lvef_drop_pp,
    ev.baseline_gls,
    ev.event_gls,
    ev.gls_relative_decrease,

    -- Pre-existing cardiac flags (reused from file 03)
    COALESCE(pre.has_pre_existing_hf,        0) AS has_pre_existing_hf,
    COALESCE(pre.has_pre_existing_cmp,       0) AS has_pre_existing_cmp,
    COALESCE(pre.has_pre_existing_hf_or_cmp, 0) AS has_pre_existing_hf_or_cmp,

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
         AND (ev.first_echo_event_time IS NULL OR ev.first_echo_event_time > d.death_time)
        THEN 1 ELSE 0
    END AS died_in_window_before_echo_event,

    CASE
        -- Echo cardiotox already occurred before this cycle's prediction time
        WHEN ev.first_echo_event_time IS NOT NULL
         AND ev.first_echo_event_time <= e.prediction_time
        THEN 'exclude_already_toxic'

        WHEN ev.first_echo_event_time >  e.prediction_time
         AND ev.first_echo_event_time <= e.prediction_time + e.toxicity_window_days * INTERVAL '1 day'
        THEN 'positive'
        WHEN d.death_time IS NOT NULL
         AND d.death_time >  e.prediction_time
         AND d.death_time <= e.prediction_time + e.toxicity_window_days * INTERVAL '1 day'
         AND (ev.first_echo_event_time IS NULL OR ev.first_echo_event_time > d.death_time)
        THEN 'censored_death'
        WHEN obs.last_observation_time >= e.prediction_time + e.toxicity_window_days * INTERVAL '1 day'
        THEN 'negative_observed'
        ELSE 'unknown_insufficient_followup'
    END AS label,

    CASE
        WHEN ev.first_echo_event_time IS NOT NULL
         AND ev.first_echo_event_time <= e.prediction_time
        THEN NULL
        WHEN ev.first_echo_event_time >  e.prediction_time
         AND ev.first_echo_event_time <= e.prediction_time + e.toxicity_window_days * INTERVAL '1 day'
        THEN 1
        WHEN d.death_time IS NOT NULL
         AND d.death_time >  e.prediction_time
         AND d.death_time <= e.prediction_time + e.toxicity_window_days * INTERVAL '1 day'
         AND (ev.first_echo_event_time IS NULL OR ev.first_echo_event_time > d.death_time)
        THEN NULL
        WHEN obs.last_observation_time >= e.prediction_time + e.toxicity_window_days * INTERVAL '1 day'
        THEN 0
        ELSE NULL
    END AS binary_label,

    -- Strict eligibility: no pre-existing HF or CMP AND not already toxic
    CASE
        WHEN ev.first_echo_event_time IS NOT NULL
         AND ev.first_echo_event_time <= e.prediction_time                                  THEN 0
        WHEN COALESCE(pre.has_pre_existing_hf_or_cmp, 0) = 1                               THEN 0
        WHEN ev.first_echo_event_time >  e.prediction_time
         AND ev.first_echo_event_time <= e.prediction_time + e.toxicity_window_days * INTERVAL '1 day' THEN 1
        WHEN d.death_time IS NOT NULL
         AND d.death_time >  e.prediction_time
         AND d.death_time <= e.prediction_time + e.toxicity_window_days * INTERVAL '1 day'
         AND (ev.first_echo_event_time IS NULL OR ev.first_echo_event_time > d.death_time) THEN 0
        WHEN obs.last_observation_time >= e.prediction_time + e.toxicity_window_days * INTERVAL '1 day' THEN 1
        ELSE 0
    END AS eligible_strict,

    -- Inclusive eligibility: not already toxic; pre-existing HF/CMP kept
    CASE
        WHEN ev.first_echo_event_time IS NOT NULL
         AND ev.first_echo_event_time <= e.prediction_time                                  THEN 0
        WHEN ev.first_echo_event_time >  e.prediction_time
         AND ev.first_echo_event_time <= e.prediction_time + e.toxicity_window_days * INTERVAL '1 day' THEN 1
        WHEN d.death_time IS NOT NULL
         AND d.death_time >  e.prediction_time
         AND d.death_time <= e.prediction_time + e.toxicity_window_days * INTERVAL '1 day'
         AND (ev.first_echo_event_time IS NULL OR ev.first_echo_event_time > d.death_time) THEN 0
        WHEN obs.last_observation_time >= e.prediction_time + e.toxicity_window_days * INTERVAL '1 day' THEN 1
        ELSE 0
    END AS eligible_inclusive

FROM hf_cycle_exposures e
LEFT JOIN first_echo_cardiotox_event  ev  ON e.subject_id = ev.subject_id
LEFT JOIN pre_existing_hf_or_cmp      pre ON e.subject_id = pre.subject_id
LEFT JOIN patient_death_hf            d   ON e.subject_id = d.subject_id
LEFT JOIN hf_patient_last_observation obs ON e.subject_id = obs.subject_id;

-- Dataset 1 (echo): Strict — excludes pre-existing HF or CMP
CREATE OR REPLACE VIEW echo_cardiotox_binary_modeling_table_strict AS
SELECT *
FROM echo_cardiotox_final_cycle_modeling_table
WHERE eligible_strict = 1
  AND label IN ('positive', 'negative_observed');

-- Dataset 2 (echo): Inclusive — no pre-existing cardiac exclusion
CREATE OR REPLACE VIEW echo_cardiotox_binary_modeling_table_inclusive AS
SELECT *
FROM echo_cardiotox_final_cycle_modeling_table
WHERE eligible_inclusive = 1
  AND label IN ('positive', 'negative_observed');
