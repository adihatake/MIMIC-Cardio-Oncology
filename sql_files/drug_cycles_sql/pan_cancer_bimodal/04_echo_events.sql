-- 04_echo_events.sql
--
-- Echo-based cardiotoxicity detection using ESC 2022 CTRCD thresholds:
--   (1) LVEF CTRCD:      decrease >= 10 percentage points from a normal baseline (>=50%) to <50%
--   (2) GLS subclinical: >15% relative decrease from baseline GLS
--
-- Both thresholds follow the ESC 2022 Cardio-Oncology Guidelines:
--   doi:10.1093/eurheartj/ehac244, Table 4
--
-- These events are combined with ICD-coded HF in 05_combined_cardiotox_event.sql.
-- The echo endpoint detects subclinical dysfunction earlier than HF admission;
-- GLS worsening typically precedes LVEF decline, which precedes symptomatic HF.
--
-- ── GLS SIGN CONVENTION NOTE ──────────────────────────────────────────────────
-- GLS is stored in mimic-iv-echo as a NEGATIVE percentage (e.g., -18.5 for -18.5%).
-- Normal GLS: typically between -16% and -22% (more negative = better function).
-- Worsening: GLS becomes less negative (e.g., -20% -> -14%).
-- A >15% relative decrease means the strain magnitude falls by >15% of baseline:
--   relative_decrease = (follow_up_gls - baseline_gls) / ABS(baseline_gls)
--   e.g., (-14 - (-20)) / |-20| = 6/20 = 0.30 -> 30% decrease (>15% threshold)
--
-- REQUIRED DATA CHECK — run before any GLS analysis:
--   SELECT COUNT(*), MIN(TRY_CAST(result AS DOUBLE)), MAX(TRY_CAST(result AS DOUBLE))
--   FROM read_csv_auto('mimic-iv-echo/structured-measurement.csv')
--   WHERE LOWER(measurement) LIKE '%gls%' OR LOWER(measurement) LIKE '%longitudinal%';
--
-- If MIN > 0 (all positive): GLS stored as positive in this build.
--   Remove the "< 0" filter in all_gls_hf and update gls_cardiotox_events:
--     (baseline_gls - follow_up_gls) / baseline_gls >= 0.15
-- ─────────────────────────────────────────────────────────────────────────────
--
-- Main outputs:
--   all_lvef_hf / all_gls_hf            (all echo measurements for cohort patients)
--   baseline_lvef_hf / baseline_gls_hf  (last pre-drug measurement within lookback window)
--   lvef_cardiotox_events               (LVEF CTRCD events after drug start)
--   gls_cardiotox_events                (GLS subclinical events after drug start)
--   first_echo_cardiotox_event          (earliest LVEF or GLS event per patient)

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
-- LIKE patterns are broad; narrow after verifying measurement names in your data.
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
-- ESC 2022 requires a NORMAL baseline LVEF (>=50%) to classify LVEF-based CTRCD.
-- Patients without a normal baseline LVEF cannot meet the LVEF CTRCD criterion.
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
             l.lvef_value           DESC NULLS LAST
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
-- All qualifying post-drug follow-up measurements are kept; first_echo_cardiotox_event picks earliest.
CREATE OR REPLACE VIEW lvef_cardiotox_events AS
SELECT
    b.subject_id,
    f.measurement_datetime AS event_time,
    'lvef_ctrcd'           AS event_type,
    b.baseline_lvef,
    f.lvef_value           AS event_lvef,
    b.baseline_lvef - f.lvef_value AS lvef_drop_pp,
    NULL::DOUBLE           AS baseline_gls,
    NULL::DOUBLE           AS event_gls,
    NULL::DOUBLE           AS gls_relative_decrease
FROM baseline_lvef_hf b
JOIN all_lvef_hf f
    ON b.subject_id = f.subject_id
WHERE b.baseline_lvef  IS NOT NULL
  AND b.baseline_lvef  >= 50
  AND f.measurement_datetime > b.first_drug_time
  AND (b.baseline_lvef - f.lvef_value) >= 10
  AND f.lvef_value < 50;

-- ── GLS cardiotoxicity events ─────────────────────────────────────────────────
-- Criterion: >15% relative decrease from baseline GLS.
-- ESC 2022: marker of subclinical cardiotoxicity (Stage 1 CTRCD when asymptomatic,
-- no biomarker rise). GLS worsening typically precedes LVEF decline.
-- Formula (negative convention): relative_decrease = (follow_up - baseline) / |baseline|
-- e.g., baseline -20%, follow_up -14%: (-14 - (-20)) / |-20| = 0.30 -> 30% decrease
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
  AND b.baseline_gls < 0
  AND f.measurement_datetime > b.first_drug_time
  AND (f.gls_value - b.baseline_gls) / ABS(b.baseline_gls) > 0.15;

-- ── First echo cardiotoxicity event per patient ───────────────────────────────
-- Combines LVEF CTRCD and GLS subclinical events; picks the earliest event.
-- On tie: LVEF-based event takes priority (more severe/clinically certain).
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
             CASE event_type WHEN 'lvef_ctrcd' THEN 0 ELSE 1 END  ASC,
             lvef_drop_pp                                          DESC NULLS LAST,
             gls_relative_decrease                                 DESC NULLS LAST
) = 1;
