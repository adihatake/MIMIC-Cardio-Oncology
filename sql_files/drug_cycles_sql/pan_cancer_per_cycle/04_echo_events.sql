-- 04_echo_events.sql  (pan_cancer_per_cycle)
--
-- Echo-based cardiotoxicity detection using ESC 2022 CTRCD thresholds:
--   (1) LVEF CTRCD:      decrease >= 10 percentage points from a normal baseline (>=50%) to <50%
--   (2) GLS subclinical: >15% relative decrease from baseline GLS
--
-- Both thresholds follow the ESC 2022 Cardio-Oncology Guidelines:
--   doi:10.1093/eurheartj/ehac244, Table 4
--
-- GLS SIGN CONVENTION:
--   GLS is stored as a NEGATIVE percentage (e.g., -18.5 for -18.5%).
--   Normal GLS: between -16% and -22% (more negative = better function).
--   Worsening: GLS becomes less negative (e.g., -20% -> -14%).
--   relative_decrease = (follow_up_gls - baseline_gls) / ABS(baseline_gls)
--   e.g., (-14 - (-20)) / |-20| = 0.30 -> 30% decrease (>15% threshold = subclinical)
--
--   If your MIMIC-IV echo data stores GLS as positive, remove the `< 0` filter and
--   update the gls_relative_decrease formula to (baseline_gls - follow_up_gls) / baseline_gls.
--
-- Main outputs:
--   pcr_all_lvef                  (all LVEF measurements for cohort patients)
--   pcr_all_gls                   (all GLS measurements for cohort patients)
--   pcr_baseline_lvef             (last pre-drug LVEF within baseline_lookback_days)
--   pcr_baseline_gls              (last pre-drug GLS within baseline_lookback_days)
--   pcr_lvef_cardiotox_events     (LVEF CTRCD events post-drug)
--   pcr_gls_cardiotox_events      (GLS subclinical events post-drug)
--   pcr_first_echo_cardiotox_event (earliest LVEF or GLS event per patient)

-- All LVEF measurements for cohort patients
CREATE OR REPLACE VIEW pcr_all_lvef AS
SELECT
    subject_id,
    CAST(measurement_datetime AS TIMESTAMP) AS measurement_datetime,
    TRY_CAST(result AS DOUBLE) AS lvef_value
FROM read_csv_auto('mimic-iv-echo/structured-measurement.csv')
WHERE LOWER(measurement) = 'lvef'
  AND result IS NOT NULL
  AND TRY_CAST(result AS DOUBLE) IS NOT NULL
  AND subject_id IN (SELECT subject_id FROM pcr_patient_first_drug);

-- All GLS measurements for cohort patients (negative convention enforced)
CREATE OR REPLACE VIEW pcr_all_gls AS
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
  AND TRY_CAST(result AS DOUBLE) < 0   -- negative convention; remove if stored as positive
  AND subject_id IN (SELECT subject_id FROM pcr_patient_first_drug);

-- Baseline LVEF: last measurement within baseline_lookback_days before first drug.
-- ESC 2022 requires a NORMAL baseline LVEF (>=50%) for LVEF-based CTRCD classification.
CREATE OR REPLACE VIEW pcr_baseline_lvef AS
SELECT
    p.subject_id,
    p.first_drug_time,
    l.measurement_datetime AS baseline_lvef_time,
    l.lvef_value           AS baseline_lvef
FROM pcr_patient_first_drug p
LEFT JOIN pcr_all_lvef l
    ON p.subject_id = l.subject_id
   AND l.measurement_datetime <  p.first_drug_time
   AND l.measurement_datetime >= p.first_drug_time - (SELECT baseline_lookback_days FROM pcr_params) * INTERVAL '1 day'
QUALIFY ROW_NUMBER() OVER (
    PARTITION BY p.subject_id
    ORDER BY l.measurement_datetime DESC NULLS LAST,
             l.lvef_value           DESC NULLS LAST
) = 1;

-- Baseline GLS: last measurement within baseline_lookback_days before first drug.
CREATE OR REPLACE VIEW pcr_baseline_gls AS
SELECT
    p.subject_id,
    p.first_drug_time,
    g.measurement_datetime AS baseline_gls_time,
    g.gls_value            AS baseline_gls
FROM pcr_patient_first_drug p
LEFT JOIN pcr_all_gls g
    ON p.subject_id = g.subject_id
   AND g.measurement_datetime <  p.first_drug_time
   AND g.measurement_datetime >= p.first_drug_time - (SELECT baseline_lookback_days FROM pcr_params) * INTERVAL '1 day'
QUALIFY ROW_NUMBER() OVER (
    PARTITION BY p.subject_id
    ORDER BY g.measurement_datetime DESC NULLS LAST,
             g.gls_value            ASC  NULLS LAST
) = 1;

-- LVEF CTRCD events: drop >= 10pp from normal (>=50%) baseline to <50%
CREATE OR REPLACE VIEW pcr_lvef_cardiotox_events AS
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
FROM pcr_baseline_lvef b
JOIN pcr_all_lvef f
    ON b.subject_id = f.subject_id
WHERE b.baseline_lvef  IS NOT NULL
  AND b.baseline_lvef  >= 50
  AND f.measurement_datetime > b.first_drug_time
  AND (b.baseline_lvef - f.lvef_value) >= 10
  AND f.lvef_value < 50;

-- GLS subclinical events: >15% relative decrease from baseline GLS
CREATE OR REPLACE VIEW pcr_gls_cardiotox_events AS
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
FROM pcr_baseline_gls b
JOIN pcr_all_gls f
    ON b.subject_id = f.subject_id
WHERE b.baseline_gls IS NOT NULL
  AND b.baseline_gls < 0
  AND f.measurement_datetime > b.first_drug_time
  AND (f.gls_value - b.baseline_gls) / ABS(b.baseline_gls) > 0.15;

-- Earliest echo CTRCD event per patient (LVEF or GLS); LVEF takes priority on ties
CREATE OR REPLACE VIEW pcr_first_echo_cardiotox_event AS
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
    SELECT * FROM pcr_lvef_cardiotox_events
    UNION ALL
    SELECT * FROM pcr_gls_cardiotox_events
)
QUALIFY ROW_NUMBER() OVER (
    PARTITION BY subject_id
    ORDER BY event_time                                            ASC,
             CASE event_type WHEN 'lvef_ctrcd' THEN 0 ELSE 1 END  ASC,
             lvef_drop_pp                                          DESC NULLS LAST,
             gls_relative_decrease                                 DESC NULLS LAST
) = 1;
