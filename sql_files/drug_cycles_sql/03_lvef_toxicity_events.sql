-- 03_lvef_toxicity_events.sql
--
-- Build LVEF-based cardiotoxicity events.
--
-- Main output:
--   lvef_toxicity_events

CREATE OR REPLACE VIEW all_lvef AS
SELECT
    subject_id,
    CAST(measurement_datetime AS TIMESTAMP) AS measurement_datetime,
    TRY_CAST(result AS DOUBLE) AS lvef_value
FROM read_csv_auto('mimic-iv-echo/structured-measurement.csv')
WHERE measurement = 'lvef'
  AND result IS NOT NULL
  AND TRY_CAST(result AS DOUBLE) IS NOT NULL;

CREATE OR REPLACE VIEW baseline_lvef_pre_first_drug AS
SELECT
    c.subject_id,
    c.first_oncology_time,
    a.measurement_datetime AS baseline_time,
    a.lvef_value AS baseline_lvef
FROM cancer_first_drug c
LEFT JOIN all_lvef a
    ON c.subject_id = a.subject_id
   AND a.measurement_datetime < c.first_oncology_time
   AND a.measurement_datetime >= c.first_oncology_time - (SELECT baseline_lookback_days FROM cycle_label_params) * INTERVAL '1 day'
QUALIFY ROW_NUMBER() OVER (
    PARTITION BY c.subject_id
    ORDER BY a.measurement_datetime DESC NULLS LAST
) = 1;

CREATE OR REPLACE VIEW lvef_toxicity_events AS
SELECT
    b.subject_id,
    f.measurement_datetime AS toxicity_time,
    'lvef_ctrctd' AS toxicity_type,
    b.baseline_time,
    b.baseline_lvef,
    f.lvef_value AS event_lvef,
    b.baseline_lvef - f.lvef_value AS absolute_lvef_drop
FROM baseline_lvef_pre_first_drug b
JOIN all_lvef f
    ON b.subject_id = f.subject_id
WHERE b.baseline_lvef IS NOT NULL
  AND f.measurement_datetime >= b.first_oncology_time
  AND (b.baseline_lvef - f.lvef_value) >= 10
  AND f.lvef_value < 50;
