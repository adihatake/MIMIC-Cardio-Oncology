-- 02_cycle_exposures.sql
--
-- Collapse drug administration rows into treatment cycles, tracked independently
-- per drug class. A gap of > cycle_gap_days between consecutive administration
-- start dates within the same drug class marks the beginning of a new cycle.
--
-- Three drug classes are tracked separately, each with its own cycle numbering
-- and its own clinical monitoring window (from 00_parameters.sql):
--   anthracycline               → 365-day window
--   her2_targeted               → 365-day window
--   immune_checkpoint_inhibitor → 90-day window
--
-- Main output:
--   hf_cycle_exposures  (one row per subject_id / drug_class / cycle_number)

CREATE OR REPLACE VIEW hf_exposure_start_days AS
SELECT
    subject_id,
    drug_class,
    CAST(starttime AS DATE) AS start_date,
    MIN(starttime)          AS first_starttime_that_day,
    COUNT(*)                AS n_prescription_rows_that_day,
    string_agg(DISTINCT drug, ' | ' ORDER BY drug) AS drugs_that_day
FROM hf_cohort_drug_starts
GROUP BY subject_id, drug_class, CAST(starttime AS DATE);

CREATE OR REPLACE VIEW hf_exposure_days_with_gaps AS
SELECT
    d.*,
    LAG(start_date) OVER (
        PARTITION BY subject_id, drug_class
        ORDER BY start_date
    ) AS previous_start_date,
    DATE_DIFF(
        'day',
        LAG(start_date) OVER (PARTITION BY subject_id, drug_class ORDER BY start_date),
        start_date
    ) AS days_since_previous_start
FROM hf_exposure_start_days d;

CREATE OR REPLACE VIEW hf_exposure_days_with_cycle_flags AS
SELECT
    d.*,
    CASE
        WHEN previous_start_date IS NULL                                              THEN 1
        WHEN days_since_previous_start > (SELECT cycle_gap_days FROM hf_cohort_params) THEN 1
        ELSE 0
    END AS is_new_cycle
FROM hf_exposure_days_with_gaps d;

CREATE OR REPLACE VIEW hf_exposure_days_with_cycle_id AS
SELECT
    d.*,
    SUM(is_new_cycle) OVER (
        PARTITION BY subject_id, drug_class
        ORDER BY start_date
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    ) AS cycle_number
FROM hf_exposure_days_with_cycle_flags d;

CREATE OR REPLACE VIEW hf_cycle_exposures AS
SELECT
    s.subject_id,
    s.drug_class,
    s.cycle_number,
    MIN(s.first_starttime_that_day)     AS prediction_time,
    MIN(s.start_date)                   AS cycle_start_date,
    MAX(s.start_date)                   AS cycle_end_date,
    COUNT(DISTINCT s.start_date)        AS n_exposure_start_days_in_cycle,
    SUM(s.n_prescription_rows_that_day) AS n_prescription_rows_in_cycle,
    string_agg(DISTINCT s.drugs_that_day, ' | ' ORDER BY s.drugs_that_day) AS drugs_in_cycle,

    -- Drug-class-specific monitoring window (not MAX across classes — each class is tracked independently)
    w.toxicity_window_days,
    w.window_rationale

FROM hf_exposure_days_with_cycle_id s
LEFT JOIN hf_drug_toxicity_windows w ON s.drug_class = w.drug_class
GROUP BY s.subject_id, s.drug_class, s.cycle_number, w.toxicity_window_days, w.window_rationale;
