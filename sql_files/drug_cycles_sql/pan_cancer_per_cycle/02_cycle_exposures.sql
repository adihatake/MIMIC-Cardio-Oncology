-- 02_cycle_exposures.sql  (pan_cancer_per_cycle)
--
-- Collapse drug administration rows into treatment cycles.
-- Cycles are defined ACROSS all drug classes jointly (one cycle number per patient,
-- not one per drug class).  A gap of > cycle_gap_days between any consecutive
-- prescription start dates marks the start of a new cycle.
--
-- Drug class exposure is encoded as binary indicator columns within each cycle row,
-- so a patient receiving anthracycline + taxane in cycle 1 produces ONE row with
-- exposed_anthracycline=1 and exposed_taxane=1.
--
-- Toxicity window = MAX of class-specific windows across all drugs in the cycle.
-- This is conservative: the longest plausible attribution horizon is used, matching
-- the ESC 2022 guidance for each drug class (from 00_parameters_and_windows.sql).
--
-- Main output:
--   pcr_cycle_exposures  (one row per subject_id / cycle_number)

-- Aggregate to one row per (patient, date) across all drug classes combined
CREATE OR REPLACE VIEW pcr_exposure_start_days AS
SELECT
    subject_id,
    CAST(starttime AS DATE)                                           AS start_date,
    MIN(starttime)                                                    AS first_starttime_that_day,
    COUNT(*)                                                          AS n_prescription_rows_that_day,
    string_agg(DISTINCT drug,       ' | ' ORDER BY drug)             AS drugs_that_day,
    string_agg(DISTINCT drug_class, ' | ' ORDER BY drug_class)       AS drug_classes_that_day
FROM pcr_cohort_drug_starts
GROUP BY subject_id, CAST(starttime AS DATE);

-- Compute gap to previous start date (across all drug classes, per patient)
CREATE OR REPLACE VIEW pcr_exposure_days_with_gaps AS
SELECT
    d.*,
    LAG(start_date) OVER (
        PARTITION BY subject_id
        ORDER BY start_date
    ) AS previous_start_date,
    DATE_DIFF(
        'day',
        LAG(start_date) OVER (PARTITION BY subject_id ORDER BY start_date),
        start_date
    ) AS days_since_previous_start
FROM pcr_exposure_start_days d;

-- Flag the first day of each new cycle
CREATE OR REPLACE VIEW pcr_exposure_days_with_cycle_flags AS
SELECT
    d.*,
    CASE
        WHEN previous_start_date IS NULL THEN 1
        WHEN days_since_previous_start > (SELECT cycle_gap_days FROM pcr_params) THEN 1
        ELSE 0
    END AS is_new_cycle
FROM pcr_exposure_days_with_gaps d;

-- Assign monotonically-increasing cycle_number per patient
CREATE OR REPLACE VIEW pcr_exposure_days_with_cycle_id AS
SELECT
    d.*,
    SUM(is_new_cycle) OVER (
        PARTITION BY subject_id
        ORDER BY start_date
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    ) AS cycle_number
FROM pcr_exposure_days_with_cycle_flags d;

-- One row per (patient, cycle_number) with binary drug-class exposure flags
-- and the per-cycle max toxicity window
CREATE OR REPLACE VIEW pcr_cycle_exposures AS
SELECT
    s.subject_id,
    s.cycle_number,
    MIN(s.first_starttime_that_day)     AS prediction_time,
    MIN(s.start_date)                   AS cycle_start_date,
    MAX(s.start_date)                   AS cycle_end_date,
    COUNT(DISTINCT s.start_date)        AS n_exposure_start_days_in_cycle,
    SUM(s.n_prescription_rows_that_day) AS n_prescription_rows_in_cycle,
    string_agg(DISTINCT d.drug,       ' | ' ORDER BY d.drug)       AS drugs_in_cycle,
    string_agg(DISTINCT d.drug_class, ' | ' ORDER BY d.drug_class) AS drug_classes_in_cycle,

    -- Max toxicity window across all drug classes present in this cycle
    MAX(COALESCE(w.toxicity_window_days,
        (SELECT baseline_lookback_days FROM pcr_params))) AS toxicity_window_days,
    string_agg(DISTINCT COALESCE(w.window_rationale, 'fallback'),
        ' | ' ORDER BY COALESCE(w.window_rationale, 'fallback')) AS window_rationales,

    -- Binary drug-class exposure flags (1 = drug class present in this cycle)
    MAX(CASE WHEN d.drug_class = 'anthracycline'               THEN 1 ELSE 0 END) AS exposed_anthracycline,
    MAX(CASE WHEN d.drug_class = 'her2_targeted'               THEN 1 ELSE 0 END) AS exposed_her2_targeted,
    MAX(CASE WHEN d.drug_class = 'immune_checkpoint_inhibitor' THEN 1 ELSE 0 END) AS exposed_immune_checkpoint_inhibitor,
    MAX(CASE WHEN d.drug_class = 'taxane'                      THEN 1 ELSE 0 END) AS exposed_taxane,
    MAX(CASE WHEN d.drug_class = 'fluoropyrimidine'            THEN 1 ELSE 0 END) AS exposed_fluoropyrimidine,
    MAX(CASE WHEN d.drug_class = 'vegf_inhibitor'              THEN 1 ELSE 0 END) AS exposed_vegf_inhibitor,
    MAX(CASE WHEN d.drug_class = 'egfr_inhibitor'              THEN 1 ELSE 0 END) AS exposed_egfr_inhibitor,
    MAX(CASE WHEN d.drug_class = 'tyrosine_kinase_inhibitor'   THEN 1 ELSE 0 END) AS exposed_tyrosine_kinase_inhibitor,
    MAX(CASE WHEN d.drug_class = 'proteasome_inhibitor'        THEN 1 ELSE 0 END) AS exposed_proteasome_inhibitor,
    MAX(CASE WHEN d.drug_class = 'immunomodulatory_agent'      THEN 1 ELSE 0 END) AS exposed_immunomodulatory_agent

FROM pcr_exposure_days_with_cycle_id s
JOIN pcr_cohort_drug_starts d
    ON s.subject_id = d.subject_id
   AND CAST(d.starttime AS DATE) = s.start_date
LEFT JOIN pcr_drug_toxicity_windows w
    ON d.drug_class = w.drug_class
GROUP BY s.subject_id, s.cycle_number;
