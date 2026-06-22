-- 02_cycle_exposures.sql
--
-- Collapse prescription rows into cycle-like exposure rows.
--
-- Main output:
--   oncology_cycle_exposures

CREATE OR REPLACE VIEW cancer_oncology_drug_starts AS
SELECT
    o.subject_id,
    o.hadm_id,
    o.pharmacy_id,
    o.starttime,
    CAST(o.starttime AS DATE) AS start_date,
    o.stoptime,
    o.drug,
    o.drug_class
FROM oncology_drugs_classified o
INNER JOIN all_cancer_patients c
    ON o.subject_id = c.subject_id;

CREATE OR REPLACE VIEW exposure_start_days AS
SELECT
    subject_id,
    start_date,
    MIN(starttime) AS first_starttime_that_day,
    COUNT(*) AS n_prescription_rows_that_day,
    string_agg(DISTINCT drug, ' | ' ORDER BY drug) AS drugs_that_day,
    string_agg(DISTINCT drug_class, ' | ' ORDER BY drug_class) AS drug_classes_that_day
FROM cancer_oncology_drug_starts
GROUP BY subject_id, start_date;

CREATE OR REPLACE VIEW exposure_start_days_with_gaps AS
SELECT
    d.*,
    LAG(start_date) OVER (
        PARTITION BY subject_id
        ORDER BY start_date
    ) AS previous_start_date,
    DATE_DIFF(
        'day',
        LAG(start_date) OVER (
            PARTITION BY subject_id
            ORDER BY start_date
        ),
        start_date
    ) AS days_since_previous_start
FROM exposure_start_days d;

CREATE OR REPLACE VIEW exposure_start_days_with_cycle_flags AS
SELECT
    d.*,
    CASE
        WHEN previous_start_date IS NULL THEN 1
        WHEN days_since_previous_start > (SELECT cycle_gap_days FROM cycle_label_params) THEN 1
        ELSE 0
    END AS is_new_cycle
FROM exposure_start_days_with_gaps d;

CREATE OR REPLACE VIEW exposure_start_days_with_cycle_id AS
SELECT
    d.*,
    SUM(is_new_cycle) OVER (
        PARTITION BY subject_id
        ORDER BY start_date
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    ) AS cycle_number
FROM exposure_start_days_with_cycle_flags d;

CREATE OR REPLACE VIEW oncology_cycle_exposures AS
SELECT
    s.subject_id,
    s.cycle_number,
    MIN(s.first_starttime_that_day) AS prediction_time,
    MIN(s.start_date) AS cycle_start_date,
    MAX(s.start_date) AS cycle_end_date,
    COUNT(*) AS n_exposure_start_days_in_cycle,
    SUM(s.n_prescription_rows_that_day) AS n_prescription_rows_in_cycle,
    string_agg(DISTINCT o.drug, ' | ' ORDER BY o.drug) AS drugs_in_cycle,
    string_agg(DISTINCT o.drug_class, ' | ' ORDER BY o.drug_class) AS drug_classes_in_cycle,
    MAX(COALESCE(w.toxicity_window_days, (SELECT default_window_days FROM cycle_label_params))) AS toxicity_window_days,
    string_agg(DISTINCT COALESCE(w.window_rationale, 'Fallback window'), ' | ' ORDER BY COALESCE(w.window_rationale, 'Fallback window')) AS window_rationales,

    MAX(CASE WHEN o.drug_class = 'anthracycline' THEN 1 ELSE 0 END) AS exposed_anthracycline,
    MAX(CASE WHEN o.drug_class = 'immune_checkpoint_inhibitor' THEN 1 ELSE 0 END) AS exposed_immune_checkpoint_inhibitor,
    MAX(CASE WHEN o.drug_class = 'her2_targeted' THEN 1 ELSE 0 END) AS exposed_her2_targeted,
    MAX(CASE WHEN o.drug_class = 'taxane' THEN 1 ELSE 0 END) AS exposed_taxane,
    MAX(CASE WHEN o.drug_class = 'fluoropyrimidine' THEN 1 ELSE 0 END) AS exposed_fluoropyrimidine,
    MAX(CASE WHEN o.drug_class = 'vegf_inhibitor' THEN 1 ELSE 0 END) AS exposed_vegf_inhibitor,
    MAX(CASE WHEN o.drug_class = 'egfr_inhibitor' THEN 1 ELSE 0 END) AS exposed_egfr_inhibitor,
    MAX(CASE WHEN o.drug_class = 'tyrosine_kinase_inhibitor' THEN 1 ELSE 0 END) AS exposed_tyrosine_kinase_inhibitor,
    MAX(CASE WHEN o.drug_class = 'proteasome_inhibitor' THEN 1 ELSE 0 END) AS exposed_proteasome_inhibitor,
    MAX(CASE WHEN o.drug_class = 'immunomodulatory_agent' THEN 1 ELSE 0 END) AS exposed_immunomodulatory_agent
FROM exposure_start_days_with_cycle_id s
JOIN cancer_oncology_drug_starts o
    ON s.subject_id = o.subject_id
   AND s.start_date = o.start_date
LEFT JOIN drug_toxicity_windows w
    ON o.drug_class = w.drug_class
GROUP BY
    s.subject_id,
    s.cycle_number;
