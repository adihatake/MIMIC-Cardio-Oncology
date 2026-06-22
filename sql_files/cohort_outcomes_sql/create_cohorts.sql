
-- Follow-up admission evidence:
-- identifies patients who had any hospital admission within 1 year
-- after first oncology drug exposure, regardless of whether it had a CV diagnosis.
CREATE OR REPLACE VIEW cancer_followup_admission_evidence AS
SELECT
    c.subject_id,
    1 AS has_followup_admission_1yr,
    MIN(a.admittime) AS first_followup_admission_time_1yr,
    COUNT(DISTINCT a.hadm_id) AS n_followup_admissions_1yr
FROM cancer_first_drug c
JOIN read_csv_auto('mimic-iv-3.1/hosp/admissions.csv') a
    ON c.subject_id = a.subject_id
WHERE a.admittime >= c.first_oncology_time
  AND a.admittime < c.first_oncology_time + INTERVAL '1 year'
GROUP BY c.subject_id;


-- Create a cardiotoxicity master table.
CREATE OR REPLACE VIEW cardiotoxicity_master AS
SELECT
    c.subject_id,
    c.first_oncology_time,

    -- ICD outcome
    COALESCE(cv.cv_event_1yr, 0) AS cv_event_1yr,

    -- LVEF availability flags
    COALESCE(l.has_baseline_lvef, 0) AS has_baseline_lvef,
    COALESCE(l.has_followup_lvef_1yr, 0) AS has_followup_lvef_1yr,

    -- LVEF values/times
    l.baseline_time,
    l.baseline_lvef,
    l.first_followup_lvef_time_1yr,
    l.worst_followup_lvef_1yr,

    -- LVEF outcome components
    COALESCE(l.lvef_drop_10_1yr, 0) AS lvef_drop_10_1yr,
    COALESCE(l.lvef_below_50_1yr, 0) AS lvef_below_50_1yr,
    COALESCE(l.definite_lvef_ctrcd_1yr, 0) AS definite_lvef_ctrcd_1yr,

    -- History flag
    COALESCE(h.pre_existing_cv_history, 0) AS pre_existing_cv_history,

    -- Composite cardiotoxicity outcome
    CASE
        WHEN COALESCE(cv.cv_event_1yr, 0) = 1
          OR COALESCE(l.definite_lvef_ctrcd_1yr, 0) = 1
        THEN 1
        ELSE 0
    END AS cardiotoxicity_1yr,

    -- Did we have any reasonable way to observe the outcome?
    CASE
        WHEN COALESCE(cv.cv_event_1yr, 0) = 1
          OR COALESCE(l.has_followup_lvef_1yr, 0) = 1
          OR COALESCE(fa.has_followup_admission_1yr, 0) = 1 -- checks if the patient had a follow-up visit.
        THEN 1
        ELSE 0
    END AS has_outcome_evidence_1yr,

    -- Safer modeling label. Divides patients into 3 groups:
    -- Positive; if they met the cardiotoxicity criteria
    -- Negative_observed; patient did not meet criteria, but had follow-up evidence
    -- Unknown; patient did not meet criteria, but had no follow-up evidence.
    CASE
        WHEN COALESCE(cv.cv_event_1yr, 0) = 1
          OR COALESCE(l.definite_lvef_ctrcd_1yr, 0) = 1
        THEN 'positive'

        WHEN COALESCE(l.has_followup_lvef_1yr, 0) = 1
          OR COALESCE(fa.has_followup_admission_1yr, 0) = 1
        THEN 'negative_observed'

        ELSE 'unknown_no_followup_evidence'
    END AS cardiotoxicity_label_1yr,

    -- Binary model eligibility flag
    -- Checks which patients/rows can be more confidently labelled as positive/negative.
    CASE
        WHEN COALESCE(cv.cv_event_1yr, 0) = 1
          OR COALESCE(l.definite_lvef_ctrcd_1yr, 0) = 1
          OR COALESCE(l.has_followup_lvef_1yr, 0) = 1
          OR COALESCE(fa.has_followup_admission_1yr, 0) = 1
        THEN 1
        ELSE 0
    END AS eligible_for_binary_label_1yr

FROM cancer_first_drug c

LEFT JOIN cancer_cv_outcomes cv
    ON c.subject_id = cv.subject_id

LEFT JOIN cancer_lvef_outcomes l
    ON c.subject_id = l.subject_id

LEFT JOIN cancer_cv_history h
    ON c.subject_id = h.subject_id

LEFT JOIN cancer_followup_admission_evidence fa
    ON c.subject_id = fa.subject_id;