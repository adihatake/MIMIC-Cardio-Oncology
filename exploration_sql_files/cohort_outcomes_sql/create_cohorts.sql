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

    -- Outcome evidence flag
    CASE
        WHEN COALESCE(cv.cv_event_1yr, 0) = 1
          OR COALESCE(l.has_followup_lvef_1yr, 0) = 1
        THEN 1
        ELSE 0
    END AS has_outcome_evidence_1yr

FROM cancer_first_drug c

LEFT JOIN cancer_cv_outcomes cv
    ON c.subject_id = cv.subject_id

LEFT JOIN cancer_lvef_outcomes l
    ON c.subject_id = l.subject_id

LEFT JOIN cancer_cv_history h
    ON c.subject_id = h.subject_id;

SELECT * FROM cardiotoxicity_master;