-- 04_cv_toxicity_events.sql
--
-- Build ICD/admission-based cardiovascular toxicity events.
--
-- Main outputs:
--   cv_toxicity_events
--   pre_existing_cv_history

CREATE OR REPLACE VIEW cancer_cv_base_all AS
SELECT
    c.subject_id,
    c.first_oncology_time,
    a.hadm_id,
    CAST(a.admittime AS TIMESTAMP) AS admittime,
    d.icd_code,
    d.icd_version
FROM cancer_first_drug c
JOIN read_csv_auto('mimic-iv-3.1/hosp/diagnoses_icd.csv') d
    ON c.subject_id = d.subject_id
JOIN read_csv_auto('mimic-iv-3.1/hosp/admissions.csv') a
    ON d.hadm_id = a.hadm_id;

CREATE OR REPLACE VIEW cancer_cv_admission_events AS
SELECT
    subject_id,
    first_oncology_time,
    hadm_id,
    admittime,
    MAX(
        CASE
            WHEN icd_code LIKE '425%' THEN 1
            WHEN icd_code LIKE '428%' THEN 1
            WHEN icd_code LIKE '410%' THEN 1
            WHEN icd_code LIKE '411%' THEN 1
            WHEN icd_code LIKE '412%' THEN 1
            WHEN icd_code LIKE '413%' THEN 1
            WHEN icd_code LIKE '414%' THEN 1
            WHEN icd_code LIKE '420%' THEN 1
            WHEN icd_code LIKE '421%' THEN 1
            WHEN icd_code LIKE '422%' THEN 1
            WHEN icd_code LIKE '423%' THEN 1
            WHEN icd_code LIKE '426%' THEN 1
            WHEN icd_code LIKE '427%' THEN 1
            WHEN icd_code LIKE 'I42%' THEN 1
            WHEN icd_code LIKE 'I50%' THEN 1
            WHEN icd_code LIKE 'I21%' THEN 1
            WHEN icd_code LIKE 'I22%' THEN 1
            WHEN icd_code LIKE 'I23%' THEN 1
            WHEN icd_code LIKE 'I63%' THEN 1
            WHEN icd_code LIKE 'I64%' THEN 1
            WHEN icd_code LIKE 'I30%' THEN 1
            WHEN icd_code LIKE 'I31%' THEN 1
            WHEN icd_code LIKE 'I40%' THEN 1
            WHEN icd_code LIKE 'I41%' THEN 1
            WHEN icd_code LIKE 'I44%' THEN 1
            WHEN icd_code LIKE 'I45%' THEN 1
            WHEN icd_code LIKE 'I46%' THEN 1
            WHEN icd_code LIKE 'I47%' THEN 1
            WHEN icd_code LIKE 'I48%' THEN 1
            WHEN icd_code LIKE 'I49%' THEN 1
            WHEN icd_code LIKE 'I514%' THEN 1
            ELSE 0
        END
    ) AS cv_event,
    string_agg(DISTINCT icd_code, ' | ' ORDER BY icd_code) AS cv_event_icd_codes
FROM cancer_cv_base_all
GROUP BY
    subject_id,
    first_oncology_time,
    hadm_id,
    admittime;

CREATE OR REPLACE VIEW cv_toxicity_events AS
SELECT
    subject_id,
    admittime AS toxicity_time,
    'cv_diagnosis_admission' AS toxicity_type,
    hadm_id,
    cv_event_icd_codes
FROM cancer_cv_admission_events
WHERE cv_event = 1
  AND admittime >= first_oncology_time;

CREATE OR REPLACE VIEW pre_existing_cv_history AS
SELECT DISTINCT
    subject_id,
    1 AS pre_existing_cv_history
FROM cancer_cv_admission_events
WHERE cv_event = 1
  AND admittime < first_oncology_time;
