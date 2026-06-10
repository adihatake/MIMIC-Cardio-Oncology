-- 1. Fetch call cancer and admissions data
-- This step links:
-- (1) cancer cohort index time
-- (2) hospital admissions
-- (3) diagnosis ICD codes
-- so we can evaluate diagnoses after drug exposure

CREATE VIEW cancer_cv_base AS
SELECT
    c.subject_id,   -- patient identifier
    c.first_oncology_time,  -- index time (first cancer drug exposure)

    a.hadm_id,     -- hospital admission ID
    a.admittime,   -- admission timestamp (used for time-window filtering)

    d.icd_code   -- diagnosis code (ICD-9 or ICD-10)
FROM cancer_first_drug c
JOIN read_csv_auto('mimic-iv-3.1/hosp/diagnoses_icd.csv') d
    ON c.subject_id = d.subject_id   -- link diagnoses to patient
JOIN read_csv_auto('mimic-iv-3.1/hosp/admissions.csv') a
    ON d.hadm_id = a.hadm_id;        -- attach admission timing information





-- 2. Define the ICD codes to filter cardiovascular outcomes
-- Goal:
-- Collapse multiple ICD codes per admission into a single binary indicator:
-- "Did this admission contain ANY cardiovascular ICD code?"

CREATE OR REPLACE VIEW cancer_cv_admission_flag AS
SELECT
    subject_id,
    hadm_id,
    admittime,
    first_oncology_time,

    MAX(
        CASE 
            -- ICD-9 cardiomyopathy / heart failure
            WHEN icd_code LIKE '425%' THEN 1  -- Cardiomyopathy
            WHEN icd_code LIKE '428%' THEN 1  -- Heart failure

            -- ICD-9 ischemic heart disease / MI
            WHEN icd_code LIKE '410%' THEN 1  -- Acute MI
            WHEN icd_code LIKE '411%' THEN 1
            WHEN icd_code LIKE '412%' THEN 1
            WHEN icd_code LIKE '413%' THEN 1
            WHEN icd_code LIKE '414%' THEN 1

            -- ICD-10 cardiomyopathy / heart failure
            WHEN icd_code LIKE 'I42%' THEN 1  -- Cardiomyopathy
            WHEN icd_code LIKE 'I50%' THEN 1  -- Heart failure

            -- ICD-10 MI / stroke
            WHEN icd_code LIKE 'I21%' THEN 1
            WHEN icd_code LIKE 'I22%' THEN 1
            WHEN icd_code LIKE 'I23%' THEN 1
            WHEN icd_code LIKE 'I63%' THEN 1
            WHEN icd_code LIKE 'I64%' THEN 1

            ELSE 0
        END
    ) AS cv_event

FROM cancer_cv_base
GROUP BY
    subject_id,
    hadm_id,
    admittime,
    first_oncology_time;



-- 3. Identify pre-existing cardiovascular disease before first oncology drug
CREATE OR REPLACE VIEW cancer_cv_history AS
SELECT DISTINCT
    subject_id,
    1 AS pre_existing_cv_history
FROM cancer_cv_admission_flag
WHERE cv_event = 1
  AND admittime < first_oncology_time;



-- 4. Create the cardiovascular outcomes table for cancer patients
CREATE OR REPLACE VIEW cancer_cv_outcomes AS
SELECT
    subject_id,
    1 AS cv_event_1yr,
    MIN(admittime) AS first_cv_event_time_1yr,
    COUNT(DISTINCT hadm_id) AS n_cv_event_admissions_1yr
FROM cancer_cv_admission_flag
WHERE cv_event = 1
  AND admittime >= first_oncology_time
  AND admittime < first_oncology_time + INTERVAL '1 year'
GROUP BY subject_id;