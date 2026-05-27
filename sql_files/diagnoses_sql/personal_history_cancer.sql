-- Get number of entries with personal history of malignant cancer 
-- Returns: 84381 rows
CREATE VIEW cancer_history AS (
    SELECT DISTINCT
        subject_id,
        hadm_id,
        seq_num,
        icd_code,
        icd_version
    FROM read_csv_auto('mimic-iv-3.1/hosp/diagnoses_icd.csv')
    WHERE
    (
        icd_version = 9
        AND icd_code LIKE 'V10%'
    )
    OR
    (
        icd_version = 10
        AND icd_code LIKE 'Z85%'
    )
);

-- Get number of distinct patients with personal history of malignant cancer 
-- Returns: 32399
SELECT COUNT(DISTINCT subject_id) FROM cancer_history;