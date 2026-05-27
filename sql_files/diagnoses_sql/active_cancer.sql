-- Get number of entries with active cancer diagnoses
-- Returns: 125471 rows
CREATE VIEW active_cancer AS (
    SELECT DISTINCT -- avoid duplicates just in case
        subject_id,
        hadm_id,
        seq_num
        icd_code,
        icd_version
    FROM read_csv_auto('mimic-iv-3.1/hosp/diagnoses_icd.csv')
    WHERE
    (
        icd_version = 9
        AND (
            icd_code BETWEEN '140' AND '172' -- exclude some skin cancers
            OR icd_code BETWEEN '174' AND '208'
        )
    )
    OR
    (
        icd_version = 10
        AND icd_code LIKE 'C%'
        AND icd_code NOT LIKE 'C44%' -- exclude some skin cancers
    )
);

-- Get number of distinct cancer patients
-- Returns: 29369
SELECT COUNT(DISTINCT subject_id) FROM active_cancer;