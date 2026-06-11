-- Get number of entries with active cancer diagnoses
CREATE VIEW active_cancer AS (
    SELECT DISTINCT
        subject_id,
        hadm_id,
        seq_num,        -- ← also add the missing comma from the earlier bug
        icd_code,
        icd_version
    FROM read_csv_auto('mimic-iv-3.1/hosp/diagnoses_icd.csv')
    WHERE
    (
        icd_version = 9
        AND (
            (TRY_CAST(SUBSTRING(icd_code, 1, 3) AS INTEGER) BETWEEN 140 AND 172 -- Excludes skin cancers
            OR TRY_CAST(SUBSTRING(icd_code, 1, 3) AS INTEGER) BETWEEN 174 AND 208)
        )
    )
    OR
    (
        icd_version = 10
        AND icd_code LIKE 'C%'
        AND icd_code NOT LIKE 'C44%' -- excludes skin cancers
    )
);