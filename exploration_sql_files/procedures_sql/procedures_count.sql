-- Load procedures and filter by procedures that might cause/indicative of cardiotoxicity. 

CREATE TABLE cancer_procedures AS
SELECT
    subject_id,
    hadm_id,
    seq_num,
    chartdate,
    icd_code,
    icd_version
FROM read_csv_auto('mimic-iv-3.1/hosp/procedures_icd.csv')
WHERE
(
    -- ICD-9
    (
        icd_version = 9 AND (
            icd_code LIKE '922%'   -- radiation
            OR icd_code = '9925'    -- chemo infusion
        )
    )

    OR

    -- ICD-10-PCS
    (
        icd_version = 10 AND (
            icd_code LIKE 'D7%'   -- radiation therapy
        )
    )
);

-- Returns 2891 patients
SELECT DISTINCT subject_id FROM cancer_procedures