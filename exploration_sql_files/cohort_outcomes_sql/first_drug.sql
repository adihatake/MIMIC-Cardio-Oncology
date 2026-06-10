-- Get the time of the first drug given to cancer patients

-- Get the subject ID and the first time the drug given
CREATE VIEW first_oncology_drug AS
SELECT
    subject_id, -- BIGINT
    MIN(starttime) AS first_oncology_time --TIMESTAMP
FROM oncology_drugs
GROUP BY subject_id; -- Borrows from revised_prescription_counts.sql

-- Find intersection with the active and past cancer cohort diagnoses
CREATE VIEW cancer_first_drug AS
SELECT
    f.subject_id,
    f.first_oncology_time
FROM first_oncology_drug f -- Borrows from history_and_active.sql
INNER JOIN all_cancer_patients c
    ON f.subject_id = c.subject_id;

-- Returns 2545 patients who have received a cancer drug and their first occurence. 
-- SELECT DISTINCT * FROM cancer_first_drug;