-- UNION both tables and find the distinct number of patients
-- Distinct patients with cancer diagnoses: 48438
CREATE OR REPLACE VIEW all_cancer_patients AS
SELECT DISTINCT subject_id
FROM (
    SELECT subject_id FROM active_cancer
    UNION
    SELECT subject_id FROM cancer_history
);

SELECT * FROM all_cancer_patients;