-- UNION both tables and find the distinct number of patients
-- Distinct patients with cancer diagnoses: 48438
SELECT DISTINCT subject_id
FROM (
    SELECT subject_id FROM active_cancer
    UNION
    SELECT subject_id FROM cancer_history
);
