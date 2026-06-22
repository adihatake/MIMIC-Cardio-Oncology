-- Normal ECG cohort
CREATE TABLE ecg_studies AS
SELECT *
FROM read_csv_auto('mimic-iv-ecg/machine_measurements.csv');

SELECT DISTINCT subject_id FROM ecg_studies;

-- Get summary statistics

WITH ecg_counts AS (
    SELECT
        subject_id,
        COUNT(*) AS num_ecgs
    FROM ecg_studies
    GROUP BY subject_id
)

SELECT
    COUNT(*) AS num_patients,
    AVG(num_ecgs) AS mean_ecgs,
    MEDIAN(num_ecgs) AS median_ecgs,
    MIN(num_ecgs) AS min_ecgs,
    MAX(num_ecgs) AS max_ecgs,
    STDDEV(num_ecgs) AS std_ecgs
FROM ecg_counts;