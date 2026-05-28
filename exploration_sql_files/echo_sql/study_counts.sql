-- Get counts for number of studies and average # of studies per patients, min, max
-- CREATE TABLE echo_measurements AS
-- SELECT
--     subject_id,
--     measurement_id,
--     measurement_datetime,
--     test_type,
--     measurement,
--     measurement_description,
--     result,
--     unit
-- FROM read_csv_auto('mimic-iv-echo/structured-measurement.csv');

-- SELECT DISTINCT subject_id FROM echo_measurements; -- returns 91 372 unique patients as expected from the dataset


-- Get counts for number of echo studies and studies per patient
CREATE TABLE echo_measurements AS
SELECT
    subject_id,
    measurement_id,
    CAST(measurement_datetime AS TIMESTAMP) AS measurement_datetime,
    test_type,
    measurement,
    measurement_description,
    result,
    unit
FROM read_csv_auto('mimic-iv-echo/structured-measurement.csv')
WHERE subject_id IS NOT NULL
  AND measurement_id IS NOT NULL;

-- Overall dataset counts
SELECT
    COUNT(*) AS n_measurement_rows,
    COUNT(DISTINCT subject_id) AS n_unique_patients,
    COUNT(DISTINCT measurement_id) AS n_unique_studies
FROM echo_measurements;

-- Studies per patient summary
WITH studies_per_patient AS (
    SELECT
        subject_id,
        COUNT(DISTINCT measurement_id) AS n_studies
    FROM echo_measurements
    GROUP BY subject_id
)

SELECT
    COUNT(*) AS n_patients,
    AVG(n_studies) AS avg_studies_per_patient,
    MIN(n_studies) AS min_studies_per_patient,
    MAX(n_studies) AS max_studies_per_patient
FROM studies_per_patient;