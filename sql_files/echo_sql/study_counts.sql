-- Get counts for number of studies and average # of studies per patients, min, max
CREATE TABLE echo_measurements AS
SELECT
    subject_id,
    measurement_id,
    measurement_datetime,
    test_type,
    measurement,
    measurement_description,
    result,
    unit
FROM read_csv_auto('mimic-iv-echo/structured-measurement.csv');

SELECT DISTINCT subject_id FROM echo_measurements; -- returns 91 372 unique patients as expected from the dataset