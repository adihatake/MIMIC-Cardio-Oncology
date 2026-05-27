-- Load the structure measurements and find the total number of patients with a dangerous LVEF (<50%)
CREATE TABLE LVEF_studies AS
SELECT
    subject_id,
    measurement_id,
    measurement_datetime,
    test_type,
    measurement,
    measurement_description,
    result,
    unit

FROM read_csv_auto('mimic-iv-echo/structured-measurement.csv')
WHERE measurement = 'lvef' AND CAST(result AS DOUBLE) < 50.0; -- result is stored as text so we need to convert

SELECT DISTINCT subject_id FROM LVEF_studies