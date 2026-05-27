-- Get counts for number of studies and average # of studies per patients, min, max
CREATE TABLE studies AS 
SELECT
    subject_id,
    study_id,
    study_datetime,
    measurement_id,
    measurement_datetime
FROM read_csv_auto("mimic-iv-echo/echo-study-list.csv");

-- Returns 4579 patients in the dataset
-- SELECT COUNT(DISTINCT subject_id) FROM studies;

SELECT DISTINCT subject_id FROM studies;