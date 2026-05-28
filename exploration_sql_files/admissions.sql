-- Find number of distinct patients in `admissions.csv` (all hospital visits in MIMIC)
-- SELECT COUNT(DISTINCT subject_id) 
-- FROM read_csv_auto('mimic-iv-3.1/hosp/admissions.csv');

-- Find the average number of hospital visits
SELECT AVG(subject_count) -- returns 2.443603 avg hospital visits
FROM(
    SELECT subject_id, COUNT(*) AS subject_count 
    FROM read_csv_auto('mimic-iv-3.1/hosp/admissions.csv')
    GROUP BY subject_id -- returns a table with 223452 rows as expected
    )
