-- Get specific columns and create a table (acts like a variable that copies some or all data from some place)
CREATE TABLE admissions AS
SELECT subject_id, hadm_id, admittime, dischtime 
FROM 'mimic-iv-3.1/hosp/admissions.csv'; -- 223452, as expected see docs

-- Note that creating tables do not output anything. Using SELECT is what outputs something to the CLI or console

-- Count the total number of patients
--SELECT COUNT(DISTINCT subject_id) FROM 'mimic-iv-3.1/hosp/admissions.csv'; -- 223452, as expected see docs
--SELECT DISTINCT subject_id FROM 'mimic-iv-3.1/hosp/patients.csv'; --364627, as expected

-- Get the top 3 rows:
-- SELECT * FROM admissions
-- LIMIT 3;

-- Filter by patient rows
--SELECT subject_id, hadm_id, admittime, dischtime FROM 'mimic-iv-3.1/hosp/admissions.csv'
--WHERE subject_id = 19999828;

-- Describe the data and its types
--DESC SELECT * FROM 'mimic-iv-3.1/hosp/admissions.csv';

-- Filter by time
-- SELECT subject_id, hadm_id, admittime, dischtime FROM 'mimic-iv-3.1/hosp/admissions.csv'
-- WHERE admittime > '2180-07-23 12:35:00';

-- Order the dataset by the admission time
--SELECT * FROM admissions
--ORDER BY admittime; -- add DESC to order from descending or ASC (default) ascending

-- Use additional operators for filtering like AND or OR (only returns if the statements are true like a truth table)
SELECT * FROM admissions
WHERE EXTRACT(YEAR FROM admittime) = 2110 
OR EXTRACT(YEAR FROM admittime) = 2089;