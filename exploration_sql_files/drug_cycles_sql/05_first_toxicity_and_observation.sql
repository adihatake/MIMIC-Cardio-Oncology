-- 05_first_toxicity_and_observation.sql
--
-- Combine toxicity events and create observation/follow-up evidence.
--
-- Main outputs:
--   first_cardiotoxicity_event
--   patient_last_observation

CREATE OR REPLACE VIEW all_cardiotoxicity_events AS
SELECT
    subject_id,
    toxicity_time,
    toxicity_type
FROM lvef_toxicity_events

UNION ALL

SELECT
    subject_id,
    toxicity_time,
    toxicity_type
FROM cv_toxicity_events;

CREATE OR REPLACE VIEW first_cardiotoxicity_event AS
SELECT
    subject_id,
    toxicity_time AS first_toxicity_time,
    toxicity_type AS first_toxicity_type
FROM all_cardiotoxicity_events
QUALIFY ROW_NUMBER() OVER (
    PARTITION BY subject_id
    ORDER BY toxicity_time, toxicity_type
) = 1;

CREATE OR REPLACE VIEW patient_observation_events AS
SELECT
    subject_id,
    CAST(admittime AS TIMESTAMP) AS observation_time,
    'admission' AS observation_type
FROM read_csv_auto('mimic-iv-3.1/hosp/admissions.csv')

UNION ALL

SELECT
    subject_id,
    measurement_datetime AS observation_time,
    'lvef' AS observation_type
FROM all_lvef

UNION ALL

SELECT
    subject_id,
    starttime AS observation_time,
    'oncology_drug' AS observation_type
FROM oncology_drugs_classified;

CREATE OR REPLACE VIEW patient_last_observation AS
SELECT
    subject_id,
    MAX(observation_time) AS last_observation_time
FROM patient_observation_events
GROUP BY subject_id;
