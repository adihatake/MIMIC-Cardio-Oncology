-- 06_observation_and_death.sql  (pan_cancer_per_cycle)
--
-- Last observation point and death timing per patient.
-- Death terminates the observation window; patients dying before a CTRCD event
-- within the monitoring window receive label 'censored_death'.
--
-- Observation sources:
--   admittime  — hospital admission start
--   dischtime  — hospital discharge (patient demonstrably followed through this date)
--   drug_administration start times
--
-- Death sources:
--   admissions.deathtime (TIMESTAMP) — precise in-hospital death; preferred.
--   patients.dod (DATE cast to midnight TIMESTAMP) — out-of-hospital; fallback only.
--
-- Main outputs:
--   pcr_patient_death             (death timestamp per patient)
--   pcr_patient_observation_events (all observation events)
--   pcr_patient_last_observation  (last known observation time per patient)

-- Death timestamps (prefer in-hospital deathtime; fall back to dod midnight)
CREATE OR REPLACE VIEW pcr_patient_death AS
SELECT
    p.subject_id,
    COALESCE(
        MAX(CAST(a.deathtime AS TIMESTAMP)),
        CAST(MAX(p.dod) AS TIMESTAMP)
    ) AS death_time
FROM read_csv_auto('mimic-iv-3.1/hosp/patients.csv') p
INNER JOIN pcr_patient_first_drug f ON p.subject_id = f.subject_id
LEFT JOIN read_csv_auto('mimic-iv-3.1/hosp/admissions.csv') a
    ON p.subject_id = a.subject_id
   AND a.deathtime IS NOT NULL
WHERE p.dod IS NOT NULL
GROUP BY p.subject_id;

-- All observation events for cohort patients
CREATE OR REPLACE VIEW pcr_patient_observation_events AS
SELECT
    subject_id,
    CAST(admittime AS TIMESTAMP) AS observation_time,
    'admission' AS observation_type
FROM read_csv_auto('mimic-iv-3.1/hosp/admissions.csv')
WHERE subject_id IN (SELECT subject_id FROM pcr_patient_first_drug)

UNION ALL

SELECT
    subject_id,
    CAST(dischtime AS TIMESTAMP) AS observation_time,
    'discharge' AS observation_type
FROM read_csv_auto('mimic-iv-3.1/hosp/admissions.csv')
WHERE subject_id IN (SELECT subject_id FROM pcr_patient_first_drug)
  AND dischtime IS NOT NULL

UNION ALL

SELECT
    subject_id,
    starttime  AS observation_time,
    'drug_administration' AS observation_type
FROM pcr_cohort_drug_starts;

-- Last known observation time per patient
CREATE OR REPLACE VIEW pcr_patient_last_observation AS
SELECT
    subject_id,
    MAX(observation_time) AS last_observation_time
FROM pcr_patient_observation_events
GROUP BY subject_id;
