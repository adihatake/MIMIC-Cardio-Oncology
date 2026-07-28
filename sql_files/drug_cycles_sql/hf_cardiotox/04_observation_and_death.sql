-- 04_observation_and_death.sql
--
-- Determine last point of observation per patient and capture death timing.
-- Death is treated as the hard end of the observation window.
--
-- Main outputs:
--   patient_death_hf
--   hf_patient_observation_events
--   hf_patient_last_observation

-- ─────────────────────────────────────────────────────────────────────────────
-- DEATH AS ENDPOINT — DESIGN NOTE
--
-- Death terminates the observation window. Patients who die within the monitoring
-- window before a recorded HF event are labeled 'censored_death' in the final
-- modeling table. This label exists so the analyst can explicitly decide how to
-- handle them. Three defensible options:
--
--   (1) EXCLUDE (recommended default)
--       Death from cancer or other non-cardiac causes is a competing event.
--       These patients did not have the opportunity to develop (and survive to
--       present with) HF. Excluding them from both the positive and negative
--       classes avoids bias in the binary label.
--       → Use: filter final table to label IN ('positive', 'negative_observed')
--
--   (2) TREAT AS NEGATIVE (optimistic)
--       If the patient's short remaining survival makes cardiac HF implausible,
--       count them as HF-free at time of death. Appropriate only when survival
--       is very short relative to the expected onset of HF.
--       → Use: recode 'censored_death' → 'negative_observed' before filtering
--
--   (3) TREAT AS POSITIVE (if cardiac cause of death available)
--       Fulminant ICI myocarditis, for example, can cause cardiac arrest before
--       a formal HF admission. MIMIC-IV does not reliably encode cause of death,
--       so this cannot be applied systematically. If ICD-coded cardiac cause of
--       death is available in a supplementary source, these rows can be recoded.
--       → Use: match on death ICD codes in external source; recode accordingly
--
-- For a competing-risks survival analysis (e.g., Fine-Gray model), all three
-- patient types (positive HF, censored_death, negative_observed) are kept and
-- modeled jointly. For binary classification, option (1) is the conservative default.
-- ─────────────────────────────────────────────────────────────────────────────

-- Death timestamps: prefer admissions.deathtime (TIMESTAMP with time-of-day) for in-hospital
-- deaths; fall back to patients.dod (DATE) for out-of-hospital deaths.
-- patients.dod is a DATE — CAST to TIMESTAMP gives midnight (00:00), which can place death
-- up to ~24 hours early relative to the actual time-of-death. Using admissions.deathtime
-- avoids systematic early-bias when comparing death to a monitoring window boundary.
CREATE OR REPLACE VIEW patient_death_hf AS
SELECT
    p.subject_id,
    COALESCE(
        MAX(CAST(a.deathtime AS TIMESTAMP)),  -- precise in-hospital death time
        CAST(p.dod AS TIMESTAMP)              -- fallback: dod midnight for out-of-hospital deaths
    ) AS death_time
FROM read_csv_auto('mimic-iv-3.1/hosp/patients.csv') p
INNER JOIN hf_patient_first_drug f ON p.subject_id = f.subject_id
LEFT JOIN read_csv_auto('mimic-iv-3.1/hosp/admissions.csv') a
    ON p.subject_id = a.subject_id
   AND a.deathtime IS NOT NULL
WHERE p.dod IS NOT NULL
GROUP BY p.subject_id;

-- Observation events: hospital admissions (admit AND discharge), and drug administrations.
-- dischtime is included because a patient admitted before the window end but discharged after
-- it was demonstrably followed through that window; using only admittime would under-count.
CREATE OR REPLACE VIEW hf_patient_observation_events AS
SELECT
    subject_id,
    CAST(admittime AS TIMESTAMP) AS observation_time,
    'admission' AS observation_type
FROM read_csv_auto('mimic-iv-3.1/hosp/admissions.csv')
WHERE subject_id IN (SELECT subject_id FROM hf_patient_first_drug)

UNION ALL

SELECT
    subject_id,
    CAST(dischtime AS TIMESTAMP) AS observation_time,
    'discharge' AS observation_type
FROM read_csv_auto('mimic-iv-3.1/hosp/admissions.csv')
WHERE subject_id IN (SELECT subject_id FROM hf_patient_first_drug)
  AND dischtime IS NOT NULL

UNION ALL

SELECT
    subject_id,
    starttime  AS observation_time,
    'drug_administration' AS observation_type
FROM hf_cohort_drug_starts;

-- Last known observation time per patient
CREATE OR REPLACE VIEW hf_patient_last_observation AS
SELECT
    subject_id,
    MAX(observation_time) AS last_observation_time
FROM hf_patient_observation_events
GROUP BY subject_id;
