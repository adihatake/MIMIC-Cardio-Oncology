-- 03_heart_failure_events.sql
--
-- Identify heart failure diagnoses from MIMIC-IV ICD codes.
-- Separates pre-existing HF (before cohort entry) from incident HF (after first drug).
--
-- ICD codes included (all subtypes of heart failure):
--   ICD-10  I50%  — Heart failure, all subtypes:
--             I50.1  Left ventricular failure (pulmonary edema)
--             I50.2x Systolic HF (unspecified / acute / chronic / acute-on-chronic)
--             I50.3x Diastolic HF (unspecified / acute / chronic / acute-on-chronic)
--             I50.4x Combined systolic + diastolic HF
--             I50.81 Right heart failure
--             I50.82 Biventricular HF
--             I50.84 End-stage HF
--             I50.89 Other specified HF
--             I50.9  HF, unspecified
--   ICD-9   428%  — Heart failure, all subtypes (428.0 – 428.9)
--
-- ICD codes intentionally excluded (to keep outcome specific to HF):
--   I42% / 425%  — Cardiomyopathy: structural precursor to HF, not HF itself.
--                  Can be added in a sensitivity analysis to detect earlier cardiotoxicity.
--   I11.0 / 402% — Hypertensive heart disease with HF: often pre-existing and
--                  not directly drug-attributable; may be added with caution.
--
-- Main outputs:
--   hf_diagnoses_all          (all HF ICD records for cohort patients, with timing)
--   pre_existing_hf           (HF present before cohort entry)
--   pre_existing_cardiomyopathy (CMP present before cohort entry — I42%/425%)
--   pre_existing_hf_or_cmp    (combined flag used by the strict binary table)
--   incident_hf_events        (first HF admission after cohort entry)

-- All HF ICD diagnoses for cohort patients, joined to admission timestamps
CREATE OR REPLACE VIEW hf_diagnoses_all AS
SELECT
    p.subject_id,
    p.first_drug_time,
    a.hadm_id,
    CAST(a.admittime AS TIMESTAMP) AS admittime,
    CAST(a.dischtime AS TIMESTAMP) AS dischtime,
    d.icd_code,
    d.icd_version
FROM hf_patient_first_drug p
JOIN read_csv_auto('mimic-iv-3.1/hosp/diagnoses_icd.csv') d
    ON p.subject_id = d.subject_id
JOIN read_csv_auto('mimic-iv-3.1/hosp/admissions.csv') a
    ON d.hadm_id = a.hadm_id
WHERE
    (d.icd_version = 10 AND d.icd_code LIKE 'I50%')
    OR
    (d.icd_version = 9  AND d.icd_code LIKE '428%');

-- Pre-existing HF: any HF-coded admission that was fully completed before first drug.
-- Uses dischtime (not admittime) as the cutoff because ICD codes are assigned at discharge.
-- Admissions that span the first drug time (admittime < first_drug_time < dischtime) are
-- excluded from this view — HF coded on those admissions may reflect post-drug development
-- and should not be treated as pre-existing. They are also excluded from incident_hf_events
-- (see note below); analysts should run a sensitivity analysis using dischtime > first_drug_time
-- to capture index-admission HF events.
CREATE OR REPLACE VIEW pre_existing_hf AS
SELECT DISTINCT
    subject_id,
    1 AS has_pre_existing_hf
FROM hf_diagnoses_all
WHERE dischtime IS NOT NULL
  AND dischtime <= first_drug_time;

-- Pre-existing cardiomyopathy: I42% (ICD-10) / 425% (ICD-9) before first drug.
-- CMP is the structural precursor to HF; patients with dilated CMP at baseline have
-- confounded attribution of downstream HF to drug. Used in the strict binary table.
-- ICD codes included:
--   I42.0 — Dilated cardiomyopathy (most directly linked to drug-induced CMP)
--   I42.x — Other cardiomyopathies (hypertrophic, restrictive, etc.)
--   425%  — ICD-9 equivalent (cardiomyopathy, all subtypes)
CREATE OR REPLACE VIEW pre_existing_cardiomyopathy AS
SELECT DISTINCT
    p.subject_id,
    1 AS has_pre_existing_cmp
FROM hf_patient_first_drug p
JOIN read_csv_auto('mimic-iv-3.1/hosp/diagnoses_icd.csv') d
    ON p.subject_id = d.subject_id
JOIN read_csv_auto('mimic-iv-3.1/hosp/admissions.csv') a
    ON d.hadm_id = a.hadm_id
WHERE (
    (d.icd_version = 10 AND d.icd_code LIKE 'I42%')
    OR
    (d.icd_version = 9  AND d.icd_code LIKE '425%')
)
AND a.dischtime IS NOT NULL
AND CAST(a.dischtime AS TIMESTAMP) <= p.first_drug_time;

-- Combined pre-existing flag: HF or CMP before first drug.
-- Used by hf_final_binary_modeling_table_strict in file 05.
CREATE OR REPLACE VIEW pre_existing_hf_or_cmp AS
SELECT
    p.subject_id,
    COALESCE(hf.has_pre_existing_hf,  0) AS has_pre_existing_hf,
    COALESCE(cmp.has_pre_existing_cmp, 0) AS has_pre_existing_cmp,
    CASE
        WHEN hf.has_pre_existing_hf  = 1
          OR cmp.has_pre_existing_cmp = 1
        THEN 1 ELSE 0
    END AS has_pre_existing_hf_or_cmp
FROM (SELECT DISTINCT subject_id FROM hf_patient_first_drug) p
LEFT JOIN pre_existing_hf          hf  ON p.subject_id = hf.subject_id
LEFT JOIN pre_existing_cardiomyopathy cmp ON p.subject_id = cmp.subject_id;

-- Incident HF: first HF-coded admission that started after cohort entry (first target drug).
-- One row per patient — the earliest qualifying HF admission.
-- NOTE: HF coded on the "index admission" (where admittime < first_drug_time < dischtime)
-- is not captured here. Because ICD codes are assigned at discharge, HF that develops
-- post-drug during the index admission cannot be reliably distinguished from pre-treatment
-- HF without within-admission timestamps. This is conservative; see pre_existing_hf for
-- the complementary exclusion.
-- QUALIFY tiebreaker: on identical admittime, pick lowest hadm_id (deterministic).
CREATE OR REPLACE VIEW incident_hf_events AS
SELECT
    subject_id,
    admittime AS hf_event_time,
    hadm_id   AS hf_hadm_id,
    string_agg(DISTINCT icd_code, ' | ' ORDER BY icd_code) AS hf_icd_codes
FROM hf_diagnoses_all
WHERE admittime > first_drug_time
GROUP BY subject_id, admittime, hadm_id
QUALIFY ROW_NUMBER() OVER (
    PARTITION BY subject_id
    ORDER BY admittime ASC,
             hadm_id   ASC   -- stable tiebreaker: deterministic when two admissions share the same timestamp
) = 1;
