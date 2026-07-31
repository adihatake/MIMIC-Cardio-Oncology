-- 03_hf_and_preexisting.sql  (pan_cancer_per_cycle)
--
-- Heart failure endpoint and pre-existing cardiac exclusion.
--
-- Endpoint (ICD-based component):
--   ICD-10  I50%  — Heart failure, all subtypes
--   ICD-9   428%  — Heart failure, all subtypes
--
-- Pre-existing exclusion (strict binary table):
--   Pre-existing HF  : I50% / 428%   → pcr_pre_existing_hf
--   Pre-existing CMP : I42% / 425%   → pcr_pre_existing_cmp
--   Combined flag    : pcr_pre_existing_hf_or_cmp
--
-- Intentionally narrower than the broad CV endpoint used by pan_cancer_ctrcd.
-- This pipeline targets CTRCD specifically (HF + echo), not the full CV spectrum.
-- Echo-based events (LVEF CTRCD and GLS subclinical) are added in 04_echo_events.sql.
--
-- Main outputs:
--   pcr_hf_diagnoses_all          (HF + CMP ICD records for cohort patients)
--   pcr_pre_existing_hf           (HF before cohort entry)
--   pcr_pre_existing_cmp          (CMP before cohort entry)
--   pcr_pre_existing_hf_or_cmp    (combined pre-existing flag)
--   pcr_incident_hf_events        (first HF admission after cohort entry per patient)

-- All HF ICD diagnoses for cohort patients joined to admission timestamps
CREATE OR REPLACE VIEW pcr_hf_diagnoses_all AS
SELECT
    p.subject_id,
    p.first_drug_time,
    a.hadm_id,
    CAST(a.admittime AS TIMESTAMP) AS admittime,
    CAST(a.dischtime AS TIMESTAMP) AS dischtime,
    d.icd_code,
    d.icd_version
FROM pcr_patient_first_drug p
JOIN read_csv_auto('mimic-iv-3.1/hosp/diagnoses_icd.csv') d
    ON p.subject_id = d.subject_id
JOIN read_csv_auto('mimic-iv-3.1/hosp/admissions.csv') a
    ON d.hadm_id = a.hadm_id
WHERE
    (d.icd_version = 10 AND (d.icd_code LIKE 'I50%' OR d.icd_code LIKE 'I42%'))
    OR
    (d.icd_version = 9  AND (d.icd_code LIKE '428%' OR d.icd_code LIKE '425%'));

-- Pre-existing HF: HF-coded admission fully completed before first drug.
-- Uses dischtime (ICD codes assigned at discharge) as the cutoff.
CREATE OR REPLACE VIEW pcr_pre_existing_hf AS
SELECT DISTINCT
    subject_id,
    1 AS has_pre_existing_hf
FROM pcr_hf_diagnoses_all
WHERE icd_code LIKE 'I50%' OR icd_code LIKE '428%'
  AND dischtime IS NOT NULL
  AND dischtime <= first_drug_time;

-- Pre-existing cardiomyopathy: I42% (ICD-10) / 425% (ICD-9) before first drug.
-- Dilated CMP at baseline confounds drug-attributed downstream HF.
CREATE OR REPLACE VIEW pcr_pre_existing_cmp AS
SELECT DISTINCT
    subject_id,
    1 AS has_pre_existing_cmp
FROM pcr_hf_diagnoses_all
WHERE icd_code LIKE 'I42%' OR icd_code LIKE '425%'
  AND dischtime IS NOT NULL
  AND dischtime <= first_drug_time;

-- Combined pre-existing flag (HF or CMP) — used by the strict binary table.
CREATE OR REPLACE VIEW pcr_pre_existing_hf_or_cmp AS
SELECT
    p.subject_id,
    COALESCE(hf.has_pre_existing_hf,  0) AS has_pre_existing_hf,
    COALESCE(cmp.has_pre_existing_cmp, 0) AS has_pre_existing_cmp,
    CASE
        WHEN hf.has_pre_existing_hf  = 1
          OR cmp.has_pre_existing_cmp = 1
        THEN 1 ELSE 0
    END AS has_pre_existing_hf_or_cmp
FROM (SELECT DISTINCT subject_id FROM pcr_patient_first_drug) p
LEFT JOIN pcr_pre_existing_hf  hf  ON p.subject_id = hf.subject_id
LEFT JOIN pcr_pre_existing_cmp cmp ON p.subject_id = cmp.subject_id;

-- Incident HF: first HF-coded admission starting after cohort entry.
-- One row per patient (earliest qualifying admission).
-- HF coded on the index admission (admittime < first_drug_time < dischtime) is
-- excluded conservatively — cannot reliably distinguish pre- from post-drug HF
-- without within-admission timestamps.
CREATE OR REPLACE VIEW pcr_incident_hf_events AS
SELECT
    subject_id,
    admittime AS hf_event_time,
    hadm_id   AS hf_hadm_id,
    string_agg(DISTINCT icd_code, ' | ' ORDER BY icd_code) AS hf_icd_codes
FROM pcr_hf_diagnoses_all
WHERE (icd_code LIKE 'I50%' OR icd_code LIKE '428%')
  AND admittime > first_drug_time
GROUP BY subject_id, admittime, hadm_id
QUALIFY ROW_NUMBER() OVER (
    PARTITION BY subject_id
    ORDER BY admittime ASC, hadm_id ASC
) = 1;
