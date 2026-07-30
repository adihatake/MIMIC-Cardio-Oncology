-- 03_cv_events.sql  (narrow_cv)
--
-- Identify cardiovascular events from MIMIC-IV ICD codes.
-- Separates pre-existing CV disease (before cohort entry) from incident CV events (after first drug).
--
-- ── INCIDENT ENDPOINT (narrow acute CV) ───────────────────────────────────────
-- Captures acute, drug-attributable cardiotoxic presentations only.
-- Arrhythmia and conduction disorder codes are intentionally EXCLUDED because
-- they are chronic comorbidities that recur on nearly every admission (e.g.
-- pre-existing AF coded as I48% on every hospitalization), making them
-- indistinguishable from new drug-induced events without longitudinal pre-drug
-- history adjudication. Analysis of pan_cancer_ctrcd_v3 showed that 723 of 1232
-- positives (59%) had no HF, MI, or CMP code — the majority driven by I48% (AF),
-- ICD-9 42789 (other dysrhythmia), ICD-9 42731 (AF), and I47-49%.
--
--   ICD-10 INCLUDED:
--     I21-23%  — MI (STEMI, NSTEMI, subsequent MI, acute MI complications)
--     I30-31%  — Pericarditis, pericardial effusion / tamponade
--     I40-41%  — Acute myocarditis
--     I42%     — Cardiomyopathy (dilated, hypertrophic, restrictive, etc.)
--     I46%     — Cardiac arrest
--     I50%     — Heart failure (all subtypes)
--     I514%    — Cardiogenic shock
--
--   ICD-10 EXCLUDED (vs pan_cancer_ctrcd):
--     I44%     — AV block / conduction disorders    ← chronic, recurs every admission
--     I45%     — Other conduction disorders          ← chronic, recurs every admission
--     I47%     — Paroxysmal tachycardia (SVT, VT)   ← frequently pre-existing
--     I48%     — Atrial fibrillation / flutter       ← most common chronic arrhythmia
--     I49%     — Other cardiac arrhythmias (VF, PVCs) ← frequently pre-existing
--
--   ICD-9 INCLUDED:
--     410-411% — Acute MI, unstable angina
--     420-423% — Pericarditis, endocarditis, myocarditis, pericardial disease
--     425%     — Cardiomyopathy
--     428%     — Heart failure
--
--   ICD-9 EXCLUDED (vs pan_cancer_ctrcd):
--     426%     — Conduction disorders               ← chronic, recurs every admission
--     427%     — Cardiac arrhythmias (AF, VT, VF)   ← frequently pre-existing
--
-- ── PRE-EXISTING EXCLUSION (strict binary table) ──────────────────────────────
-- Pre-existing HF  : I50% / 428%   → pre_existing_hf
-- Pre-existing CMP : I42% / 425%   → pre_existing_cardiomyopathy
-- Combined flag    : pre_existing_hf_or_cmp → drives the strict binary table.
--
-- Main outputs:
--   hf_diagnoses_all            (HF + CMP ICD records — for pre-existing checks only)
--   cv_diagnoses_all            (narrow acute CV ICD records — for incident event detection)
--   pre_existing_hf             (HF before cohort entry)
--   pre_existing_cardiomyopathy (CMP before cohort entry)
--   pre_existing_hf_or_cmp      (combined flag used by the strict binary table)
--   incident_cv_events          (first narrow acute CV event after cohort entry per patient)

-- ── HF + CMP diagnoses (for pre-existing checks only) ────────────────────────
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
    (d.icd_version = 10 AND (d.icd_code LIKE 'I50%' OR d.icd_code LIKE 'I42%'))
    OR
    (d.icd_version = 9  AND (d.icd_code LIKE '428%' OR d.icd_code LIKE '425%'));

-- Pre-existing HF: any HF-coded admission discharged before first drug.
-- Uses dischtime (ICD codes assigned at discharge). Admissions spanning the first
-- drug time are excluded — HF coded then may reflect post-drug development.
CREATE OR REPLACE VIEW pre_existing_hf AS
SELECT DISTINCT
    subject_id,
    1 AS has_pre_existing_hf
FROM hf_diagnoses_all
WHERE (
    (icd_version = 10 AND icd_code LIKE 'I50%')
    OR (icd_version = 9  AND icd_code LIKE '428%')
)
AND dischtime IS NOT NULL
AND dischtime <= first_drug_time;

-- Pre-existing cardiomyopathy: I42% / 425% discharged before first drug.
CREATE OR REPLACE VIEW pre_existing_cardiomyopathy AS
SELECT DISTINCT
    subject_id,
    1 AS has_pre_existing_cmp
FROM hf_diagnoses_all
WHERE (
    (icd_version = 10 AND icd_code LIKE 'I42%')
    OR (icd_version = 9  AND icd_code LIKE '425%')
)
AND dischtime IS NOT NULL
AND dischtime <= first_drug_time;

-- Combined pre-existing flag (HF or CMP) used by the strict binary table.
CREATE OR REPLACE VIEW pre_existing_hf_or_cmp AS
SELECT
    p.subject_id,
    COALESCE(hf.has_pre_existing_hf,   0) AS has_pre_existing_hf,
    COALESCE(cmp.has_pre_existing_cmp, 0) AS has_pre_existing_cmp,
    CASE
        WHEN hf.has_pre_existing_hf  = 1
          OR cmp.has_pre_existing_cmp = 1
        THEN 1 ELSE 0
    END AS has_pre_existing_hf_or_cmp
FROM (SELECT DISTINCT subject_id FROM hf_patient_first_drug) p
LEFT JOIN pre_existing_hf             hf  ON p.subject_id = hf.subject_id
LEFT JOIN pre_existing_cardiomyopathy cmp ON p.subject_id = cmp.subject_id;

-- ── Narrow acute CV diagnoses (for incident event detection) ──────────────────
-- Arrhythmia/conduction codes (I44-45%, I47-49%, ICD-9 426-427%) are excluded.
-- See file header for rationale.
CREATE OR REPLACE VIEW cv_diagnoses_all AS
SELECT
    p.subject_id,
    p.first_drug_time,
    a.hadm_id,
    CAST(a.admittime AS TIMESTAMP) AS admittime,
    d.icd_code,
    d.icd_version
FROM hf_patient_first_drug p
JOIN read_csv_auto('mimic-iv-3.1/hosp/diagnoses_icd.csv') d
    ON p.subject_id = d.subject_id
JOIN read_csv_auto('mimic-iv-3.1/hosp/admissions.csv') a
    ON d.hadm_id = a.hadm_id
WHERE
    (d.icd_version = 10 AND (
        d.icd_code LIKE 'I21%'   -- STEMI / NSTEMI
        OR d.icd_code LIKE 'I22%'  -- Subsequent MI
        OR d.icd_code LIKE 'I23%'  -- Acute MI complications
        OR d.icd_code LIKE 'I30%'  -- Acute pericarditis
        OR d.icd_code LIKE 'I31%'  -- Pericardial effusion / tamponade
        OR d.icd_code LIKE 'I40%'  -- Acute myocarditis
        OR d.icd_code LIKE 'I41%'  -- Myocarditis in diseases classified elsewhere
        OR d.icd_code LIKE 'I42%'  -- Cardiomyopathy
        -- I44% (AV block / conduction) EXCLUDED — chronic, recurs every admission
        -- I45% (other conduction)      EXCLUDED — chronic, recurs every admission
        OR d.icd_code LIKE 'I46%'  -- Cardiac arrest
        -- I47% (paroxysmal tachycardia) EXCLUDED — frequently pre-existing
        -- I48% (AF / flutter)           EXCLUDED — most common chronic arrhythmia
        -- I49% (other arrhythmias)      EXCLUDED — frequently pre-existing
        OR d.icd_code LIKE 'I50%'  -- Heart failure
        OR d.icd_code LIKE 'I514%' -- Cardiogenic shock
    ))
    OR
    (d.icd_version = 9 AND (
        d.icd_code LIKE '410%'  -- Acute MI
        OR d.icd_code LIKE '411%'  -- Unstable angina / other acute ischemic
        -- 412-414 excluded: chronic codes re-assigned on every admission.
        OR d.icd_code LIKE '420%'  -- Acute pericarditis
        OR d.icd_code LIKE '421%'  -- Acute/subacute endocarditis
        OR d.icd_code LIKE '422%'  -- Acute myocarditis
        OR d.icd_code LIKE '423%'  -- Other pericardial disease (effusion, tamponade)
        OR d.icd_code LIKE '425%'  -- Cardiomyopathy
        -- 426% (conduction disorders) EXCLUDED — chronic, recurs every admission
        -- 427% (arrhythmias)          EXCLUDED — frequently pre-existing
        OR d.icd_code LIKE '428%'  -- Heart failure
    ));

-- Incident CV event: first narrow acute CV-coded admission after cohort entry.
-- One row per patient — the earliest qualifying admission.
-- Column names retain the hf_ prefix for downstream compatibility with
-- 05_combined_cardiotox_event.sql and 07_final_modeling_table.sql.
-- QUALIFY tiebreaker: on identical admittime, pick lowest hadm_id (deterministic).
CREATE OR REPLACE VIEW incident_cv_events AS
SELECT
    subject_id,
    admittime AS hf_event_time,
    hadm_id   AS hf_hadm_id,
    string_agg(DISTINCT icd_code, ' | ' ORDER BY icd_code) AS hf_icd_codes
FROM cv_diagnoses_all
WHERE admittime > first_drug_time
GROUP BY subject_id, admittime, hadm_id
QUALIFY ROW_NUMBER() OVER (
    PARTITION BY subject_id
    ORDER BY admittime ASC,
             hadm_id   ASC
) = 1;
