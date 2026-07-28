-- 01_drug_classification.sql
--
-- Classify pharmacy records into the three target drug classes and anchor
-- each patient to their first exposure of each class.
--
-- Requires:
--   oncology_drugs       (pharmacy records pre-filtered to oncology agents by run_cohort.py)
--   all_cancer_patients  (cancer diagnosis anchor)
--
-- Main outputs:
--   hf_drugs_classified
--   hf_cohort_drug_starts
--   hf_patient_first_drug
--   hf_patient_first_drug_per_class

-- Classify drug records into the three HF-relevant classes.
-- Records that do not match any class are NULL and are excluded downstream.
--
-- Drug name coverage notes:
--   Anthracyclines: doxorubicin (incl. liposomal Doxil/Caelyx), daunorubicin (incl. liposomal
--     DaunoXome), epirubicin, idarubicin, mitoxantrone (anthracenedione, similar mechanism).
--   HER2-targeted: IV monoclonal antibodies (trastuzumab, pertuzumab, margetuximab), ADCs
--     (ado-trastuzumab emtansine/Kadcyla, trastuzumab deruxtecan/Enhertu), and oral TKIs
--     (lapatinib, neratinib, tucatinib). Oral agents may be underrepresented in MIMIC-IV
--     inpatient pharmacy records.
--   ICIs: All approved PD-1 (pembrolizumab, nivolumab, cemiplimab), PD-L1 (atezolizumab,
--     durvalumab, avelumab), and CTLA-4 (ipilimumab, tremelimumab) inhibitors.
CREATE OR REPLACE VIEW hf_drugs_classified AS
SELECT
    subject_id,
    hadm_id,
    pharmacy_id,
    CAST(starttime AS TIMESTAMP) AS starttime,
    CAST(stoptime  AS TIMESTAMP) AS stoptime,
    LOWER(drug) AS drug,
    CASE
        WHEN regexp_matches(LOWER(drug),
            'doxorubicin|adriamycin|doxil|caelyx|myocet|'
            'daunorubicin|cerubidine|daunoxome|'
            'epirubicin|ellence|pharmorubicin|'
            'idarubicin|idamycin|zavedos|'
            'mitoxantrone|novantrone'
        ) THEN 'anthracycline'

        WHEN regexp_matches(LOWER(drug),
            'trastuzumab|herceptin|'
            'pertuzumab|perjeta|'
            'ado[- ]?trastuzumab|kadcyla|emtansine|'
            'trastuzumab.deruxtecan|enhertu|'
            'margetuximab|margenza|'
            'lapatinib|tykerb|'
            'neratinib|nerlynx|'
            'tucatinib|tukysa'
        ) THEN 'her2_targeted'

        WHEN regexp_matches(LOWER(drug),
            'pembrolizumab|keytruda|'
            'nivolumab|opdivo|'
            'cemiplimab|libtayo|'
            'atezolizumab|tecentriq|'
            'durvalumab|imfinzi|'
            'avelumab|bavencio|'
            'ipilimumab|yervoy|'
            'tremelimumab|imjudo'
        ) THEN 'immune_checkpoint_inhibitor'

        ELSE NULL
    END AS drug_class
FROM oncology_drugs
WHERE starttime IS NOT NULL;

-- Target-class drug records for cancer patients only
CREATE OR REPLACE VIEW hf_cohort_drug_starts AS
SELECT d.*
FROM hf_drugs_classified d
INNER JOIN all_cancer_patients c ON d.subject_id = c.subject_id
WHERE d.drug_class IS NOT NULL;

-- Cohort entry: first administration of any target drug per patient
CREATE OR REPLACE VIEW hf_patient_first_drug AS
SELECT
    subject_id,
    MIN(starttime) AS first_drug_time
FROM hf_cohort_drug_starts
GROUP BY subject_id;

-- Per-class anchors: first and last dose of each target class per patient
-- Used for class-specific monitoring window calculations if needed
CREATE OR REPLACE VIEW hf_patient_first_drug_per_class AS
SELECT
    subject_id,
    drug_class,
    MIN(starttime) AS first_class_drug_time,
    MAX(starttime) AS last_class_drug_time
FROM hf_cohort_drug_starts
GROUP BY subject_id, drug_class;
