-- 01_drug_classification_and_first_drug.sql
--
-- Classify oncology drug rows and create first-drug anchors.
--
-- Requires:
--   oncology_drugs
--   all_cancer_patients
--   mimic-iv-3.1/hosp/diagnoses_icd.csv
--   mimic-iv-3.1/hosp/admissions.csv
--
-- Data quality checks applied:
--   1. patients_with_prior_cancer_dx — patient must have at least one
--      oncology ICD code (ICD-9 140–209/230–234 or ICD-10 C*/D0*) in an
--      admission whose admittime <= first_oncology_time.  Uses admission
--      START rather than discharge because MIMIC-IV codes diagnoses at
--      discharge; an admission that began before the first drug and
--      overlapped with it still reflects a pre-existing diagnosis.
--   2. Baseline LVEF >= 50 applied downstream in 03 and 06.

CREATE OR REPLACE VIEW oncology_drugs_classified AS
SELECT
    subject_id,
    hadm_id,
    pharmacy_id,
    CAST(starttime AS TIMESTAMP) AS starttime,
    CAST(stoptime AS TIMESTAMP) AS stoptime,
    LOWER(drug) AS drug,
    CASE
        WHEN regexp_matches(LOWER(drug), 'doxorubicin|daunorubicin|epirubicin|idarubicin')
        THEN 'anthracycline'

        WHEN regexp_matches(LOWER(drug), 'nivolumab|pembrolizumab|atezolizumab|ipilimumab|durvalumab|avelumab|cemiplimab')
        THEN 'immune_checkpoint_inhibitor'

        WHEN regexp_matches(LOWER(drug), 'trastuzumab|pertuzumab|ado-trastuzumab|emtansine')
        THEN 'her2_targeted'

        WHEN regexp_matches(LOWER(drug), 'paclitaxel|docetaxel|cabazitaxel')
        THEN 'taxane'

        WHEN regexp_matches(LOWER(drug), 'fluorouracil|5-fluorouracil|capecitabine')
        THEN 'fluoropyrimidine'

        WHEN regexp_matches(LOWER(drug), 'bevacizumab|aflibercept|ramucirumab')
        THEN 'vegf_inhibitor'

        WHEN regexp_matches(LOWER(drug), 'cetuximab|panitumumab')
        THEN 'egfr_inhibitor'

        WHEN regexp_matches(LOWER(drug), 'sunitinib|imatinib|dasatinib|nilotinib|ponatinib|sorafenib|pazopanib|cabozantinib|axitinib|lenvatinib')
        THEN 'tyrosine_kinase_inhibitor'

        WHEN regexp_matches(LOWER(drug), 'bortezomib|carfilzomib|ixazomib')
        THEN 'proteasome_inhibitor'

        WHEN regexp_matches(LOWER(drug), 'lenalidomide|thalidomide|pomalidomide')
        THEN 'immunomodulatory_agent'

        ELSE 'other_oncology'
    END AS drug_class
FROM oncology_drugs
WHERE starttime IS NOT NULL;

CREATE OR REPLACE VIEW first_oncology_drug AS
SELECT
    subject_id,
    MIN(starttime) AS first_oncology_time
FROM oncology_drugs_classified
GROUP BY subject_id;

-- Data quality check 1: cancer diagnosis must exist in an admission that
-- started on or before the first oncology drug administration.
CREATE OR REPLACE VIEW patients_with_prior_cancer_dx AS
SELECT DISTINCT
    f.subject_id
FROM first_oncology_drug f
JOIN read_csv_auto('mimic-iv-3.1/hosp/diagnoses_icd.csv') d
    ON f.subject_id = d.subject_id
JOIN read_csv_auto('mimic-iv-3.1/hosp/admissions.csv') a
    ON d.hadm_id = a.hadm_id
WHERE CAST(a.admittime AS TIMESTAMP) <= f.first_oncology_time
  AND (
      -- ICD-9: malignant neoplasms (140–209), carcinoma in situ (230–234)
      (d.icd_version = 9 AND (
          d.icd_code LIKE '14%' OR d.icd_code LIKE '15%' OR d.icd_code LIKE '16%'
       OR d.icd_code LIKE '17%' OR d.icd_code LIKE '18%' OR d.icd_code LIKE '19%'
       OR d.icd_code LIKE '20%' OR d.icd_code LIKE '23%'
      ))
      OR
      -- ICD-10: malignant neoplasms (C00–C96), carcinoma in situ (D00–D09)
      (d.icd_version = 10 AND (
          d.icd_code LIKE 'C%' OR d.icd_code LIKE 'D0%'
      ))
  );

CREATE OR REPLACE VIEW cancer_first_drug AS
SELECT
    f.subject_id,
    f.first_oncology_time
FROM first_oncology_drug f
INNER JOIN all_cancer_patients c
    ON f.subject_id = c.subject_id
INNER JOIN patients_with_prior_cancer_dx dx
    ON f.subject_id = dx.subject_id;
