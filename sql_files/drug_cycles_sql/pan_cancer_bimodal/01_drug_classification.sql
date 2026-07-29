-- 01_drug_classification.sql  (pan_cancer_ctrcd)
--
-- Classify pharmacy records into 10 cardiotoxic drug classes and anchor
-- each patient to their first exposure of each class.
--
-- Requires:
--   oncology_drugs       (pharmacy records pre-filtered by prescriptions_pan_cancer.sql)
--   all_cancer_patients  (cancer diagnosis anchor)
--
-- Main outputs:
--   hf_drugs_classified
--   hf_cohort_drug_starts
--   hf_patient_first_drug
--   hf_patient_first_drug_per_class

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

        WHEN regexp_matches(LOWER(drug),
            'paclitaxel|taxol|abraxane|'
            'docetaxel|taxotere|'
            'cabazitaxel|jevtana'
        ) THEN 'taxane'

        WHEN regexp_matches(LOWER(drug),
            'fluorouracil|5-fluorouracil|5fu|'
            'capecitabine|xeloda'
        ) THEN 'fluoropyrimidine'

        WHEN regexp_matches(LOWER(drug),
            'bevacizumab|avastin|'
            'aflibercept|zaltrap|eylea|'
            'ramucirumab|cyramza'
        ) THEN 'vegf_inhibitor'

        WHEN regexp_matches(LOWER(drug),
            'cetuximab|erbitux|'
            'panitumumab|vectibix'
        ) THEN 'egfr_inhibitor'

        WHEN regexp_matches(LOWER(drug),
            'sunitinib|sutent|'
            'imatinib|gleevec|glivec|'
            'dasatinib|sprycel|'
            'nilotinib|tasigna|'
            'ponatinib|iclusig|'
            'sorafenib|nexavar|'
            'pazopanib|votrient|'
            'cabozantinib|cabometyx|cometriq|'
            'axitinib|inlyta|'
            'lenvatinib|lenvima'
        ) THEN 'tyrosine_kinase_inhibitor'

        WHEN regexp_matches(LOWER(drug),
            'bortezomib|velcade|'
            'carfilzomib|kyprolis|'
            'ixazomib|ninlaro'
        ) THEN 'proteasome_inhibitor'

        WHEN regexp_matches(LOWER(drug),
            'lenalidomide|revlimid|'
            'thalidomide|thalomid|'
            'pomalidomide|pomalyst'
        ) THEN 'immunomodulatory_agent'

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
CREATE OR REPLACE VIEW hf_patient_first_drug_per_class AS
SELECT
    subject_id,
    drug_class,
    MIN(starttime) AS first_class_drug_time,
    MAX(starttime) AS last_class_drug_time
FROM hf_cohort_drug_starts
GROUP BY subject_id, drug_class;
