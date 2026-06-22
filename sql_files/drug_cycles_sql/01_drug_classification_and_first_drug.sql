-- 01_drug_classification_and_first_drug.sql
--
-- Classify oncology drug rows and create first-drug anchors.
--
-- Requires:
--   oncology_drugs
--   all_cancer_patients

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

CREATE OR REPLACE VIEW cancer_first_drug AS
SELECT
    f.subject_id,
    f.first_oncology_time
FROM first_oncology_drug f
INNER JOIN all_cancer_patients c
    ON f.subject_id = c.subject_id;
