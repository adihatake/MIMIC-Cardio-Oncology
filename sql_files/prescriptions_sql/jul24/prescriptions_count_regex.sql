-- Get patients who have the following oncology drugs
CREATE OR REPLACE VIEW oncology_drugs AS
SELECT
    subject_id,
    hadm_id,
    pharmacy_id,
    starttime,
    stoptime,
    LOWER(drug) AS drug
FROM read_csv_auto('mimic-iv-3.1/hosp/prescriptions.csv')
WHERE drug IS NOT NULL
  AND starttime IS NOT NULL
  AND regexp_matches(
        LOWER(drug),
        -- anthracyclines
        'doxorubicin|daunorubicin|epirubicin|idarubicin'
        -- immune checkpoint inhibitors
        '|nivolumab|pembrolizumab|atezolizumab|ipilimumab|durvalumab|avelumab|cemiplimab'
        -- HER2-targeted
        '|trastuzumab|pertuzumab|ado-trastuzumab|emtansine'
        -- taxanes
        '|paclitaxel|docetaxel|cabazitaxel'
        -- fluoropyrimidines
        '|fluorouracil|5-fluorouracil|capecitabine'
        -- VEGF inhibitors
        '|bevacizumab|aflibercept|ramucirumab'
        -- EGFR inhibitors
        '|cetuximab|panitumumab'
        -- tyrosine kinase inhibitors
        '|sunitinib|imatinib|dasatinib|nilotinib|ponatinib|sorafenib|pazopanib|cabozantinib|axitinib|lenvatinib'
        -- proteasome inhibitors
        '|bortezomib|carfilzomib|ixazomib'
        -- immunomodulatory agents
        '|lenalidomide|thalidomide|pomalidomide'
      );

 -- Returns: 2565
--SELECT DISTINCT subject_id FROM oncology_drugs;