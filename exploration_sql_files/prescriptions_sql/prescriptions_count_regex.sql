-- Get patients who have the following oncology drugs
CREATE VIEW oncology_drugs AS
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
        'doxorubicin|daunorubicin|epirubicin|trastuzumab|paclitaxel|docetaxel|fluorouracil|5-fluorouracil|capecitabine|bevacizumab|cetuximab|sunitinib|imatinib|bortezomib|carfilzomib|lenalidomide|thalidomide|pomalidomide|nivolumab|pembrolizumab|atezolizumab'
      );

 -- Returns: 2565
--SELECT DISTINCT subject_id FROM oncology_drugs;