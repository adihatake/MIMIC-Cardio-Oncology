-- Load prescriptions and filter based on oncology-specific drugs administered

CREATE TABLE oncology_drugs AS
SELECT 
    subject_id,
    hadm_id,
    pharmacy_id,
    starttime,
    stoptime,
    LOWER(drug) AS drug
FROM read_csv_auto("mimic-iv-3.1/hosp/prescriptions.csv")

-- Search and filter by drugs administered
WHERE drug IS NOT NULL -- basic cleaning
  AND starttime IS NOT NULL 
  AND ( -- check for all of these drugs
        LOWER(drug) LIKE '%doxorubicin%'
     OR LOWER(drug) LIKE '%daunorubicin%'
     OR LOWER(drug) LIKE '%epirubicin%'
     OR LOWER(drug) LIKE '%cyclophosphamide%'
     OR LOWER(drug) LIKE '%cisplatin%'
     OR LOWER(drug) LIKE '%carboplatin%'
     OR LOWER(drug) LIKE '%5-fu%'
     OR LOWER(drug) LIKE '%fluorouracil%'
     OR LOWER(drug) LIKE '%capecitabine%'
     OR LOWER(drug) LIKE '%trastuzumab%'
     OR LOWER(drug) LIKE '%paclitaxel%'
     OR LOWER(drug) LIKE '%docetaxel%'
     OR LOWER(drug) LIKE '%sunitinib%'
     OR LOWER(drug) LIKE '%sorafenib%'
     OR LOWER(drug) LIKE '%imatinib%'
  );

-- Returns: 3222 patients 
SELECT DISTINCT subject_id FROM oncology_drugs;