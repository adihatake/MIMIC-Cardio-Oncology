-- prescriptions_pan_cancer.sql  (pan_cancer_uniform_365)
--
-- Pre-filter: pull all pharmacy records for pan-cancer cardiotoxic drug classes.
-- Final classification with per-class regex happens in 01_drug_classification.sql.
--
-- Covers:
--   Anthracyclines:          doxorubicin, daunorubicin, epirubicin, idarubicin, mitoxantrone
--   HER2-targeted:           trastuzumab, pertuzumab, ado-trastuzumab, margetuximab,
--                            lapatinib, neratinib, tucatinib
--   ICIs:                    pembrolizumab, nivolumab, cemiplimab (PD-1),
--                            atezolizumab, durvalumab, avelumab (PD-L1),
--                            ipilimumab, tremelimumab (CTLA-4)
--   Taxanes:                 paclitaxel, docetaxel, cabazitaxel
--   Fluoropyrimidines:       fluorouracil, capecitabine
--   VEGF inhibitors:         bevacizumab, aflibercept, ramucirumab
--   EGFR inhibitors:         cetuximab, panitumumab
--   Tyrosine kinase inh.:    sunitinib, imatinib, dasatinib, nilotinib, ponatinib,
--                            sorafenib, pazopanib, cabozantinib, axitinib, lenvatinib
--   Proteasome inhibitors:   bortezomib, carfilzomib, ixazomib
--   Immunomodulatory agents: lenalidomide, thalidomide, pomalidomide
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
        'doxorubicin|adriamycin|doxil|caelyx|myocet|'
        'daunorubicin|cerubidine|daunoxome|'
        'epirubicin|ellence|pharmorubicin|'
        'idarubicin|idamycin|zavedos|'
        'mitoxantrone|novantrone|'
        'trastuzumab|herceptin|'
        'pertuzumab|perjeta|'
        'ado[- ]?trastuzumab|kadcyla|emtansine|enhertu|'
        'margetuximab|margenza|'
        'lapatinib|tykerb|'
        'neratinib|nerlynx|'
        'tucatinib|tukysa|'
        'pembrolizumab|keytruda|'
        'nivolumab|opdivo|'
        'cemiplimab|libtayo|'
        'atezolizumab|tecentriq|'
        'durvalumab|imfinzi|'
        'avelumab|bavencio|'
        'ipilimumab|yervoy|'
        'tremelimumab|imjudo|'
        'paclitaxel|taxol|abraxane|'
        'docetaxel|taxotere|'
        'cabazitaxel|jevtana|'
        'fluorouracil|5-fluorouracil|5fu|'
        'capecitabine|xeloda|'
        'bevacizumab|avastin|'
        'aflibercept|zaltrap|eylea|'
        'ramucirumab|cyramza|'
        'cetuximab|erbitux|'
        'panitumumab|vectibix|'
        'sunitinib|sutent|'
        'imatinib|gleevec|glivec|'
        'dasatinib|sprycel|'
        'nilotinib|tasigna|'
        'ponatinib|iclusig|'
        'sorafenib|nexavar|'
        'pazopanib|votrient|'
        'cabozantinib|cabometyx|cometriq|'
        'axitinib|inlyta|'
        'lenvatinib|lenvima|'
        'bortezomib|velcade|'
        'carfilzomib|kyprolis|'
        'ixazomib|ninlaro|'
        'lenalidomide|revlimid|'
        'thalidomide|thalomid|'
        'pomalidomide|pomalyst'
      );
