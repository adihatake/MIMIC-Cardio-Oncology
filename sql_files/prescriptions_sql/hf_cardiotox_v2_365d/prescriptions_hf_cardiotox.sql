-- prescriptions_hf_cardiotox.sql
--
-- Pre-filter: pull all pharmacy records that could match the three target drug
-- classes in 01_drug_classification.sql. This view is intentionally broad —
-- final classification (with exact per-class regex) happens in file 01.
--
-- Covers:
--   Anthracyclines: doxorubicin (+ brands), daunorubicin, epirubicin,
--                   idarubicin, mitoxantrone (anthracenedione)
--   HER2-targeted:  trastuzumab, pertuzumab, ado-trastuzumab/Kadcyla,
--                   trastuzumab deruxtecan/Enhertu, margetuximab,
--                   lapatinib, neratinib, tucatinib (oral TKIs)
--   ICIs:           pembrolizumab, nivolumab, cemiplimab (PD-1),
--                   atezolizumab, durvalumab, avelumab (PD-L1),
--                   ipilimumab, tremelimumab (CTLA-4)
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
        'kadcyla|emtansine|'
        'enhertu|'
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
        'tremelimumab|imjudo'
      );
