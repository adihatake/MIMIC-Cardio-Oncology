-- 00_parameters.sql  (anthracycline_only_exposure)
--
-- Parameters for the anthracycline-only cohort.
--
-- All 10 cardiotoxic drug classes are tracked internally so that:
--   (1) anthracycline_first_patients can exclude patients who received
--       any other cardiotoxic drug before their first anthracycline dose.
--   (2) has_concurrent_other_class can flag cycles where any non-anthracycline
--       cardiotoxic drug was co-administered.
--
-- The final modeling table retains only anthracycline cycles.
-- Toxicity windows for non-anthracycline classes are used only in the
-- intermediate cycle backbone and do not affect anthracycline labels.
--
-- Class-specific windows follow July28 reference queries and ESC 2022 guidance.

CREATE OR REPLACE TEMP VIEW hf_cohort_params AS
SELECT
    7::INTEGER   AS cycle_gap_days,        -- Days between prescription starts to mark a new cycle
    365::INTEGER AS baseline_lookback_days; -- Days before first drug to look for pre-existing HF

CREATE OR REPLACE VIEW hf_drug_toxicity_windows AS
SELECT *
FROM (
    VALUES
        ('anthracycline',               365, 'ESC 2022: >=12 months post-treatment'),
        ('her2_targeted',               365, 'ESC 2022: q3 months during treatment + 12 months post-completion'),
        ('immune_checkpoint_inhibitor',  90, 'Onset literature: 28-43 day median; 90-day acute window'),
        ('taxane',                       90, 'Shorter window for acute/subacute CV events'),
        ('fluoropyrimidine',             30, 'Acute/subacute ischemia and vasospasm window'),
        ('vegf_inhibitor',             180, 'Hypertension/HF/ischemic risk over months'),
        ('egfr_inhibitor',             180, 'General CV surveillance window'),
        ('tyrosine_kinase_inhibitor',  180, 'General CV surveillance window'),
        ('proteasome_inhibitor',       180, 'HF/ischemia/arrhythmia risk over months'),
        ('immunomodulatory_agent',     180, 'Thrombotic/CV risk over months')
) AS t(drug_class, toxicity_window_days, window_rationale);
