-- 00_parameters.sql  (pan_cancer_ctrcd)
--
-- Parameters for the pan-cancer CTRCD cohort.
-- Covers 10 cardiotoxic drug classes (see 01_drug_classification.sql).
--
-- Monitoring window: uniform 365 days for all drug classes.
-- This allows direct cross-class comparison of CTRCD rates without
-- the labeling asymmetry that arises from class-specific windows.
-- Use hf_cardiotox_v2 for ESC 2022 class-specific windows (ICI: 90d).

CREATE OR REPLACE TEMP VIEW hf_cohort_params AS
SELECT
    7::INTEGER   AS cycle_gap_days,        -- Days between prescription starts to mark a new cycle
    365::INTEGER AS baseline_lookback_days; -- Days before first drug to look for pre-existing HF

-- Uniform 365-day monitoring window for all drug classes
CREATE OR REPLACE VIEW hf_drug_toxicity_windows AS
SELECT *
FROM (
    VALUES
        ('anthracycline',               365, 'uniform 365-day window'),
        ('her2_targeted',               365, 'uniform 365-day window'),
        ('immune_checkpoint_inhibitor', 365, 'uniform 365-day window (ESC 2022 acute window is 90d)'),
        ('taxane',                      365, 'uniform 365-day window'),
        ('fluoropyrimidine',            365, 'uniform 365-day window'),
        ('vegf_inhibitor',              365, 'uniform 365-day window'),
        ('egfr_inhibitor',              365, 'uniform 365-day window'),
        ('tyrosine_kinase_inhibitor',   365, 'uniform 365-day window'),
        ('proteasome_inhibitor',        365, 'uniform 365-day window'),
        ('immunomodulatory_agent',      365, 'uniform 365-day window')
) AS t(drug_class, toxicity_window_days, window_rationale);
