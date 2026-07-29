-- 00_parameters.sql  (pan_cancer_bimodal)
--
-- Parameters for the pan-cancer CTRCD cohort with two-tier toxicity windows.
-- Covers 10 cardiotoxic drug classes grouped by onset mechanism:
--
--   Acute  ( 90 days): immune_checkpoint_inhibitor, taxane, fluoropyrimidine
--     — rapid-onset mechanisms: ICI myocarditis (median 28-43 days),
--       taxane arrhythmia, fluoropyrimidine coronary vasospasm.
--
--   Moderate (365 days): anthracycline, her2_targeted, vegf_inhibitor,
--     egfr_inhibitor, tyrosine_kinase_inhibitor, proteasome_inhibitor,
--     immunomodulatory_agent
--     — cumulative or sustained cardiotoxicity over months.
--
-- Use pan_cancer_ctrcd for class-specific windows,
-- or pan_cancer_ctrcd (with ICI at 90d) as a reference.

CREATE OR REPLACE TEMP VIEW hf_cohort_params AS
SELECT
    7::INTEGER   AS cycle_gap_days,        -- Days between prescription starts to mark a new cycle
    365::INTEGER AS baseline_lookback_days; -- Days before first drug to look for pre-existing HF

CREATE OR REPLACE VIEW hf_drug_toxicity_windows AS
SELECT *
FROM (
    VALUES
        -- Moderate window (365 days): cumulative / sustained toxicity
        ('anthracycline',             365, 'moderate window: cumulative dose-dependent cardiotoxicity'),
        ('her2_targeted',             365, 'moderate window: reversible LV dysfunction over months'),
        ('vegf_inhibitor',            365, 'moderate window: hypertension and HF risk over months'),
        ('egfr_inhibitor',            365, 'moderate window: sustained CV surveillance'),
        ('tyrosine_kinase_inhibitor', 365, 'moderate window: sustained CV surveillance'),
        ('proteasome_inhibitor',      365, 'moderate window: HF and arrhythmia risk over months'),
        ('immunomodulatory_agent',    365, 'moderate window: thrombotic and CV risk over months'),
        -- Acute window (90 days): rapid-onset toxicity
        ('immune_checkpoint_inhibitor', 90, 'acute window: myocarditis median onset 28-43 days'),
        ('taxane',                      90, 'acute window: acute arrhythmia and hypersensitivity'),
        ('fluoropyrimidine',            90, 'acute window: coronary vasospasm, typically within weeks')
) AS t(drug_class, toxicity_window_days, window_rationale);
