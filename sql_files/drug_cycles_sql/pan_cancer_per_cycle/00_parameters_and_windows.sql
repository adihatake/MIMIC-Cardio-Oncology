-- 00_parameters_and_windows.sql  (pan_cancer_per_cycle)
--
-- Cohort parameters and drug-class-specific toxicity windows.
--
-- Unlike pan_cancer_ctrcd (uniform 365d window for all classes), this pipeline
-- uses ESC 2022 class-specific windows.  For multi-drug cycles, the final
-- modeling table applies the MAX window across all drugs received in that cycle
-- (conservative: the longest plausible attribution horizon).
--
-- ICI window: 90d (acute myocarditis onset); anthracycline: 365d (delayed CMP);
-- fluoropyrimidine: 30d (acute vasospasm).  See window_rationale column.

CREATE OR REPLACE TEMP VIEW pcr_params AS
SELECT
    7::INTEGER   AS cycle_gap_days,         -- prescription gap (days) triggering a new cycle
    365::INTEGER AS baseline_lookback_days;  -- lookback window for pre-drug baseline echo

CREATE OR REPLACE VIEW pcr_drug_toxicity_windows AS
SELECT *
FROM (
    VALUES
        ('anthracycline',               365, 'Early-onset and late CTRCD/HF assessed within 1 year (ESC 2022)'),
        ('her2_targeted',               365, 'HER2-related LV dysfunction monitored over 1 year (ESC 2022)'),
        ('immune_checkpoint_inhibitor',  90, 'ICI myocarditis mostly within weeks–3 months (ESC 2022)'),
        ('taxane',                        90, 'Acute/subacute CV events; shorter surveillance window'),
        ('fluoropyrimidine',              30, 'Acute vasospasm/ischemia; very early onset'),
        ('vegf_inhibitor',               180, 'Hypertension, HF, and ischemic risk over months'),
        ('egfr_inhibitor',               180, 'General cardio-oncology surveillance window'),
        ('tyrosine_kinase_inhibitor',    180, 'General cardio-oncology surveillance window'),
        ('proteasome_inhibitor',         180, 'HF, ischemia, arrhythmia risk over months'),
        ('immunomodulatory_agent',       180, 'Thrombotic and CV risk over months'),
        ('other_oncology',               365, 'Conservative fallback window')
) AS t(drug_class, toxicity_window_days, window_rationale);
