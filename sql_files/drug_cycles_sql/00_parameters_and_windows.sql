-- 00_parameters_and_windows.sql
--
-- Parameters and drug-specific toxicity windows.

CREATE OR REPLACE TEMP VIEW cycle_label_params AS
SELECT
    7::INTEGER AS cycle_gap_days,
    365::INTEGER AS default_window_days,
    365::INTEGER AS baseline_lookback_days;

CREATE OR REPLACE VIEW drug_toxicity_windows AS
SELECT *
FROM (
    VALUES
        ('anthracycline', 365, 'Early-onset CTRCD/HF commonly assessed within 1 year'),
        ('immune_checkpoint_inhibitor', 90, 'ICI myocarditis usually occurs early, often within weeks to 3 months'),
        ('her2_targeted', 365, 'HER2-related LV dysfunction commonly monitored over months to 1 year'),
        ('taxane', 90, 'Shorter window for acute/subacute CV events'),
        ('fluoropyrimidine', 30, 'Often acute/subacute ischemia/vasospasm window'),
        ('vegf_inhibitor', 180, 'Hypertension/HF/ischemic risk over months'),
        ('egfr_inhibitor', 180, 'General CV surveillance window'),
        ('tyrosine_kinase_inhibitor', 180, 'General CV surveillance window'),
        ('proteasome_inhibitor', 180, 'HF/ischemia/arrhythmia risk over months'),
        ('immunomodulatory_agent', 180, 'Thrombotic/CV risk over months'),
        ('other_oncology', 365, 'Fallback window')
) AS t(drug_class, toxicity_window_days, window_rationale);
