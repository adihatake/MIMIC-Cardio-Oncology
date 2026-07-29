-- 00_parameters.sql
--
-- Parameters for the HF-specific cardiotoxicity cohort.
-- Restricted to: anthracycline, HER2-targeted, immune checkpoint inhibitor.
--
-- Monitoring windows: uniform 365 days for all drug classes.
-- This is a sensitivity analysis variant of hf_cardiotox_v2, which uses the
-- ESC 2022 class-specific windows (ICI: 90d, anthracycline/HER2: 365d).
-- Using 365 days for ICI allows direct cross-class comparison of event rates
-- and avoids any labeling asymmetry introduced by the shorter ICI window.

CREATE OR REPLACE TEMP VIEW hf_cohort_params AS
SELECT
    7::INTEGER   AS cycle_gap_days,        -- Days between prescription starts to consider a new cycle
    365::INTEGER AS baseline_lookback_days; -- Days before first drug to look for pre-existing HF

-- Drug-class-specific monitoring windows (days after prediction_time to look for HF event)
CREATE OR REPLACE VIEW hf_drug_toxicity_windows AS
SELECT *
FROM (
    VALUES
        (
            'anthracycline',
            365,
            'uniform 365-day window (same as ESC 2022 window for this class)'
        ),
        (
            'her2_targeted',
            365,
            'uniform 365-day window (same as ESC 2022 window for this class)'
        ),
        (
            'immune_checkpoint_inhibitor',
            365,
            'uniform 365-day window — extended from ESC 2022 90-day acute window for sensitivity analysis'
        )
) AS t(drug_class, toxicity_window_days, window_rationale);
