-- 00_parameters_and_windows.sql  (pan_cancer_uniform_365)
--
-- Cohort parameters with a uniform 365-day monitoring window for all drug classes.
--
-- Unlike pan_cancer_per_cycle (ESC 2022 class-specific windows), this pipeline
-- assigns a fixed 365-day window from prediction_time for every cycle regardless
-- of which drug classes were administered.
--
-- Rationale: for risk stratification (will this patient have a cardiac event in
-- the next year?), a uniform window avoids the label contradictions that arise
-- when the same real-world event falls inside one cycle's window but outside
-- another's due to differing class-specific horizons.

CREATE OR REPLACE TEMP VIEW pcu_params AS
SELECT
    7::INTEGER   AS cycle_gap_days,              -- prescription gap (days) triggering a new cycle
    365::INTEGER AS baseline_lookback_days,       -- lookback window for pre-drug baseline echo
    365::INTEGER AS uniform_toxicity_window_days; -- uniform monitoring window applied to all cycles
