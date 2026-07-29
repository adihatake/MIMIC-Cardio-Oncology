-- 00_parameters.sql
--
-- Parameters for the HF-specific cardiotoxicity cohort.
-- Restricted to: anthracycline, HER2-targeted, immune checkpoint inhibitor.
--
-- Monitoring windows are aligned with ESC 2022 Cardio-Oncology Guidelines
-- (doi:10.1093/eurheartj/ehac244) and ASCO cardio-oncology statements.

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
            -- ESC 2022: Echo at end of treatment, 3 months, and 12 months post-completion for
            -- moderate/high-risk patients (cumulative dose >300 mg/m² doxorubicin equivalent,
            -- mediastinal RT, or baseline LV dysfunction). High-risk: annually thereafter.
            -- This 365-day window captures early-onset cardiotoxic HF; late-onset (>1 year,
            -- common in pediatric/AYA survivors) requires extended follow-up not modeled here.
            'ESC 2022: >=12 months post-treatment; late-onset HF not captured in this window'
        ),
        (
            'her2_targeted',
            365,
            -- ESC 2022: Echo every 3 months during active HER2-targeted therapy, then at
            -- 3 months post-completion. HER2 cardiotoxicity is Type II (reversible): LV
            -- dysfunction typically recovers within months of discontinuation.
            -- 365-day window covers active treatment + post-treatment monitoring period.
            'ESC 2022: q3 months during treatment + 12 months post-completion'
        ),
        (
            'immune_checkpoint_inhibitor',
            90,
            -- Literature reports ICI cardiac event onset at 28–43 days (median), with >90% of
            -- myocarditis cases occurring within 3 months of treatment start. A 90-day window
            -- captures the clinically relevant acute-onset window for ICI-related HF (myocarditis
            -- → cardiogenic shock / acute HF) without inflating the negative class with patients
            -- far removed from the acute immune-mediated injury period.
            -- ESC 2022 recommends monitoring up to 12 months post-discontinuation for all cardiac
            -- irAEs; if a broader window is desired, rerun with 365 days as a sensitivity analysis.
            'Onset literature: 28-43 day median; 90-day window captures acute myocarditis-related HF'
        )
) AS t(drug_class, toxicity_window_days, window_rationale);
