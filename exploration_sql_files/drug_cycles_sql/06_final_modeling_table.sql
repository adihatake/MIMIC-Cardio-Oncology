-- 06_final_modeling_table.sql
--
-- Final cycle-level modelling labels.
--
-- Main outputs:
--   final_cycle_modeling_table
--   final_cycle_binary_modeling_table

CREATE OR REPLACE VIEW final_cycle_modeling_table AS
SELECT
    e.subject_id,
    e.cycle_number,
    e.prediction_time,
    e.cycle_start_date,
    e.cycle_end_date,
    e.n_exposure_start_days_in_cycle,
    e.n_prescription_rows_in_cycle,

    e.drugs_in_cycle,
    e.drug_classes_in_cycle,
    e.toxicity_window_days,
    e.prediction_time + e.toxicity_window_days * INTERVAL '1 day' AS prediction_window_end,
    e.window_rationales,

    e.exposed_anthracycline,
    e.exposed_immune_checkpoint_inhibitor,
    e.exposed_her2_targeted,
    e.exposed_taxane,
    e.exposed_fluoropyrimidine,
    e.exposed_vegf_inhibitor,
    e.exposed_egfr_inhibitor,
    e.exposed_tyrosine_kinase_inhibitor,
    e.exposed_proteasome_inhibitor,
    e.exposed_immunomodulatory_agent,

    ft.first_toxicity_time,
    ft.first_toxicity_type,

    CASE
        WHEN ft.first_toxicity_time > e.prediction_time
         AND ft.first_toxicity_time <= e.prediction_time + e.toxicity_window_days * INTERVAL '1 day'
        THEN 1 ELSE 0
    END AS toxicity_in_window,

    COALESCE(h.pre_existing_cv_history, 0) AS pre_existing_cv_history,

    l.baseline_time,
    l.baseline_lvef,

    obs.last_observation_time,

    CASE
        WHEN obs.last_observation_time >= e.prediction_time + e.toxicity_window_days * INTERVAL '1 day'
        THEN 1 ELSE 0
    END AS observed_through_prediction_window,

    CASE
        WHEN EXISTS (
            SELECT 1
            FROM patient_observation_events oe
            WHERE oe.subject_id = e.subject_id
              AND oe.observation_time > e.prediction_time
              AND oe.observation_time <= e.prediction_time + e.toxicity_window_days * INTERVAL '1 day'
        )
        THEN 1 ELSE 0
    END AS has_observation_in_prediction_window,

    CASE
        WHEN ft.first_toxicity_time <= e.prediction_time
        THEN 0
        WHEN ft.first_toxicity_time > e.prediction_time
         AND ft.first_toxicity_time <= e.prediction_time + e.toxicity_window_days * INTERVAL '1 day'
        THEN 1
        WHEN obs.last_observation_time >= e.prediction_time + e.toxicity_window_days * INTERVAL '1 day'
        THEN 1
        ELSE 0
    END AS eligible_for_binary_label,

    CASE
        WHEN ft.first_toxicity_time <= e.prediction_time
        THEN NULL
        WHEN ft.first_toxicity_time > e.prediction_time
         AND ft.first_toxicity_time <= e.prediction_time + e.toxicity_window_days * INTERVAL '1 day'
        THEN 1
        WHEN obs.last_observation_time >= e.prediction_time + e.toxicity_window_days * INTERVAL '1 day'
        THEN 0
        ELSE NULL
    END AS binary_label,

    CASE
        WHEN ft.first_toxicity_time <= e.prediction_time
        THEN 'exclude_already_toxic'

        WHEN ft.first_toxicity_time > e.prediction_time
         AND ft.first_toxicity_time <= e.prediction_time + e.toxicity_window_days * INTERVAL '1 day'
        THEN 'positive'

        WHEN obs.last_observation_time >= e.prediction_time + e.toxicity_window_days * INTERVAL '1 day'
        THEN 'negative_observed'

        ELSE 'unknown_no_followup_evidence'
    END AS label

FROM oncology_cycle_exposures e
LEFT JOIN first_cardiotoxicity_event ft
    ON e.subject_id = ft.subject_id
LEFT JOIN pre_existing_cv_history h
    ON e.subject_id = h.subject_id
LEFT JOIN baseline_lvef_pre_first_drug l
    ON e.subject_id = l.subject_id
LEFT JOIN patient_last_observation obs
    ON e.subject_id = obs.subject_id;

CREATE OR REPLACE VIEW final_cycle_binary_modeling_table AS
SELECT *
FROM final_cycle_modeling_table
WHERE eligible_for_binary_label = 1
  AND label IN ('positive', 'negative_observed');
