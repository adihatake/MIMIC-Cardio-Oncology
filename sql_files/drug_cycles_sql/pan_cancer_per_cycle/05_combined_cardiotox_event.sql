-- 05_combined_cardiotox_event.sql  (pan_cancer_per_cycle)
--
-- Unified CTRCD endpoint combining ICD-coded HF (file 03) and echo-based
-- cardiotoxicity (file 04) into a single earliest event per patient.
--
-- Positive = earliest of:
--   (1) HF admission  — ICD-10 I50%, ICD-9 428% (file 03)
--   (2) LVEF CTRCD    — drop >=10pp from normal baseline to <50% (file 04)
--   (3) GLS subclinical — >15% relative decrease from baseline GLS (file 04)
--
-- Follows the ESC 2022 CTRCD spectrum (doi:10.1093/eurheartj/ehac244):
--   Stage 1 (subclinical): GLS decrease alone
--   Stage 2 (mild):        asymptomatic LVEF drop
--   Stage 3-4 (moderate-severe): symptomatic HF admission
--
-- For patients without echo data, the endpoint falls back to ICD HF alone.
-- On tie: HF admission takes priority (most clinically severe).
--
-- Main output:
--   pcr_first_combined_cardiotox_event

CREATE OR REPLACE VIEW pcr_first_combined_cardiotox_event AS
SELECT
    p.subject_id,

    CASE
        WHEN hf.hf_event_time   IS NULL AND echo.first_echo_event_time IS NULL THEN NULL
        WHEN hf.hf_event_time   IS NULL                                         THEN echo.first_echo_event_time
        WHEN echo.first_echo_event_time IS NULL                                 THEN hf.hf_event_time
        WHEN hf.hf_event_time   <= echo.first_echo_event_time                  THEN hf.hf_event_time
        ELSE                                                                         echo.first_echo_event_time
    END AS first_event_time,

    CASE
        WHEN hf.hf_event_time   IS NULL AND echo.first_echo_event_time IS NULL THEN NULL
        WHEN hf.hf_event_time   IS NULL                                         THEN echo.first_echo_event_type
        WHEN echo.first_echo_event_time IS NULL                                 THEN 'hf_admission'
        WHEN hf.hf_event_time   <= echo.first_echo_event_time                  THEN 'hf_admission'
        ELSE                                                                         echo.first_echo_event_type
    END AS first_event_type,

    -- ICD HF details
    hf.hf_event_time,
    hf.hf_hadm_id,
    hf.hf_icd_codes,

    -- Echo CTRCD details
    echo.first_echo_event_time,
    echo.first_echo_event_type,
    echo.baseline_lvef,
    echo.event_lvef,
    echo.lvef_drop_pp,
    echo.baseline_gls,
    echo.event_gls,
    echo.gls_relative_decrease

FROM (SELECT DISTINCT subject_id FROM pcr_patient_first_drug) p
LEFT JOIN pcr_incident_hf_events          hf   ON p.subject_id = hf.subject_id
LEFT JOIN pcr_first_echo_cardiotox_event  echo ON p.subject_id = echo.subject_id;
