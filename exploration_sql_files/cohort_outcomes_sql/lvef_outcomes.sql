-- Obtain all LVEF studies
CREATE VIEW all_lvef AS
SELECT
    subject_id, -- patient identifier
    measurement_datetime,-- time of echo measurement

    TRY_CAST(result AS DOUBLE) AS lvef_value  -- safely convert text → number
                                              -- invalid strings become NULL instead of crashing
FROM read_csv_auto('mimic-iv-echo/structured-measurement.csv')

WHERE measurement = 'lvef' -- only keep LVEF measurements (ignore other echo metrics)
  AND result IS NOT NULL; -- remove missing values



-- Get a baseline LVEF
CREATE VIEW baseline_lvef AS
SELECT
    c.subject_id,  -- cancer patient ID
    c.first_oncology_time, -- index time (first drug exposure)

    a.measurement_datetime AS baseline_time,  -- timestamp of baseline echo
    a.lvef_value AS baseline_lvef  -- baseline heart function
FROM cancer_first_drug c
LEFT JOIN all_lvef a   -- LEFT JOIN keeps patients even if no baseline exists
    ON c.subject_id = a.subject_id
   AND a.measurement_datetime < c.first_oncology_time  -- only pre-treatment echos
   AND a.measurement_datetime >= c.first_oncology_time - INTERVAL '1 year' -- define how far we want to look back for pre-treatment echos

QUALIFY ROW_NUMBER() OVER (
    PARTITION BY c.subject_id -- one baseline per patient
    ORDER BY a.measurement_datetime DESC  -- pick most recent pre-drug echo
) = 1;



-- Get follow-up LVEFs (1 year window after first drug exposure)
CREATE VIEW followup_lvef AS
SELECT
    c.subject_id,  -- patient ID
    c.first_oncology_time,   -- index time

    a.measurement_datetime, -- follow-up echo time
    a.lvef_value            -- follow-up heart function
FROM cancer_first_drug c
JOIN all_lvef a    -- only patients with LVEF data
    ON c.subject_id = a.subject_id

WHERE a.measurement_datetime >= c.first_oncology_time   -- after drug start
  AND a.measurement_datetime < c.first_oncology_time + INTERVAL '1 year';  -- within 1-year window




-- LVEF outcome components within 1 year after first oncology drug
CREATE OR REPLACE VIEW cancer_lvef_outcomes AS
SELECT
    b.subject_id,
    b.first_oncology_time,

    -- Baseline LVEF info
    b.baseline_time,
    b.baseline_lvef,

    -- Availability flags
    CASE 
        WHEN b.baseline_lvef IS NOT NULL THEN 1 
        ELSE 0 
    END AS has_baseline_lvef,

    CASE 
        WHEN COUNT(f.lvef_value) > 0 THEN 1 
        ELSE 0 
    END AS has_followup_lvef_1yr,

    -- Follow-up LVEF summary
    MIN(f.lvef_value) AS worst_followup_lvef_1yr,

    MIN(f.measurement_datetime) AS first_followup_lvef_time_1yr,

    -- Component 1: absolute LVEF drop of at least 10 percentage points
    MAX(
        CASE
            WHEN f.lvef_value IS NOT NULL
             AND b.baseline_lvef IS NOT NULL
             AND (b.baseline_lvef - f.lvef_value) >= 10
            THEN 1 ELSE 0
        END
    ) AS lvef_drop_10_1yr,

    -- Component 2: follow-up LVEF below 50%
    MAX(
        CASE
            WHEN f.lvef_value IS NOT NULL
             AND f.lvef_value < 50
            THEN 1 ELSE 0
        END
    ) AS lvef_below_50_1yr,

    -- Strict LVEF-defined cardiotoxicity:
    -- drop >= 10 percentage points AND follow-up LVEF < 50%
    MAX(
        CASE
            WHEN f.lvef_value IS NOT NULL
             AND b.baseline_lvef IS NOT NULL
             AND (b.baseline_lvef - f.lvef_value) >= 10
             AND f.lvef_value < 50
            THEN 1 ELSE 0
        END
    ) AS definite_lvef_ctrcd_1yr

FROM baseline_lvef b
LEFT JOIN followup_lvef f
    ON b.subject_id = f.subject_id

GROUP BY
    b.subject_id,
    b.first_oncology_time,
    b.baseline_time,
    b.baseline_lvef;

-- Returns columns:
        -- has_baseline_lvef
        -- has_followup_lvef_1yr
        -- lvef_drop_10_1yr
        -- lvef_below_50_1yr
        -- definite_lvef_ctrcd_1yr