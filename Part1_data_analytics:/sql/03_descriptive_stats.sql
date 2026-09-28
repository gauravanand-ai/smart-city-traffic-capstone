-- Task 2.1: Descriptive statistics on traffic_volume (deduplicated hourly table)
SELECT
    COUNT(*) AS n,
    ROUND(AVG(traffic_volume), 2) AS mean,
    (SELECT ROUND(AVG(traffic_volume), 2) FROM
        (SELECT traffic_volume FROM traffic_hourly ORDER BY traffic_volume
         LIMIT 2 - (SELECT COUNT(*) FROM traffic_hourly) % 2
         OFFSET (SELECT (COUNT(*) - 1) / 2 FROM traffic_hourly))
    ) AS median,
    MIN(traffic_volume) AS min_volume,
    MAX(traffic_volume) AS max_volume,
    MAX(traffic_volume) - MIN(traffic_volume) AS range_volume,
    ROUND(
        SQRT(AVG(traffic_volume * 1.0 * traffic_volume) - AVG(traffic_volume * 1.0) * AVG(traffic_volume * 1.0)), 2
    ) AS population_stddev,
    ROUND(
        AVG(traffic_volume * 1.0 * traffic_volume) - AVG(traffic_volume * 1.0) * AVG(traffic_volume * 1.0), 2
    ) AS population_variance,
    ROUND(
        SQRT( (AVG(traffic_volume*1.0*traffic_volume) - AVG(traffic_volume*1.0)*AVG(traffic_volume*1.0))
              * (COUNT(*)*1.0/(COUNT(*)-1)) ), 2
    ) AS sample_stddev,
    ROUND(
        (AVG(traffic_volume*1.0*traffic_volume) - AVG(traffic_volume*1.0)*AVG(traffic_volume*1.0))
        * (COUNT(*)*1.0/(COUNT(*)-1)), 2
    ) AS sample_variance
FROM traffic_hourly;
