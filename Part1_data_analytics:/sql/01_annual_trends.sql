-- Task 1.2: Annual traffic trends 2012-2017 (deduplicated hourly table)
SELECT
    strftime('%Y', date_time)                                   AS year,
    COUNT(*)                                                     AS hours_recorded,
    SUM(traffic_volume)                                          AS total_volume,
    ROUND(AVG(traffic_volume), 1)                                AS avg_hourly_volume,
    LAG(SUM(traffic_volume)) OVER (ORDER BY strftime('%Y', date_time))      AS prev_year_total,
    SUM(traffic_volume) - LAG(SUM(traffic_volume)) OVER (ORDER BY strftime('%Y', date_time)) AS yoy_change,
    ROUND(
        100.0 * (SUM(traffic_volume) - LAG(SUM(traffic_volume)) OVER (ORDER BY strftime('%Y', date_time)))
        / LAG(SUM(traffic_volume)) OVER (ORDER BY strftime('%Y', date_time)), 2
    )                                                             AS yoy_pct_change
FROM traffic_hourly
WHERE strftime('%Y', date_time) BETWEEN '2012' AND '2017'
GROUP BY year
ORDER BY year;
