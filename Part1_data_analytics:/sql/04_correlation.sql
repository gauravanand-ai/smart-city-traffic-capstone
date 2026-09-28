-- Task 2.2: Pearson correlation coefficient between temp and traffic_volume.
-- SQLite has no built-in CORR(); computed from the standard formula using aggregate sums.
WITH stats AS (
    SELECT
        COUNT(*) AS n,
        SUM(temp) AS sum_x,
        SUM(traffic_volume) AS sum_y,
        SUM(temp * traffic_volume) AS sum_xy,
        SUM(temp * temp) AS sum_x2,
        SUM(traffic_volume * 1.0 * traffic_volume) AS sum_y2
    FROM traffic_hourly
)
SELECT
    n,
    ROUND(
        (n * sum_xy - sum_x * sum_y) /
        (SQRT(n * sum_x2 - sum_x * sum_x) * SQRT(n * sum_y2 - sum_y * sum_y))
    , 4) AS pearson_r
FROM stats;
