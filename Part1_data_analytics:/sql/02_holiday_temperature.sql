-- Task 1.3: Temperature patterns on New Year's Day and Labor Day, 2015-2017.
-- Methodology note: the `holiday` column in the source data is only populated on a single
-- hour of the holiday (usually 00:00, and for New Year's 2017 it is mis-tagged on Jan 2
-- instead of Jan 1) -- relying on it alone would compare a single reading per year and would
-- silently drop 2015 New Year's Day (no January 2015 data exists in the source at all -- see
-- the coverage-gap finding from Task 1.2). Instead we match on the calendar DATE, which is
-- more complete and correct: New Year's Day = Jan 1 every year; Labor Day = the first Monday
-- of September, whose actual date we take from the dataset's own 'Labor Day' tag each year.

SELECT
    'New Years Day' AS holiday_name,
    strftime('%Y', date_time) AS year,
    date(date_time) AS calendar_date,
    COUNT(*) AS hours_available,
    ROUND(AVG(temp), 2) AS avg_temp_k,
    ROUND(AVG(temp) - 273.15, 2) AS avg_temp_c,
    ROUND(MIN(temp), 2) AS min_temp_k,
    ROUND(MAX(temp), 2) AS max_temp_k,
    ROUND(AVG(traffic_volume), 1) AS avg_traffic_volume
FROM traffic_hourly
WHERE strftime('%m-%d', date_time) = '01-01'
  AND strftime('%Y', date_time) IN ('2015', '2016', '2017')
GROUP BY year

UNION ALL

SELECT
    'Labor Day' AS holiday_name,
    strftime('%Y', date_time) AS year,
    date(date_time) AS calendar_date,
    COUNT(*) AS hours_available,
    ROUND(AVG(temp), 2) AS avg_temp_k,
    ROUND(AVG(temp) - 273.15, 2) AS avg_temp_c,
    ROUND(MIN(temp), 2) AS min_temp_k,
    ROUND(MAX(temp), 2) AS max_temp_k,
    ROUND(AVG(traffic_volume), 1) AS avg_traffic_volume
FROM traffic_hourly
WHERE date(date_time) IN (
    SELECT date(date_time) FROM traffic_hourly WHERE holiday = 'Labor Day'
      AND strftime('%Y', date_time) IN ('2015', '2016', '2017')
)
GROUP BY year
ORDER BY holiday_name, year;
