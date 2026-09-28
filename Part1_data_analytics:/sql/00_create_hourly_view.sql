-- Data quality fix: the raw `traffic` table has one row per (hour, weather observation),
-- so many hours appear 2-6 times with identical date_time/traffic_volume/temp but different
-- weather_description (multiple simultaneous weather readings logged by the source API).
-- Verified: traffic_volume and temp are IDENTICAL within every duplicated timestamp group,
-- so deduplicating on date_time (keeping one row per hour) does not lose information for
-- volume/temperature analysis and prevents inflated SUM()/COUNT() results.
DROP TABLE IF EXISTS traffic_hourly;

CREATE TABLE traffic_hourly AS
SELECT holiday, temp, rain_1h, snow_1h, clouds_all, weather_main, weather_description,
       date_time, traffic_volume
FROM traffic
WHERE rowid IN (
    SELECT MIN(rowid) FROM traffic GROUP BY date_time
);

CREATE INDEX idx_traffic_hourly_datetime ON traffic_hourly(date_time);
