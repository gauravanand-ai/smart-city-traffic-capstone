-- Task 3: Probability and congestion analysis
-- Congestion := traffic_volume > 5500. Clear Weather := weather_main = 'Clear'.
-- High Temperature := temp > 292K.

-- 3.1 Basic probabilities
SELECT
    ROUND(1.0 * SUM(CASE WHEN traffic_volume > 5500 THEN 1 ELSE 0 END) / COUNT(*), 4) AS p_congestion,
    ROUND(1.0 * SUM(CASE WHEN weather_main = 'Clear' THEN 1 ELSE 0 END) / COUNT(*), 4) AS p_clear_weather,
    ROUND(1.0 * SUM(CASE WHEN traffic_volume > 5500 AND weather_main = 'Clear' THEN 1 ELSE 0 END) / COUNT(*), 4) AS p_congestion_and_clear
FROM traffic_hourly;

-- 3.2 Conditional probabilities
-- P(Clear Weather | Congestion) = P(Clear AND Congestion) / P(Congestion)
SELECT
    ROUND(
        1.0 * SUM(CASE WHEN traffic_volume > 5500 AND weather_main = 'Clear' THEN 1 ELSE 0 END)
        / SUM(CASE WHEN traffic_volume > 5500 THEN 1 ELSE 0 END)
    , 4) AS p_clear_given_congestion,
    ROUND(
        1.0 * SUM(CASE WHEN traffic_volume > 5500 AND temp > 292 THEN 1 ELSE 0 END)
        / SUM(CASE WHEN traffic_volume > 5500 THEN 1 ELSE 0 END)
    , 4) AS p_hightemp_given_congestion
FROM traffic_hourly;

-- Independence check: compare P(Congestion AND Clear) to P(Congestion) * P(Clear)
SELECT
    ROUND(1.0 * SUM(CASE WHEN traffic_volume > 5500 AND weather_main = 'Clear' THEN 1 ELSE 0 END) / COUNT(*), 4) AS p_a_and_b_observed,
    ROUND(
        (1.0 * SUM(CASE WHEN traffic_volume > 5500 THEN 1 ELSE 0 END) / COUNT(*)) *
        (1.0 * SUM(CASE WHEN weather_main = 'Clear' THEN 1 ELSE 0 END) / COUNT(*))
    , 4) AS p_a_times_p_b_if_independent
FROM traffic_hourly;

-- Odds ratio of congestion: Clear weather vs Cloudy weather
-- Odds(Congestion | Clear) = P(Cong|Clear) / (1 - P(Cong|Clear))
-- Odds(Congestion | Clouds) = P(Cong|Clouds) / (1 - P(Cong|Clouds))
-- Odds Ratio = Odds(Clear) / Odds(Clouds)
WITH clear AS (
    SELECT
        1.0 * SUM(CASE WHEN traffic_volume > 5500 THEN 1 ELSE 0 END) / COUNT(*) AS p_cong
    FROM traffic_hourly WHERE weather_main = 'Clear'
),
cloudy AS (
    SELECT
        1.0 * SUM(CASE WHEN traffic_volume > 5500 THEN 1 ELSE 0 END) / COUNT(*) AS p_cong
    FROM traffic_hourly WHERE weather_main = 'Clouds'
)
SELECT
    ROUND(clear.p_cong, 4) AS p_congestion_given_clear,
    ROUND(cloudy.p_cong, 4) AS p_congestion_given_cloudy,
    ROUND(clear.p_cong / (1 - clear.p_cong), 4) AS odds_congestion_clear,
    ROUND(cloudy.p_cong / (1 - cloudy.p_cong), 4) AS odds_congestion_cloudy,
    ROUND( (clear.p_cong / (1 - clear.p_cong)) / (cloudy.p_cong / (1 - cloudy.p_cong)), 4) AS odds_ratio_clear_vs_cloudy
FROM clear, cloudy;
