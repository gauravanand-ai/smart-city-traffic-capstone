#!/usr/bin/env python3
"""
serving_common.py
Part 3, Task 6.3 support module: turns ONE raw traffic/weather reading into
the same 24-column feature row Part 2's feature_engineering.py + Part 3's
ml_common.py build for a whole CSV -- so deploy_api.py can score a live
request with the exact same transformations the trained models were fit on,
not an approximation of them.

Why this exists as its own module rather than inline in deploy_api.py: the
transformation logic (cyclical time encodings, one-hot weather, z-score/
min-max scaling, holiday flag) is inference-time feature engineering, not a
route handler -- keeping it separate makes it independently testable and
mirrors how a real deployment usually separates its feature-transformation
code from its serving code.

Raw input mirrors the ORIGINAL dataset's own raw columns (holiday, temp,
rain_1h, snow_1h, clouds_all, weather_main, date_time) rather than asking
the API caller to already know about hour_sin/cos or which one-hot column to
set -- exactly what a real upstream system (a traffic sensor + a weather
feed) would actually have available, and exactly the same raw shape Part 2's
pipeline.py/feature_engineering.py consumed.

Scaling constants: temp_zscore's mean/std and clouds_all_minmax's min/max
are NOT recomputed here. Recomputing them from whatever data happens to be
on hand at serving time would silently drift the model's input scale away
from what it was trained on. Instead they're read once from
feature_scaling_params.json (written by compute_and_save_scaling_params(),
below) which snapshots the exact same statistics Part 2's
add_scaled_numeric_features() computed over the full training corpus
(traffic_features.csv) -- the standard MLOps practice of persisting
"training-time statistics" as their own versioned artifact rather than
silently recomputing them wherever the model happens to run next.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from ml_common import COMMON_FEATURE_COLUMNS

logger = logging.getLogger(__name__)

# Exactly Part 2's feature_engineering.py groupings (SEVERE_WEATHER,
# LOW_VISIBILITY_WEATHER) -- duplicated here rather than imported across the
# pipeline/ and ml/ directories, same rationale as ml_common.py's own
# logging setup duplication.
WEATHER_CATEGORIES = [
    "clear", "clouds", "drizzle", "fog", "haze", "mist",
    "rain", "smoke", "snow", "squall", "thunderstorm",
]
SEVERE_WEATHER = {"thunderstorm", "squall"}
LOW_VISIBILITY_WEATHER = {"fog", "mist", "haze", "smoke"}
WEEKEND_DAYS = {5, 6}


class FeatureEngineeringError(Exception):
    """Raised when a raw reading can't be turned into a valid feature row
    (bad weather category, unparseable date_time, etc.) -- always caught and
    turned into a clean HTTP 422, never a raw traceback."""


def compute_and_save_scaling_params(input_csv: Path, output_path: Path) -> dict:
    """Computes temp_zscore/clouds_all_minmax's scaling constants from the
    full feature-engineered dataset (the same dataset every model in this
    project was trained from) and persists them to JSON. Run once (or
    whenever the dataset is refreshed) -- not on every API request."""
    df = pd.read_csv(input_csv, usecols=["temp", "clouds_all"])
    params = {
        "temp_mean": float(df["temp"].mean()),
        "temp_std": float(df["temp"].std()),
        "clouds_min": float(df["clouds_all"].min()),
        "clouds_max": float(df["clouds_all"].max()),
        "source_csv": str(input_csv),
        "source_rows": int(len(df)),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(params, indent=2))
    logger.info("Wrote feature scaling parameters to %s: %s", output_path, params)
    return params


def load_scaling_params(path: Path) -> dict:
    if not path.exists():
        raise FeatureEngineeringError(
            f"Feature scaling parameters not found at {path}. Run "
            f"serving_common.compute_and_save_scaling_params() first (deploy_api.py "
            f"does this automatically on first startup if the file is missing)."
        )
    return json.loads(path.read_text())


def engineer_feature_row(raw: dict, scaling: dict) -> pd.DataFrame:
    """Turns one raw reading (see RawReading in deploy_api.py for the exact
    fields) into a single-row DataFrame with COMMON_FEATURE_COLUMNS, in the
    same column order every trained model expects."""
    weather_main = raw["weather_main"].strip().lower()
    if weather_main not in WEATHER_CATEGORIES:
        raise FeatureEngineeringError(
            f"Unknown weather_main '{raw['weather_main']}'. Valid options "
            f"(case-insensitive): {', '.join(WEATHER_CATEGORIES)}."
        )

    try:
        dt = pd.Timestamp(raw["date_time"])
    except (ValueError, TypeError) as e:
        raise FeatureEngineeringError(f"Could not parse date_time '{raw['date_time']}': {e}") from e

    hour = dt.hour
    day_of_week = dt.dayofweek
    is_weekend = int(day_of_week in WEEKEND_DAYS)
    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)
    dow_sin = np.sin(2 * np.pi * day_of_week / 7)
    dow_cos = np.cos(2 * np.pi * day_of_week / 7)

    temp = float(raw["temp"])
    clouds_all = float(raw["clouds_all"])
    rain_1h = float(raw.get("rain_1h", 0.0) or 0.0)
    snow_1h = float(raw.get("snow_1h", 0.0) or 0.0)
    is_holiday = bool(raw.get("is_holiday", False))

    temp_zscore = (temp - scaling["temp_mean"]) / scaling["temp_std"]
    clouds_range = scaling["clouds_max"] - scaling["clouds_min"]
    clouds_all_minmax = (clouds_all - scaling["clouds_min"]) / clouds_range if clouds_range else 0.0

    row = {
        "hour_sin": hour_sin, "hour_cos": hour_cos,
        "dow_sin": dow_sin, "dow_cos": dow_cos,
        "is_weekend": is_weekend, "month": dt.month, "year": dt.year,
        "is_precipitating": int(rain_1h > 0 or snow_1h > 0),
        "is_severe_weather": int(weather_main in SEVERE_WEATHER),
        "is_low_visibility": int(weather_main in LOW_VISIBILITY_WEATHER),
        "temp_zscore": temp_zscore, "clouds_all_minmax": clouds_all_minmax,
        "holiday_flag": int(is_holiday),
    }
    for category in WEATHER_CATEGORIES:
        row[f"weather_{category}"] = int(category == weather_main)

    X = pd.DataFrame([row])[COMMON_FEATURE_COLUMNS]
    return X
