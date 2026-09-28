#!/usr/bin/env python3
"""
ml_common.py
Shared code for Part 3's supervised learning scripts (train_classification.py,
train_regression.py): logging setup, the proxy accident-risk label, the
holiday flag, and the single common feature set both models train on.

Not meant to be run directly -- imported by the two train_*.py scripts.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------
# Duplicated from Part 2's pipeline.py (rather than imported across the
# pipeline/ and ml/ directories) so this folder can be run and delivered on
# its own without a cross-directory import path. Same handler setup, same
# formatter, same INFO/WARNING/ERROR/DEBUG discipline as every other script
# in this project -- see the project README for the full logging writeup.
def configure_logging(log_file: Path, verbose: bool = False) -> None:
    """Configure handlers/formatting once, at program start. Individual
    modules/functions still log through logging.getLogger(__name__)."""
    level = logging.DEBUG if verbose else logging.INFO
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()
    root.addHandler(console_handler)
    root.addHandler(file_handler)

    # matplotlib's own DEBUG output (font matching, backend selection, etc.)
    # is noisy and irrelevant to this project -- keep it at WARNING even in
    # -v mode so -v surfaces this project's own DEBUG messages, not a
    # third-party library's internals.
    logging.getLogger("matplotlib").setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# Proxy accident-risk label (exactly as specified in the Part 3 brief)
# ---------------------------------------------------------------------------
# No accident dataset was provided for this capstone. high_risk is a PROXY
# label built from data already in traffic_features.csv, intended only to
# demonstrate the classification workflow -- not a real accident prediction.
SEVERE_WEATHER = {"thunderstorm", "squall"}

# Columns this module needs to already be present in the input CSV -- i.e.
# Part 2's feature_engineering.py output, not its raw or merely-cleaned
# intermediates. Checked immediately after loading so pointing these scripts
# at the wrong file produces one clear logged ERROR instead of a raw
# KeyError traceback part-way through label construction or feature
# selection.
REQUIRED_INPUT_COLUMNS = [
    "date_time", "holiday", "traffic_volume", "weather_main",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_weekend",
    "weather_clear", "weather_clouds", "weather_drizzle", "weather_fog",
    "weather_haze", "weather_mist", "weather_rain", "weather_smoke",
    "weather_snow", "weather_squall", "weather_thunderstorm",
    "is_precipitating", "is_severe_weather", "is_low_visibility",
    "temp_zscore", "clouds_all_minmax",
]


class UserInputError(Exception):
    """Raised for invalid user input (bad path, empty dataset, etc.) --
    always caught and turned into a clean logged ERROR, never a raw traceback."""


def load_dataset(path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        logger.error(
            "Dataset not found at %s (run pipeline.py + feature_engineering.py "
            "from Part 2 first)", path, exc_info=True,
        )
        raise
    except pd.errors.EmptyDataError:
        logger.error("Dataset at %s is empty", path, exc_info=True)
        raise
    except pd.errors.ParserError:
        logger.error("Dataset at %s could not be parsed", path, exc_info=True)
        raise
    except (OSError, UnicodeDecodeError):
        logger.error("I/O error while reading %s", path, exc_info=True)
        raise

    if df.empty:
        raise UserInputError(f"Dataset at {path} has no rows.")

    missing = [c for c in REQUIRED_INPUT_COLUMNS if c not in df.columns]
    if missing:
        raise UserInputError(
            f"{path} is missing expected column(s): {missing}. This script needs "
            f"Part 2's feature-engineered output -- run pipeline.py then "
            f"feature_engineering.py first, and point --input at the resulting "
            f"traffic_features.csv."
        )

    df["date_time"] = pd.to_datetime(df["date_time"])
    logger.info("Loaded dataset: %d rows, %d columns (source: %s)", *df.shape, path.name)
    return df


def add_holiday_flag(df: pd.DataFrame) -> pd.DataFrame:
    """holiday_flag = 1 on a recognized US holiday, 0 otherwise. The raw
    'holiday' column is ~99.9% the literal string 'none' (standardized to
    lowercase by Part 2's pipeline.py) with a handful of specific holiday
    names on the remaining rows -- collapse that into one binary flag."""
    df = df.copy()
    df["holiday_flag"] = (df["holiday"].str.lower() != "none").astype(int)
    n_holiday = int(df["holiday_flag"].sum())
    logger.info(
        "Added holiday_flag: %d of %d row(s) (%.2f%%) fall on a recognized holiday",
        n_holiday, len(df), 100 * n_holiday / len(df),
    )
    return df


def add_time_extras(df: pd.DataFrame) -> pd.DataFrame:
    """month/year as plain integer features -- Part 2 already engineered
    hour/day-of-week (including their cyclical hour_sin/hour_cos and
    dow_sin/dow_cos encodings), but not month or year, which give the models
    seasonal and multi-year-trend signal that hour/day-of-week alone can't."""
    df = df.copy()
    df["month"] = df["date_time"].dt.month
    df["year"] = df["date_time"].dt.year
    logger.info("Added time features: month, year")
    return df


def add_proxy_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Adds this task's own congestion_category (Low/Medium/High/Severe,
    quartile-based) and the proxy high_risk label, using the exact logic
    given in the Part 3 brief.

    Note: this congestion_category is Part 3's own quartile bucketing and is
    intentionally kept separate from Part 2's congestion_category
    (Low/Medium/High/Very High, built with pd.qcut) -- same quartile idea,
    different label set and boundary convention (<=), computed here to match
    the brief's reference implementation exactly rather than reuse Part 2's.
    """
    df = df.copy()

    q1, q2, q3 = df["traffic_volume"].quantile([0.25, 0.5, 0.75]).values
    logger.debug("Part 3 congestion_category quartile thresholds: q1=%.1f, q2=%.1f, q3=%.1f", q1, q2, q3)

    def bucket(v: float) -> str:
        if v <= q1:
            return "Low"
        elif v <= q2:
            return "Medium"
        elif v <= q3:
            return "High"
        return "Severe"

    df["congestion_category_p3"] = df["traffic_volume"].apply(bucket)

    high_congestion = df["congestion_category_p3"].isin(["High", "Severe"])
    risky_weather = df["weather_main"].isin(SEVERE_WEATHER) | (df["is_low_visibility"] == 1)
    df["high_risk"] = (high_congestion & risky_weather).astype(int)

    n_positive = int(df["high_risk"].sum())
    positive_rate = n_positive / len(df)
    logger.info(
        "Created proxy 'high_risk' label: %d of %d row(s) (%.2f%%) flagged high risk "
        "(high/severe congestion AND severe/low-visibility weather)",
        n_positive, len(df), 100 * positive_rate,
    )
    if positive_rate < 0.10:
        logger.warning(
            "high_risk classes are imbalanced (positive rate %.2f%%) -- using "
            "class_weight='balanced' and reporting F1/ROC AUC alongside accuracy",
            100 * positive_rate,
        )
    return df


def prepare_ml_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Runs the full Part 3 preprocessing: holiday flag, month/year, and the
    proxy congestion/high_risk labels, in that order."""
    df = add_holiday_flag(df)
    df = add_time_extras(df)
    df = add_proxy_labels(df)
    return df


# ---------------------------------------------------------------------------
# Common feature set -- shared, unchanged, between the classifier and the
# regressor. Deliberately EXCLUDES traffic_volume, both congestion_category
# columns, and high_risk itself: each is either the regression target or a
# column built directly from it (or, for high_risk, directly IS the
# classification target), so including any of them as a feature would leak
# the answer rather than let the model learn from context. See README.
# ---------------------------------------------------------------------------
TIME_FEATURES = [
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",  # cyclical (Part 2)
    "is_weekend", "month", "year",
]

WEATHER_FEATURES = [
    "weather_clear", "weather_clouds", "weather_drizzle", "weather_fog",
    "weather_haze", "weather_mist", "weather_rain", "weather_smoke",
    "weather_snow", "weather_squall", "weather_thunderstorm",  # one-hot (Part 2)
    "is_precipitating", "is_severe_weather", "is_low_visibility",  # derived (Part 2)
    "temp_zscore", "clouds_all_minmax",  # scaled numerics (Part 2)
]

HOLIDAY_FEATURES = ["holiday_flag"]

COMMON_FEATURE_COLUMNS = TIME_FEATURES + WEATHER_FEATURES + HOLIDAY_FEATURES

LEAKAGE_COLUMNS = [
    "traffic_volume", "congestion_category", "congestion_category_p3", "high_risk",
]


def get_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in COMMON_FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise UserInputError(
            f"Dataset is missing expected feature column(s): {missing}. "
            f"Did you run pipeline.py and feature_engineering.py (Part 2) first?"
        )
    X = df[COMMON_FEATURE_COLUMNS].copy()
    logger.info(
        "Built common feature matrix: %d rows x %d columns (%s)",
        *X.shape, ", ".join(COMMON_FEATURE_COLUMNS),
    )
    return X


def chronological_split(df: pd.DataFrame, test_size: float = 0.2):
    """Splits by time (earliest rows -> train, most recent rows -> test)
    rather than a random shuffle. Consecutive hourly readings are highly
    autocorrelated, so a random split would let near-duplicate neighboring
    hours leak between train and test and inflate every metric; a
    chronological split is the standard, honest choice for time-series data
    and also mirrors how the model would actually be used (trained on the
    past, evaluated on more recent data it hasn't seen)."""
    df_sorted = df.sort_values("date_time").reset_index(drop=True)
    split_idx = int(len(df_sorted) * (1 - test_size))
    train_df = df_sorted.iloc[:split_idx]
    test_df = df_sorted.iloc[split_idx:]
    logger.info(
        "Chronological train/test split: train=%d row(s) (%s to %s), "
        "test=%d row(s) (%s to %s)",
        len(train_df), train_df["date_time"].min(), train_df["date_time"].max(),
        len(test_df), test_df["date_time"].min(), test_df["date_time"].max(),
    )
    return train_df, test_df
