#!/usr/bin/env python3
"""
feature_engineering.py
Task 2: build ML-ready features from Task 1's cleaned dataset, using NumPy/Pandas.

Builds directly on pipeline.py's output (traffic_cleaned_pipeline.csv) rather than
re-implementing cleaning here -- reuses its logging setup too.

Usage:
    python feature_engineering.py [--input PATH] [--output PATH] [--log-file PATH] [-v]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline import configure_logging  # reuse Task 1's logging setup

logger = logging.getLogger(__name__)

WEEKEND_DAYS = {5, 6}  # Saturday, Sunday (Monday=0 .. Sunday=6)
SEVERE_WEATHER = {"thunderstorm", "squall"}
LOW_VISIBILITY_WEATHER = {"fog", "mist", "haze", "smoke"}


# ---------------------------------------------------------------------------
# Time features
# ---------------------------------------------------------------------------
def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Hour, day of week, weekend indicator, and a cyclical (sin/cos) encoding
    of hour-of-day so midnight (23 -> 0) doesn't look like the largest possible
    jump to a model that only sees the raw integer."""
    df = df.copy()
    df["date_time"] = pd.to_datetime(df["date_time"])

    df["hour"] = df["date_time"].dt.hour
    df["day_of_week"] = df["date_time"].dt.dayofweek  # 0=Monday .. 6=Sunday
    df["day_name"] = df["date_time"].dt.day_name()
    df["is_weekend"] = df["day_of_week"].isin(WEEKEND_DAYS).astype(int)

    # Cyclical encoding: hour-of-day (period 24) and day-of-week (period 7)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

    logger.info(
        "Added time features: hour, day_of_week, day_name, is_weekend, "
        "hour_sin/hour_cos (cyclical, period 24), dow_sin/dow_cos (cyclical, period 7)"
    )
    return df


# ---------------------------------------------------------------------------
# Weather features
# ---------------------------------------------------------------------------
def add_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode weather_main and add a small set of derived boolean
    indicators that are more directly useful to a model than the raw category
    name (e.g. 'is it precipitating at all', 'is visibility degraded')."""
    df = df.copy()

    dummies = pd.get_dummies(df["weather_main"], prefix="weather").astype(int)
    df = pd.concat([df, dummies], axis=1)

    df["is_precipitating"] = ((df["rain_1h"] > 0) | (df["snow_1h"] > 0)).astype(int)
    df["is_severe_weather"] = df["weather_main"].isin(SEVERE_WEATHER).astype(int)
    df["is_low_visibility"] = df["weather_main"].isin(LOW_VISIBILITY_WEATHER).astype(int)

    logger.info(
        "Added weather features: %d one-hot 'weather_*' column(s) from weather_main, "
        "plus derived indicators is_precipitating, is_severe_weather, is_low_visibility",
        dummies.shape[1],
    )
    return df


# ---------------------------------------------------------------------------
# Numerical features (scaling)
# ---------------------------------------------------------------------------
def add_scaled_numeric_features(df: pd.DataFrame) -> pd.DataFrame:
    """Two continuous variables, scaled two different ways:
    - temp -> z-score standardized (mean 0, std 1): the natural choice when a
      variable (temperature) is roughly symmetric and models care about how
      many standard deviations from average a reading is.
    - clouds_all -> min-max scaled to [0, 1]: it's already a 0-100 percentage,
      so min-max preserves that intuitive bounded-range meaning.
    The raw mean/std/min/max are intermediate calculations, not part of the
    final feature output, so they're logged at DEBUG only.
    """
    df = df.copy()

    temp_mean = df["temp"].mean()
    temp_std = df["temp"].std()
    logger.debug("temp z-score parameters: mean=%.4f, std=%.4f", temp_mean, temp_std)
    df["temp_zscore"] = (df["temp"] - temp_mean) / temp_std

    clouds_min = df["clouds_all"].min()
    clouds_max = df["clouds_all"].max()
    logger.debug("clouds_all min-max parameters: min=%.4f, max=%.4f", clouds_min, clouds_max)
    df["clouds_all_minmax"] = (df["clouds_all"] - clouds_min) / (clouds_max - clouds_min)

    logger.info(
        "Added scaled numeric features: temp_zscore (z-score standardized), "
        "clouds_all_minmax (min-max scaled to [0, 1])"
    )
    return df


# ---------------------------------------------------------------------------
# Traffic target: data-driven congestion category
# ---------------------------------------------------------------------------
def add_congestion_target(df: pd.DataFrame) -> pd.DataFrame:
    """Congestion category, derived from the data itself rather than a fixed
    cutoff: traffic_volume is split into quartiles (4 equal-sized bins, ~25%
    of observed hours each), so 'Low' vs 'High' reflects this corridor's own
    observed distribution rather than an assumed volume threshold that may
    not transfer to a different road or dataset.

    Logic: Q1 = 25th percentile, Q2 = median, Q3 = 75th percentile of
    traffic_volume. Bin boundaries: [min, Q1] = Low, (Q1, Q2] = Medium,
    (Q2, Q3] = High, (Q3, max] = Very High.
    """
    df = df.copy()
    q1, q2, q3 = df["traffic_volume"].quantile([0.25, 0.5, 0.75])
    logger.debug(
        "traffic_volume quartile thresholds used for congestion_category: "
        "Q1=%.1f, Q2(median)=%.1f, Q3=%.1f", q1, q2, q3,
    )

    labels = ["Low", "Medium", "High", "Very High"]
    df["congestion_category"] = pd.qcut(df["traffic_volume"], q=4, labels=labels)

    counts = df["congestion_category"].value_counts().reindex(labels).to_dict()
    logger.info(
        "Created data-driven 'congestion_category' target (quartile-based, 4 bins) "
        "from traffic_volume: %s", counts,
    )
    return df


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Feature engineering input shape: %d rows, %d columns", *df.shape)

    df = add_time_features(df)
    df = add_weather_features(df)
    df = add_scaled_numeric_features(df)
    df = add_congestion_target(df)

    logger.info("Feature engineering output shape: %d rows, %d columns", *df.shape)
    return df


def load_cleaned_csv(path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        logger.error(
            "Cleaned input CSV not found at %s (run pipeline.py first)", path, exc_info=True
        )
        raise
    except pd.errors.EmptyDataError:
        logger.error("Cleaned input CSV at %s is empty", path, exc_info=True)
        raise
    except pd.errors.ParserError:
        logger.error("Cleaned input CSV at %s could not be parsed", path, exc_info=True)
        raise
    except (OSError, UnicodeDecodeError):
        logger.error("I/O error while reading %s", path, exc_info=True)
        raise

    logger.info("Loaded cleaned dataset: %d rows, %d columns (source: %s)", *df.shape, path.name)
    return df


def save_features_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False)
    logger.info("Wrote feature-engineered dataset to %s: %d rows, %d columns", path, *df.shape)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Engineer ML-ready features for the traffic dataset.")
    parser.add_argument("--input", type=Path, default=here / "traffic_cleaned_pipeline.csv",
                         help="Path to Task 1's cleaned CSV.")
    parser.add_argument("--output", type=Path, default=here / "traffic_features.csv",
                         help="Path to write the feature-engineered CSV.")
    parser.add_argument("--log-file", type=Path, default=here / "feature_engineering.log",
                         help="Path to write the log file.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable DEBUG-level logging.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging(args.log_file, verbose=args.verbose)

    try:
        df = load_cleaned_csv(args.input)
        df = engineer_features(df)
        save_features_csv(df, args.output)
    except (FileNotFoundError, pd.errors.EmptyDataError, pd.errors.ParserError,
            OSError, UnicodeDecodeError):
        logger.error("Feature engineering aborted: could not produce an output dataset", exc_info=True)
        return 1
    except Exception:
        logger.error("Feature engineering aborted due to an unexpected error", exc_info=True)
        return 1

    logger.info("Feature engineering completed successfully")
    return 0


if __name__ == "__main__":
    sys.exit(main())
