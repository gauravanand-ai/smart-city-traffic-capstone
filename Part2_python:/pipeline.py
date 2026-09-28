#!/usr/bin/env python3
"""
pipeline.py
Reproducible data pipeline for the Metro Interstate Traffic Volume dataset.

Stages: load -> validate schema -> clean (standardize / parse / dedupe / outliers)
-> write cleaned CSV. Every stage logs what it did and why, so the processing
history can be reconstructed from the log alone without re-running the code.

Usage:
    python pipeline.py [--input PATH] [--output PATH] [--log-file PATH] [-v]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------
# Module-level logger via getLogger(__name__) -- never the root logger directly
# (i.e. never logging.info(...) / logging.warning(...) at module scope).
logger = logging.getLogger(__name__)


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


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EXPECTED_COLUMNS = [
    "holiday", "temp", "rain_1h", "snow_1h", "clouds_all",
    "weather_main", "weather_description", "date_time", "traffic_volume",
]

MIN_PLAUSIBLE_TEMP_K = 1.0        # 0 K (absolute zero) is a sensor fault, not weather
MAX_PLAUSIBLE_RAIN_MM = 9000.0    # world-record hourly rainfall is well under this


class SchemaValidationError(Exception):
    """Raised when the input CSV does not have the columns the pipeline expects."""


# ---------------------------------------------------------------------------
# Stage 1: Load
# ---------------------------------------------------------------------------
def load_raw_csv(path: Path) -> pd.DataFrame:
    """Load the raw CSV. Raises on failure after logging the error."""
    try:
        # keep_default_na/na_values: the literal text "None" in the `holiday`
        # column means "not a holiday" (a valid category), not a missing value --
        # pandas' default NA-string list would otherwise silently swallow it.
        df = pd.read_csv(path, keep_default_na=False, na_values=[""])
    except FileNotFoundError:
        logger.error("Raw CSV file not found at %s", path, exc_info=True)
        raise
    except pd.errors.EmptyDataError:
        logger.error("Raw CSV file at %s is empty", path, exc_info=True)
        raise
    except pd.errors.ParserError:
        logger.error("Raw CSV file at %s could not be parsed (malformed CSV)", path, exc_info=True)
        raise
    except (OSError, UnicodeDecodeError):
        logger.error("I/O error while reading %s", path, exc_info=True)
        raise

    logger.info(
        "Loaded raw CSV successfully: %d rows, %d columns (source: %s)",
        df.shape[0], df.shape[1], path.name,
    )
    return df


# ---------------------------------------------------------------------------
# Stage 2: Validate schema (must happen before any other processing)
# ---------------------------------------------------------------------------
def validate_schema(df: pd.DataFrame) -> None:
    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        logger.error("Schema validation failed: missing expected column(s) %s", missing)
        raise SchemaValidationError(f"Missing expected columns: {missing}")

    unexpected = [c for c in df.columns if c not in EXPECTED_COLUMNS]
    if unexpected:
        logger.warning(
            "Schema validation: %d unexpected column(s) present and will be ignored: %s",
            len(unexpected), unexpected,
        )

    logger.info(
        "Schema validation passed: all %d expected columns are present",
        len(EXPECTED_COLUMNS),
    )


# ---------------------------------------------------------------------------
# Stage 3: Clean -- each sub-step logs its own outcome separately
# ---------------------------------------------------------------------------
def standardize_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Fix inconsistent casing/whitespace in categorical text columns.

    Found in this dataset: weather_description mixes case for the same real
    condition (e.g. "Sky is Clear" vs "sky is clear", "SQUALLS" vs the rest in
    lowercase), which would otherwise be counted as separate categories.
    """
    df = df.copy()
    changed_total = 0

    for col in ("weather_main", "weather_description", "holiday"):
        original = df[col].astype(str)
        canonical = original.str.strip().str.lower()

        n_categories_before = original.nunique()
        n_categories_after = canonical.nunique()

        if n_categories_after < n_categories_before:
            # Real inconsistency: two or more original spellings collapse to the
            # same canonical value (e.g. "Sky is Clear" / "sky is clear",
            # "SQUALLS" / "squalls"). Count rows that used a minority spelling
            # within their canonical group -- not every row, since most were
            # already consistent.
            modal_spelling = original.groupby(canonical).transform(lambda s: s.value_counts().idxmax())
            n_changed = int((original != modal_spelling).sum())
            logger.warning(
                "Standardized casing/whitespace in '%s': %d category name(s) collapsed "
                "into %d, affecting %d row(s) that used a non-dominant spelling",
                col, n_categories_before, n_categories_after, n_changed,
            )
            changed_total += n_changed
        else:
            logger.info("Categorical check on '%s': %d categories, no case/whitespace inconsistency found",
                        col, n_categories_before)

        df[col] = canonical

    if changed_total == 0:
        logger.info("Categorical standardization: no inconsistent values found across checked columns")

    return df


def parse_and_validate_datetime(df: pd.DataFrame) -> pd.DataFrame:
    """Parse date_time to a real datetime dtype; drop rows that don't parse."""
    df = df.copy()
    before = len(df)
    df["date_time"] = pd.to_datetime(df["date_time"], errors="coerce")

    unparseable = df["date_time"].isna()
    n_bad = int(unparseable.sum())
    if n_bad:
        df = df.loc[~unparseable].copy()
        logger.warning(
            "Dropped %d row(s) with an unparseable date_time value "
            "(could not be interpreted as a valid date/time)",
            n_bad,
        )
    else:
        logger.info("date_time parsing: all %d rows parsed to a valid timestamp", before)

    return df


def remove_duplicate_rows(df: pd.DataFrame) -> pd.DataFrame:
    """The raw export has 2-6 rows per hour (repeated weather observations
    logged against the same timestamp). Keep one row per hour."""
    df = df.sort_values("date_time")
    before = len(df)
    df = df.drop_duplicates(subset="date_time", keep="first")
    removed = before - len(df)

    if removed:
        logger.warning(
            "Removed %d duplicate row(s) sharing a date_time with an earlier row "
            "(same hour logged multiple times with different weather observations)",
            removed,
        )
    else:
        logger.info("Duplicate check: no duplicate date_time rows found")

    return df.reset_index(drop=True)


def _impute_by_month(df: pd.DataFrame, column: str, outlier_mask: pd.Series, reason: str) -> pd.DataFrame:
    """Impute flagged outlier values in `column` using that row's own
    calendar-month median (computed from the *non-outlier* rows of the same
    month), rather than one global average for the whole dataset.

    Uses an explicit loop over the affected months, since each month needs
    its own median calculated and applied.
    """
    n_outliers = int(outlier_mask.sum())
    if n_outliers == 0:
        logger.info("Outlier check on '%s': no values outside the plausible range", column)
        return df

    df = df.copy()
    months = df.loc[outlier_mask, "date_time"].dt.month
    affected_months = sorted(months.unique())

    for month in affected_months:
        month_rows = df["date_time"].dt.month == month
        clean_rows_this_month = month_rows & (~outlier_mask)
        if clean_rows_this_month.any():
            month_median = df.loc[clean_rows_this_month, column].median()
        else:
            # No clean reference values for this month anywhere in the dataset;
            # fall back to the column's overall clean median as a last resort.
            month_median = df.loc[~outlier_mask, column].median()

        target_rows = month_rows & outlier_mask
        n_this_month = int(target_rows.sum())
        df.loc[target_rows, column] = month_median
        logger.debug(
            "Month %02d: imputed %d '%s' outlier(s) with month median %.2f",
            month, n_this_month, column, month_median,
        )

    logger.warning(
        "Imputed %d row(s) in '%s' using month-specific medians (reason: %s)",
        n_outliers, column, reason,
    )
    return df


def handle_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Detect and impute physically impossible sensor readings."""
    df = df.copy()

    temp_outlier_mask = df["temp"] < MIN_PLAUSIBLE_TEMP_K
    df = _impute_by_month(
        df, "temp", temp_outlier_mask,
        reason=f"temperature below {MIN_PLAUSIBLE_TEMP_K}K (0 K / absolute zero is a sensor fault, not real weather)",
    )

    rain_outlier_mask = df["rain_1h"] > MAX_PLAUSIBLE_RAIN_MM
    df = _impute_by_month(
        df, "rain_1h", rain_outlier_mask,
        reason=f"hourly rainfall above {MAX_PLAUSIBLE_RAIN_MM:.0f}mm (physically implausible in a single hour)",
    )

    # Defensive range checks -- log if anything else looks wrong, without
    # inventing an imputation rule the task didn't ask for.
    negative_values = (df["rain_1h"] < 0).sum() + (df["snow_1h"] < 0).sum()
    out_of_range_clouds = ((df["clouds_all"] < 0) | (df["clouds_all"] > 100)).sum()
    if negative_values or out_of_range_clouds:
        logger.warning(
            "Additional suspect values found (not auto-corrected): "
            "%d negative rain/snow reading(s), %d clouds_all value(s) outside 0-100%%",
            int(negative_values), int(out_of_range_clouds),
        )
    else:
        logger.info("Range check on rain_1h/snow_1h/clouds_all: no additional suspect values found")

    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = standardize_categoricals(df)
    df = parse_and_validate_datetime(df)
    df = remove_duplicate_rows(df)
    df = handle_outliers(df)
    return df


# ---------------------------------------------------------------------------
# Stage 4: Save
# ---------------------------------------------------------------------------
def save_cleaned_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False)
    logger.info("Wrote cleaned dataset to %s: %d rows, %d columns", path, df.shape[0], df.shape[1])


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def run_pipeline(input_path: Path, output_path: Path) -> pd.DataFrame:
    df = load_raw_csv(input_path)
    validate_schema(df)
    df = clean_data(df)
    save_cleaned_csv(df, output_path)
    return df


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Clean the Metro Interstate Traffic Volume dataset.")
    parser.add_argument("--input", type=Path, default=here / "Metro_Interstate_Traffic_Volume.csv",
                         help="Path to the raw CSV file.")
    parser.add_argument("--output", type=Path, default=here / "traffic_cleaned_pipeline.csv",
                         help="Path to write the cleaned CSV file.")
    parser.add_argument("--log-file", type=Path, default=here / "pipeline.log",
                         help="Path to write the log file.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable DEBUG-level logging.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging(args.log_file, verbose=args.verbose)

    try:
        run_pipeline(args.input, args.output)
    except (FileNotFoundError, pd.errors.EmptyDataError, pd.errors.ParserError,
            OSError, UnicodeDecodeError, SchemaValidationError):
        # Already logged with full detail at the point of failure; log once more
        # at the top level so it's unmistakable the pipeline did not complete.
        logger.error("Pipeline aborted: could not produce a cleaned dataset", exc_info=True)
        return 1
    except Exception:
        # Anything unanticipated still exits gracefully rather than crashing
        # with an unhandled traceback -- never a bare `except:`.
        logger.error("Pipeline aborted due to an unexpected error", exc_info=True)
        return 1

    logger.info("Pipeline completed successfully")
    return 0


if __name__ == "__main__":
    sys.exit(main())
