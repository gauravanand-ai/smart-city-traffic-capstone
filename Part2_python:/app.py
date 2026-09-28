#!/usr/bin/env python3
"""
app.py
Task 4: Mini Traffic Analytics Application -- a command-line tool for querying
the processed (Task 2 feature-engineered) traffic dataset.

Commands (5, comfortably over the "at least three" requirement):
    query    Look up traffic/weather conditions at a specific date/time.
    busiest  Identify the highest-traffic periods of the day.
    quietest Identify recommended (lowest-traffic) travel periods.
    compare  Compare weekday vs. weekend traffic.
    weather  Retrieve weather + traffic information for a specific date,
             or aggregate traffic stats for a weather condition.

Design note on print() vs. logging: this is an interactive CLI tool, so its
actual query RESULTS are written to stdout with print() -- that is the
program's normal output channel and what the user runs it to see. The
logging module is reserved for the *operational* trail: which command ran,
with what arguments, and what went wrong -- exactly what the brief asks for.

Usage:
    python app.py query --datetime "2017-06-15 08:00"
    python app.py busiest --top 5 --day-type weekday
    python app.py quietest --top 5 --day-type weekend
    python app.py compare
    python app.py weather --date 2017-06-15
    python app.py weather --condition rain
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from pipeline import configure_logging  # reuse Task 1's logging setup

logger = logging.getLogger(__name__)


class UserInputError(Exception):
    """Raised for invalid user input (bad date, unknown value, etc.) -- always
    caught and turned into a clean logged ERROR, never a raw traceback."""


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_dataset(path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        logger.error("Dataset not found at %s (run feature_engineering.py first)", path, exc_info=True)
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

    df["date_time"] = pd.to_datetime(df["date_time"])
    logger.info("Loaded dataset: %d rows, %d columns (source: %s)", *df.shape, path.name)
    return df


# ---------------------------------------------------------------------------
# Shared validation helpers
# ---------------------------------------------------------------------------
DATETIME_FORMATS = ["%Y-%m-%d %H:%M", "%Y-%m-%d %H", "%Y-%m-%d"]


def parse_datetime_arg(raw: str) -> pd.Timestamp:
    """Parse a user-supplied date/time string. Raises UserInputError (never
    lets a raw ValueError/traceback reach the user) on a malformed value."""
    for fmt in DATETIME_FORMATS:
        try:
            return pd.Timestamp(pd.to_datetime(raw, format=fmt))
        except (ValueError, TypeError):
            continue
    raise UserInputError(
        f"Could not understand date/time '{raw}'. Expected one of: "
        f"{', '.join(f for f in DATETIME_FORMATS)} (e.g. '2017-06-15 08:00')."
    )


def parse_date_arg(raw: str) -> pd.Timestamp:
    try:
        return pd.Timestamp(pd.to_datetime(raw, format="%Y-%m-%d"))
    except (ValueError, TypeError):
        raise UserInputError(f"Could not understand date '{raw}'. Expected format: YYYY-MM-DD (e.g. '2017-06-15').")


def validate_day_type(value: str) -> str:
    allowed = {"all", "weekday", "weekend"}
    if value not in allowed:
        raise UserInputError(f"Invalid day-type '{value}'. Must be one of: {', '.join(sorted(allowed))}.")
    return value


def filter_by_day_type(df: pd.DataFrame, day_type: str) -> pd.DataFrame:
    if day_type == "weekday":
        return df[df["is_weekend"] == 0]
    if day_type == "weekend":
        return df[df["is_weekend"] == 1]
    return df


def format_row(row: pd.Series) -> str:
    return (
        f"{row['date_time']:%Y-%m-%d %H:%M} ({row['day_name']})  "
        f"traffic={row['traffic_volume']:,.0f} veh/hr  "
        f"category={row['congestion_category']}  "
        f"weather={row['weather_main']}  "
        f"temp={row['temp'] - 273.15:.1f}°C"
    )


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------
def cmd_query(df: pd.DataFrame, args: argparse.Namespace) -> None:
    target = parse_datetime_arg(args.datetime)

    exact = df[df["date_time"] == target]
    if not exact.empty:
        print(f"Exact match for {target:%Y-%m-%d %H:%M}:")
        print("  " + format_row(exact.iloc[0]))
        return

    # No exact reading at that hour (the dataset has coverage gaps) --
    # fall back to the nearest available reading rather than failing outright.
    nearest_idx = (df["date_time"] - target).abs().idxmin()
    nearest = df.loc[nearest_idx]
    gap = abs(nearest["date_time"] - target)
    print(f"No reading recorded exactly at {target:%Y-%m-%d %H:%M}.")
    print(f"Nearest available reading ({gap} away):")
    print("  " + format_row(nearest))


def _rank_periods(df: pd.DataFrame, day_type: str, top_n: int, ascending: bool) -> pd.DataFrame:
    scoped = filter_by_day_type(df, day_type)
    if scoped.empty:
        raise UserInputError(f"No data available for day-type '{day_type}'.")
    ranked = (
        scoped.groupby("hour")["traffic_volume"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "avg_traffic_volume", "count": "hours_observed"})
        .sort_values("avg_traffic_volume", ascending=ascending)
    )
    return ranked.head(top_n)


def cmd_busiest(df: pd.DataFrame, args: argparse.Namespace) -> None:
    day_type = validate_day_type(args.day_type)
    if args.top < 1:
        raise UserInputError(f"--top must be a positive integer, got {args.top}.")
    ranked = _rank_periods(df, day_type, args.top, ascending=False)

    print(f"Top {len(ranked)} busiest hour(s) of day ({day_type}):")
    for hour, row in ranked.iterrows():
        print(f"  {hour:02d}:00  avg {row['avg_traffic_volume']:,.0f} veh/hr  "
              f"(based on {row['hours_observed']:,.0f} observed hour(s))")


def cmd_quietest(df: pd.DataFrame, args: argparse.Namespace) -> None:
    day_type = validate_day_type(args.day_type)
    if args.top < 1:
        raise UserInputError(f"--top must be a positive integer, got {args.top}.")
    ranked = _rank_periods(df, day_type, args.top, ascending=True)

    print(f"Recommended travel window(s) -- {len(ranked)} quietest hour(s) of day ({day_type}):")
    for hour, row in ranked.iterrows():
        print(f"  {hour:02d}:00  avg {row['avg_traffic_volume']:,.0f} veh/hr  "
              f"(based on {row['hours_observed']:,.0f} observed hour(s))")


def cmd_compare(df: pd.DataFrame, _args: argparse.Namespace) -> None:
    weekday = df.loc[df["is_weekend"] == 0, "traffic_volume"]
    weekend = df.loc[df["is_weekend"] == 1, "traffic_volume"]

    print("Weekday vs. weekend traffic volume:")
    print(f"  {'':12}{'Weekday':>12}{'Weekend':>12}{'Difference':>14}")
    for label, wd_val, we_val in [
        ("Mean", weekday.mean(), weekend.mean()),
        ("Median", weekday.median(), weekend.median()),
        ("Std dev", weekday.std(), weekend.std()),
        ("Hours (n)", weekday.count(), weekend.count()),
    ]:
        diff = wd_val - we_val
        print(f"  {label:12}{wd_val:12,.1f}{we_val:12,.1f}{diff:14,.1f}")


def cmd_weather(df: pd.DataFrame, args: argparse.Namespace) -> None:
    if args.date is None and args.condition is None:
        raise UserInputError("weather command requires either --date or --condition.")
    if args.date is not None and args.condition is not None:
        raise UserInputError("Please supply only one of --date or --condition, not both.")

    if args.date is not None:
        day = parse_date_arg(args.date)
        day_rows = df[df["date_time"].dt.date == day.date()].sort_values("date_time")
        if day_rows.empty:
            print(f"No readings found for {day:%Y-%m-%d}.")
            return
        print(f"Weather + traffic on {day:%Y-%m-%d} ({len(day_rows)} hour(s) recorded):")
        for _, row in day_rows.iterrows():
            print("  " + format_row(row))
        return

    condition = args.condition.strip().lower()
    matches = df[df["weather_main"] == condition]
    if matches.empty:
        available = ", ".join(sorted(df["weather_main"].unique()))
        raise UserInputError(f"Unknown weather condition '{args.condition}'. Available: {available}.")

    print(f"Traffic under '{condition}' conditions ({len(matches):,} hour(s) observed):")
    print(f"  Average traffic volume: {matches['traffic_volume'].mean():,.0f} veh/hr")
    print(f"  Median traffic volume:  {matches['traffic_volume'].median():,.0f} veh/hr")
    print(f"  Congestion breakdown:")
    for category, count in matches["congestion_category"].value_counts().reindex(
        ["Low", "Medium", "High", "Very High"]
    ).items():
        pct = 100 * count / len(matches) if pd.notna(count) else 0
        print(f"    {category:10} {count:>6,.0f} hour(s)  ({pct:4.1f}%)")


COMMANDS = {
    "query": cmd_query,
    "busiest": cmd_busiest,
    "quietest": cmd_quietest,
    "compare": cmd_compare,
    "weather": cmd_weather,
}


# ---------------------------------------------------------------------------
# CLI plumbing
# ---------------------------------------------------------------------------
def build_parser(here: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="app.py", description="Mini Traffic Analytics Application (Metro Interstate Traffic Volume).",
    )
    parser.add_argument("--input", type=Path, default=here / "traffic_features.csv",
                         help="Path to the feature-engineered CSV (Task 2 output).")
    parser.add_argument("--log-file", type=Path, default=here / "app.log", help="Path to write the log file.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable DEBUG-level logging.")

    sub = parser.add_subparsers(dest="command", required=True)

    p_query = sub.add_parser("query", help="Look up traffic/weather at a specific date/time.")
    p_query.add_argument("--datetime", required=True, help="e.g. '2017-06-15 08:00' or '2017-06-15'.")

    p_busiest = sub.add_parser("busiest", help="Identify the highest-traffic hour(s) of the day.")
    p_busiest.add_argument("--top", type=int, default=5, help="How many hours to show (default: 5).")
    p_busiest.add_argument("--day-type", default="all", choices=["all", "weekday", "weekend"])

    p_quietest = sub.add_parser("quietest", help="Identify recommended (quietest) travel hour(s).")
    p_quietest.add_argument("--top", type=int, default=5, help="How many hours to show (default: 5).")
    p_quietest.add_argument("--day-type", default="all", choices=["all", "weekday", "weekend"])

    sub.add_parser("compare", help="Compare weekday vs. weekend traffic statistics.")

    p_weather = sub.add_parser("weather", help="Weather + traffic info for a date, or stats for a condition.")
    p_weather.add_argument("--date", help="e.g. '2017-06-15' -- show every recorded hour that day.")
    p_weather.add_argument("--condition", help="e.g. 'rain' -- show aggregate stats for that weather_main.")

    return parser


def main(argv: list[str] | None = None) -> int:
    here = Path(__file__).resolve().parent
    parser = build_parser(here)
    args = parser.parse_args(argv)

    configure_logging(args.log_file, verbose=args.verbose)
    logger.info("Command invoked: '%s' with arguments: %s", args.command, vars(args))

    try:
        df = load_dataset(args.input)
        COMMANDS[args.command](df, args)
    except UserInputError as e:
        # Invalid user input: a clean, single-line ERROR -- never a raw traceback.
        logger.error("Invalid input for command '%s': %s", args.command, e)
        return 1
    except (FileNotFoundError, pd.errors.EmptyDataError, pd.errors.ParserError,
            OSError, UnicodeDecodeError):
        logger.error("Command '%s' aborted: could not load the dataset", args.command, exc_info=True)
        return 1
    except Exception:
        logger.error("Command '%s' aborted due to an unexpected error", args.command, exc_info=True)
        return 1

    logger.info("Command '%s' completed successfully", args.command)
    return 0


if __name__ == "__main__":
    sys.exit(main())
