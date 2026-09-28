#!/usr/bin/env python3
"""
route_recommender.py
Part 3, Task 5: a Traffic Recommendation System that turns this project's
analysis and trained model outputs into practical travel-timing advice.

The dataset represents a single corridor (westbound I-94), so -- per the
brief -- this recommends WHEN to travel, not which route to take:

    "For a weekday journey, consider travelling between 10:00 and 11:00 AM,
    when historical traffic volumes are typically lower."

How a recommendation is built, for a given day type (weekday/weekend) and an
optional weather condition:
  1. The headline "lower-traffic" signal is the ACTUAL HISTORICAL mean
     traffic_volume for each hour of the day, filtered to matching
     day-type/weather rows -- transparent and directly auditable, and it's
     literally what the brief's own example sentence refers to ("historical
     traffic volumes"). This is not a model prediction.
  2. A secondary RISK signal for each hour comes from Task 1's already-
     trained Random Forest classifier (`classification_random_forest.joblib`,
     predicting the proxy `high_risk` label): its predicted probability is
     averaged over the real historical rows matching that hour/day-type/
     weather, rather than fabricating a synthetic input row -- this is the
     literal "transform ... model outputs into a practical system" the brief
     asks for, and stays robust even for a sparse (hour, weather) cell, since
     the model generalizes across features rather than needing many exact
     historical matches.
  3. Task 2's mined association rules are cross-referenced: if the requested
     day-type/weather combination matches a known high-lift rule predicting
     High/Severe congestion, that's surfaced as a caution alongside the
     recommendation.
  4. Consecutive low-traffic hours (below the search range's own median) are
     merged into windows, ranked by traffic level, and turned into the
     plain-language sentences printed below.

Usage:
    python route_recommender.py --day-type weekday [--weather Clear]
                                 [--start 0] [--end 23] [--top-n 3]
                                 [--input PATH] [--output-dir PATH] [--log-file PATH] [-v]
    python route_recommender.py --date 2025-06-18 --weather Rain
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ml_common import (
    COMMON_FEATURE_COLUMNS, UserInputError, add_time_of_day, add_weather_severity,
    configure_logging, load_dataset, prepare_ml_dataset,
)

logger = logging.getLogger(__name__)

COLOR_RECOMMEND = "#1baf7a"  # green -- reused from cluster_traffic.py's palette
COLOR_NEUTRAL = "#2a78d6"    # blue
COLOR_AVOID = "#e34948"      # red

VALID_WEATHER = [
    "Clear", "Clouds", "Drizzle", "Fog", "Haze", "Mist",
    "Rain", "Smoke", "Snow", "Squall", "Thunderstorm",
]

MIN_CELL_SIZE = 10       # below this many historical rows, warn the hour's estimate is low-confidence
RISK_ELEVATED_RATIO = 1.5  # an hour's risk is "elevated" if its P(high_risk) exceeds 1.5x the search range's own average
RULE_MIN_LIFT = 2.0       # only surface an association-rule caution if it's a genuinely strong pattern


def resolve_day_type(day_type: str | None, date: str | None) -> tuple[str, int, str | None]:
    """Returns (day_type_label, is_weekend_flag, resolved_date_note)."""
    if day_type and date:
        raise UserInputError("Pass either --day-type or --date, not both.")
    if not day_type and not date:
        raise UserInputError("Pass one of --day-type {weekday,weekend} or --date YYYY-MM-DD.")

    if date:
        try:
            parsed = datetime.strptime(date, "%Y-%m-%d")
        except ValueError:
            raise UserInputError(f"--date '{date}' is not a valid YYYY-MM-DD date.")
        is_weekend = parsed.weekday() >= 5
        label = "weekend" if is_weekend else "weekday"
        note = f"{date} is a {parsed.strftime('%A')}"
        return label, int(is_weekend), note

    if day_type not in ("weekday", "weekend"):
        raise UserInputError(f"--day-type must be 'weekday' or 'weekend', got '{day_type}'.")
    return day_type, int(day_type == "weekend"), None


def resolve_weather(weather: str | None) -> str | None:
    if weather is None:
        return None
    match = next((w for w in VALID_WEATHER if w.lower() == weather.lower()), None)
    if match is None:
        raise UserInputError(
            f"Unknown --weather '{weather}'. Valid options (case-insensitive): {', '.join(VALID_WEATHER)}."
        )
    return match


def build_hourly_profile(df: pd.DataFrame, classifier, is_weekend_flag: int, weather: str | None) -> pd.DataFrame:
    """One row per hour of day (0-23): historical mean/median traffic_volume,
    sample size, dominant congestion category, and the classifier's mean
    predicted high_risk probability over the matching historical rows."""
    subset = df[df["is_weekend"] == is_weekend_flag]
    if weather:
        subset = subset[subset["weather_main"].str.lower() == weather.lower()]
    if subset.empty:
        raise UserInputError(
            "No historical rows match that day type" + (f" + weather='{weather}'" if weather else "")
            + " -- try a different --weather, or omit it."
        )

    rows = []
    for hour in range(24):
        h_subset = subset[subset["hour"] == hour]
        n = len(h_subset)
        if n == 0:
            rows.append({"hour": hour, "n": 0, "traffic_mean": np.nan, "traffic_median": np.nan,
                         "risk_prob": np.nan, "dominant_congestion": None})
            continue

        traffic_mean = h_subset["traffic_volume"].mean()
        traffic_median = h_subset["traffic_volume"].median()
        X = h_subset[COMMON_FEATURE_COLUMNS]
        risk_prob = float(classifier.predict_proba(X)[:, 1].mean())
        dominant_congestion = h_subset["congestion_category_p3"].mode().iat[0]

        if n < MIN_CELL_SIZE:
            logger.warning(
                "Hour %02d:00 has only %d historical row(s) for this day-type/weather combination -- "
                "its estimate is low-confidence.", hour, n,
            )

        rows.append({
            "hour": hour, "n": n, "traffic_mean": traffic_mean, "traffic_median": traffic_median,
            "risk_prob": risk_prob, "dominant_congestion": dominant_congestion,
        })

    profile = pd.DataFrame(rows).set_index("hour")
    logger.info(
        "Built 24-hour profile from %d historical row(s) (%d hour(s) with data)",
        len(subset), profile["n"].gt(0).sum(),
    )
    return profile


MAX_RECOMMENDED_WINDOW_HOURS = 2  # keeps a recommended window compact/actionable
                                    # (e.g. "10:00-11:00"), not a sprawling
                                    # multi-hour block -- see build_compact_windows.


def merge_into_windows(hours: list[int], profile: pd.DataFrame) -> list[dict]:
    """Merges a sorted list of individual hour-of-day integers into
    contiguous [start, end) windows (e.g. hours [9, 10] -> 09:00-11:00),
    each carrying that window's own average traffic/risk. Used for the
    'avoid' side, where the FULL extent of a busy period (e.g. all of rush
    hour) is more useful than an artificially short summary."""
    if not hours:
        return []
    hours = sorted(hours)
    windows, current = [], [hours[0]]
    for h in hours[1:]:
        if h == current[-1] + 1:
            current.append(h)
        else:
            windows.append(current)
            current = [h]
    windows.append(current)

    result = []
    for w in windows:
        sub = profile.loc[w]
        result.append({
            "start": w[0], "end": w[-1] + 1,
            "traffic_mean": sub["traffic_mean"].mean(),
            "risk_prob": sub["risk_prob"].mean(),
            "n": int(sub["n"].sum()),
        })
    return result


def build_compact_windows(candidates: pd.DataFrame, baseline_traffic: float) -> list[dict]:
    """Builds recommended windows greedily from the single best (lowest-
    traffic) hour outward: each window starts at a not-yet-used hour below
    the baseline, then optionally grows by one more hour -- whichever
    adjacent neighbor is lower-traffic -- capped at
    MAX_RECOMMENDED_WINDOW_HOURS. This keeps each reported window short and
    genuinely below baseline (never merging in an above-baseline hour just
    because it happens to be adjacent), unlike a single unbounded
    below-threshold run, which for this dataset's deep overnight lull would
    otherwise merge 5-6 hours into one block -- true, but not the kind of
    specific, actionable window the brief's own example ('10:00 and 11:00
    AM') illustrates."""
    below = candidates[candidates["traffic_mean"] < baseline_traffic].sort_values("traffic_mean")
    used: set[int] = set()
    windows = []

    for hour in below.index:
        if hour in used:
            continue
        window = [hour]
        used.add(hour)
        while len(window) < MAX_RECOMMENDED_WINDOW_HOURS:
            left, right = window[0] - 1, window[-1] + 1
            left_ok = left in candidates.index and left not in used and candidates.loc[left, "traffic_mean"] < baseline_traffic
            right_ok = right in candidates.index and right not in used and candidates.loc[right, "traffic_mean"] < baseline_traffic
            if not left_ok and not right_ok:
                break
            if left_ok and (not right_ok or candidates.loc[left, "traffic_mean"] <= candidates.loc[right, "traffic_mean"]):
                window.insert(0, left)
                used.add(left)
            else:
                window.append(right)
                used.add(right)

        windows.append({
            "start": min(window), "end": max(window) + 1,
            "traffic_mean": candidates.loc[window, "traffic_mean"].mean(),
            "risk_prob": candidates.loc[window, "risk_prob"].mean(),
            "n": int(candidates.loc[window, "n"].sum()),
        })

    windows.sort(key=lambda w: w["traffic_mean"])
    return windows


def recommend_windows(profile: pd.DataFrame, start: int, end: int, top_n: int) -> tuple[list[dict], list[dict], float, float]:
    """Returns (recommended windows ascending by traffic, worst window(s) to
    avoid, baseline_traffic, baseline_risk) -- baselines are the search
    range's own average, used to phrase recommendations relative to what's
    'typical' for the hours the user is actually willing to consider. Both
    sides are anchored to the same baseline (the range's mean traffic), so a
    'recommended' window is always genuinely below it and an 'avoid' window
    always genuinely above it -- never a threshold artifact."""
    candidates = profile.loc[start:end].dropna(subset=["traffic_mean"])
    if candidates.empty:
        raise UserInputError(
            f"No historical data for any hour between {start:02d}:00 and {end:02d}:00 "
            f"under this day-type/weather combination."
        )

    baseline_traffic = candidates["traffic_mean"].mean()
    baseline_risk = candidates["risk_prob"].mean()

    recommended = build_compact_windows(candidates, baseline_traffic)

    high_hours = candidates[candidates["traffic_mean"] >= baseline_traffic].index.tolist()
    avoid = merge_into_windows(high_hours, profile)
    avoid.sort(key=lambda w: -w["traffic_mean"])

    return recommended[:top_n], avoid[:1], baseline_traffic, baseline_risk


def find_risk_rule(rules_df: pd.DataFrame | None, day_type_label: str, weather_severity_label: str | None) -> pd.Series | None:
    """Looks for the highest-lift Task 2 association rule whose antecedent
    matches this exact day-type (+ weather severity, if given) and whose
    consequent is High/Severe congestion -- a qualitative cross-check from a
    completely different technique (market-basket analysis vs. a trained
    classifier) pointing at the same kind of risky combination."""
    if rules_df is None or rules_df.empty:
        return None

    weekday_item = f"weekday_type_{'Weekend' if day_type_label == 'weekend' else 'Weekday'}"
    weather_item = f"weather_{weather_severity_label}" if weather_severity_label else None

    def matches(antecedents: str) -> bool:
        items = [a.strip() for a in str(antecedents).split(",")]
        if weekday_item not in items:
            return False
        if weather_item and weather_item not in items:
            return False
        return True

    candidates = rules_df[
        rules_df["antecedents"].apply(matches)
        & rules_df["consequent"].isin(["High", "Severe"])
        & (rules_df["lift"] >= RULE_MIN_LIFT)
    ]
    if candidates.empty:
        return None
    return candidates.sort_values("lift", ascending=False).iloc[0]


def format_window(w: dict) -> str:
    return f"{w['start']:02d}:00-{w['end']:02d}:00"


def build_recommendation_text(
    day_type_label: str, date_note: str | None, weather: str | None,
    recommended: list[dict], avoid: list[dict], baseline_traffic: float, baseline_risk: float,
    risk_rule: pd.Series | None,
) -> list[str]:
    weather_clause = f" in {weather.lower()} conditions" if weather else ""
    date_clause = f" ({date_note})" if date_note else ""
    lines = []

    if not recommended:
        lines.append(f"No clearly lower-traffic window was found for a {day_type_label} journey{weather_clause}.")
    else:
        best = recommended[0]
        pct_lower = 100 * (baseline_traffic - best["traffic_mean"]) / baseline_traffic
        lines.append(
            f"For a {day_type_label} journey{date_clause}{weather_clause}, consider travelling between "
            f"{format_window(best)}, when historical traffic volumes are typically {pct_lower:.0f}% lower "
            f"than the average for the hours considered (~{best['traffic_mean']:,.0f} vs. "
            f"~{baseline_traffic:,.0f} vehicles/hour)."
        )
        if best["risk_prob"] > RISK_ELEVATED_RATIO * baseline_risk and baseline_risk > 0:
            lines.append(
                f"Note: even this window shows a somewhat elevated historical accident-risk signal "
                f"({best['risk_prob']:.0%} vs. a typical {baseline_risk:.0%} for the hours considered) -- "
                f"likely weather-related; check current conditions before travelling."
            )
        for alt in recommended[1:]:
            pct = 100 * (baseline_traffic - alt["traffic_mean"]) / baseline_traffic
            lines.append(
                f"Also worth considering: {format_window(alt)} (~{pct:.0f}% lower than average, "
                f"~{alt['traffic_mean']:,.0f} vehicles/hour)."
            )

    if avoid:
        worst = avoid[0]
        pct_higher = 100 * (worst["traffic_mean"] - baseline_traffic) / baseline_traffic
        lines.append(
            f"Avoid if possible: {format_window(worst)}, typically ~{pct_higher:.0f}% busier than average "
            f"(~{worst['traffic_mean']:,.0f} vehicles/hour)."
        )

    if risk_rule is not None:
        lines.append(
            f"Heads up (from Task 2's association-rule mining): historically, "
            f"{risk_rule['antecedents'].replace('_', ' ')} predicts '{risk_rule['consequent']}' congestion "
            f"{100 * risk_rule['confidence']:.0f}% of the time ({risk_rule['lift']:.1f}x more likely than a "
            f"random hour) -- worth keeping in mind regardless of which window you choose."
        )

    return lines


def plot_profile(
    profile: pd.DataFrame, recommended: list[dict], avoid: list[dict],
    day_type_label: str, weather: str | None, output_dir: Path,
) -> Path:
    fig, ax = plt.subplots(figsize=(11, 5.5))
    recommended_hours = {h for w in recommended for h in range(w["start"], w["end"])}
    avoid_hours = {h for w in avoid for h in range(w["start"], w["end"])}

    colors = []
    for hour in profile.index:
        if hour in recommended_hours:
            colors.append(COLOR_RECOMMEND)
        elif hour in avoid_hours:
            colors.append(COLOR_AVOID)
        else:
            colors.append(COLOR_NEUTRAL)

    ax.bar(profile.index, profile["traffic_mean"], color=colors)
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Historical mean traffic volume (veh/hr)")
    weather_suffix = f", {weather} weather" if weather else ""
    ax.set_title(f"Hourly Traffic Profile -- {day_type_label.capitalize()}{weather_suffix}")
    ax.set_xticks(range(0, 24, 2))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=COLOR_RECOMMEND, label="Recommended (lower traffic)"),
        plt.Rectangle((0, 0), 1, 1, color=COLOR_AVOID, label="Avoid (higher traffic)"),
        plt.Rectangle((0, 0), 1, 1, color=COLOR_NEUTRAL, label="Neutral / outside search range"),
    ]
    ax.legend(handles=handles, frameon=False, loc="upper left")
    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{weather.lower()}" if weather else ""
    path = output_dir / f"route_recommendation_{day_type_label}{suffix}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Saved figure: %s", path)
    return path


def save_profile_table(profile: pd.DataFrame, day_type_label: str, weather: str | None, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{weather.lower()}" if weather else ""
    path = output_dir / f"route_recommendation_{day_type_label}{suffix}.csv"
    profile.to_csv(path)
    logger.info("Wrote hourly profile table to %s", path)
    return path


def run(
    input_path: Path, day_type: str | None, date: str | None, weather: str | None,
    start: int, end: int, top_n: int, output_dir: Path,
) -> list[str]:
    day_type_label, is_weekend_flag, date_note = resolve_day_type(day_type, date)
    weather = resolve_weather(weather)

    if not (0 <= start <= 23 and 0 <= end <= 23 and start <= end):
        raise UserInputError(f"--start/--end must satisfy 0 <= start <= end <= 23, got start={start}, end={end}.")

    import joblib
    classifier_path = output_dir / "classification_random_forest.joblib"
    if not classifier_path.exists():
        raise UserInputError(
            f"Trained classifier not found at {classifier_path}. Run train_classification.py "
            f"(Part 3 Task 1) first -- it saves this file."
        )
    classifier = joblib.load(classifier_path)
    logger.info("Loaded trained classifier from %s", classifier_path)

    rules_path = output_dir / "association_rules_all.csv"
    rules_df = pd.read_csv(rules_path) if rules_path.exists() else None
    if rules_df is None:
        logger.warning(
            "Association rules table not found at %s -- run association_rules.py (Part 3 Task 2) first "
            "for the rule-mining cross-check; continuing without it.", rules_path,
        )

    df = load_dataset(input_path)
    df = prepare_ml_dataset(df)
    df = add_weather_severity(df)
    df = add_time_of_day(df)

    profile = build_hourly_profile(df, classifier, is_weekend_flag, weather)
    recommended, avoid, baseline_traffic, baseline_risk = recommend_windows(profile, start, end, top_n)

    weather_severity_label = None
    if weather:
        weather_severity_label = df.loc[df["weather_main"].str.lower() == weather.lower(), "weather_severity_label"].iat[0]
    risk_rule = find_risk_rule(rules_df, day_type_label, weather_severity_label)

    lines = build_recommendation_text(
        day_type_label, date_note, weather, recommended, avoid, baseline_traffic, baseline_risk, risk_rule,
    )

    save_profile_table(profile, day_type_label, weather, output_dir)
    plot_profile(profile, recommended, avoid, day_type_label, weather, output_dir)

    return lines


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Recommend lower-traffic travel windows for the I-94 corridor from historical data "
                     "and Part 3's trained models."
    )
    day_group = parser.add_argument_group("day type (choose exactly one)")
    day_group.add_argument("--day-type", choices=["weekday", "weekend"], help="Travel on a typical weekday or weekend.")
    day_group.add_argument("--date", type=str, help="Travel on a specific date (YYYY-MM-DD) -- day type is derived from it.")
    parser.add_argument("--weather", type=str, default=None,
                         help=f"Expected weather condition. One of: {', '.join(VALID_WEATHER)}. Omit for an all-weather average.")
    parser.add_argument("--start", type=int, default=0, help="Earliest hour to consider, 0-23 (default: 0).")
    parser.add_argument("--end", type=int, default=23, help="Latest hour to consider, 0-23 (default: 23).")
    parser.add_argument("--top-n", type=int, default=3, help="How many recommended windows to report (default: 3).")
    parser.add_argument("--input", type=Path, default=here.parent / "pipeline" / "traffic_features.csv",
                         help="Path to Part 2's feature-engineered CSV.")
    parser.add_argument("--output-dir", type=Path, default=here / "artifacts",
                         help="Directory to read trained-model/rule artifacts from and write the profile table/chart to.")
    parser.add_argument("--log-file", type=Path, default=here / "route_recommender.log",
                         help="Path to write the log file.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable DEBUG-level logging.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging(args.log_file, verbose=args.verbose)
    logger.info("Command invoked: route_recommender with arguments: %s", vars(args))

    if args.top_n < 1:
        logger.error("Invalid input: --top-n must be at least 1, got %d.", args.top_n)
        return 1

    try:
        lines = run(
            args.input, args.day_type, args.date, args.weather,
            args.start, args.end, args.top_n, args.output_dir,
        )
    except UserInputError as e:
        logger.error("Invalid input: %s", e)
        return 1
    except (FileNotFoundError, pd.errors.EmptyDataError, pd.errors.ParserError,
            OSError, UnicodeDecodeError):
        logger.error("Recommendation aborted: could not load the dataset", exc_info=True)
        return 1
    except Exception:
        logger.error("Recommendation aborted due to an unexpected error", exc_info=True)
        return 1

    # This tool's actual output -- the plain-language recommendation -- is
    # printed, not logged: it's the interactive result the user asked for,
    # the same print()-for-results/logging-for-status split app.py (Part 2
    # Task 4) already established for this project's interactive CLI tools.
    print()
    for line in lines:
        print(line)
    print()

    logger.info("Recommendation generated successfully -- see %s for the supporting profile table/chart", args.output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
