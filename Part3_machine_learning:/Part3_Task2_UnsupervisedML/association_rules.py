#!/usr/bin/env python3
"""
association_rules.py
Part 3, Task 2 (association rule mining half): discretize time of day,
weekday type, and weather, then mine association rules that predict
congestion level (Low/Medium/High/Severe -- the same proxy congestion
category used throughout Part 3). Reports the rules with the highest lift,
each with a plain-language explanation generated from the rule itself.

Usage:
    python association_rules.py [--input PATH] [--min-support F] [--output-dir PATH] [--log-file PATH] [-v]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

from ml_common import (
    UserInputError, add_time_of_day, add_weather_severity, configure_logging,
    load_dataset, prepare_ml_dataset,
)

logger = logging.getLogger(__name__)

ITEM_COLUMNS = ["time_of_day", "weekday_type", "weather", "congestion"]
CONGESTION_ORDER = ["Low", "Medium", "High", "Severe"]
CHART_COLOR = "#2a78d6"  # reused from the palette used throughout Parts 1-3

# Plain-language phrase for each item value, used to turn a mined rule like
# {time_of_day_Overnight (0-5), weekday_type_Weekend} -> {congestion_Low}
# into a sentence, rather than hand-writing one explanation per rule.
_PHRASES = {
    "time_of_day": {
        "Overnight (0-5)": "it's overnight (midnight-5am)",
        "Morning Rush (6-9)": "it's the morning rush (6-9am)",
        "Midday (10-15)": "it's midday (10am-3pm)",
        "Evening Rush (16-18)": "it's the evening rush (4-6pm)",
        "Evening (19-23)": "it's evening (7-11pm)",
    },
    "weekday_type": {"Weekday": "it's a weekday", "Weekend": "it's the weekend"},
    "weather": {
        "Clear": "the weather is clear",
        "Clouds": "the weather is cloudy",
        "Precipitation": "it's raining/snowing",
        "Low visibility": "visibility is reduced (fog/mist/haze/smoke)",
        "Severe": "there's severe weather (thunderstorm/squall)",
    },
}


def build_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """One 'transaction' per hourly record: which time-of-day bin, weekday
    type, weather-severity tier, and congestion level it belongs to -- the
    standard one-hot 'basket' format apriori expects."""
    df = df.copy()
    df["weekday_type"] = df["is_weekend"].map({0: "Weekday", 1: "Weekend"})

    items = df[["time_of_day", "weekday_type", "weather_severity_label", "congestion_category_p3"]].copy()
    items.columns = ITEM_COLUMNS
    basket = pd.get_dummies(items.astype(str))
    logger.info(
        "Built %d transactions x %d items (%s)",
        *basket.shape, ", ".join(f"{c}={items[c].nunique()} values" for c in ITEM_COLUMNS),
    )
    return basket


def mine_rules(basket: pd.DataFrame, min_support: float, min_confidence: float) -> pd.DataFrame:
    frequent_itemsets = apriori(basket, min_support=min_support, use_colnames=True, max_len=4)
    logger.info("Found %d frequent itemset(s) at min_support=%.3f", len(frequent_itemsets), min_support)

    if frequent_itemsets.empty:
        raise UserInputError(
            f"No itemsets meet min_support={min_support} -- every combination of time/weekday/weather/"
            f"congestion is rarer than that in the data. Try a lower --min-support."
        )

    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)
    rules = rules[rules["confidence"] >= min_confidence]
    logger.info(
        "Mined %d association rule(s) (lift > 1.0, confidence >= %.2f)", len(rules), min_confidence,
    )

    congestion_items = {f"congestion_{c}" for c in CONGESTION_ORDER}
    is_congestion_rule = rules["consequents"].apply(lambda s: len(s) == 1 and next(iter(s)) in congestion_items)
    congestion_rules = rules[is_congestion_rule].copy()
    congestion_rules["consequent"] = congestion_rules["consequents"].apply(lambda s: next(iter(s)).removeprefix("congestion_"))
    congestion_rules = congestion_rules.sort_values("lift", ascending=False).reset_index(drop=True)
    logger.info(
        "%d of those rule(s) predict a congestion level specifically (the required target)",
        len(congestion_rules),
    )
    if congestion_rules.empty:
        logger.warning(
            "No congestion-predicting rules survived min_support=%.3f -- try a lower --min-support",
            min_support,
        )
    return congestion_rules


def _antecedent_phrase(antecedent: frozenset) -> str:
    clauses = []
    for item in antecedent:
        col, _, value = item.partition("_")
        # values can themselves contain underscores (e.g. weekday_type), so
        # split on the first underscore only, then match against ITEM_COLUMNS
        for candidate_col in ITEM_COLUMNS:
            if item.startswith(candidate_col + "_"):
                value = item[len(candidate_col) + 1:]
                clauses.append(_PHRASES[candidate_col].get(value, f"{candidate_col}={value}") if candidate_col in _PHRASES else f"{candidate_col}={value}")
                break
    return " and ".join(clauses)


def explain_rule(row: pd.Series) -> str:
    condition = _antecedent_phrase(row["antecedents"])
    return (
        f"When {condition}, congestion is '{row['consequent']}' {100 * row['confidence']:.1f}% of the time -- "
        f"{row['lift']:.2f}x more likely than for a random hour. This combination covers "
        f"{100 * row['support']:.1f}% of all hours in the dataset."
    )


def save_rules_table(rules: pd.DataFrame, output_dir: Path, top_n: int) -> pd.DataFrame:
    top = rules.head(top_n).copy()
    top["antecedents_readable"] = top["antecedents"].apply(lambda s: ", ".join(sorted(s)))
    top["explanation"] = top.apply(explain_rule, axis=1)

    output_dir.mkdir(parents=True, exist_ok=True)
    all_path = output_dir / "association_rules_all.csv"
    top_path = output_dir / "association_rules_top.csv"
    rules.assign(antecedents=lambda d: d["antecedents"].apply(lambda s: ", ".join(sorted(s)))).drop(columns=["consequents"]).to_csv(all_path, index=False)
    top[["antecedents_readable", "consequent", "support", "confidence", "lift", "explanation"]].to_csv(top_path, index=False)
    logger.info("Wrote all %d congestion rule(s) to %s", len(rules), all_path)
    logger.info("Wrote top %d rule(s) by lift to %s", len(top), top_path)

    for i, row in top.iterrows():
        logger.info("#%d (lift=%.2f): %s", i + 1, row["lift"], row["explanation"])
    return top


def plot_top_rule_per_congestion_level(rules: pd.DataFrame, output_dir: Path) -> Path:
    """The single highest-lift rule for each of the 4 congestion levels --
    more informative as a chart than the raw top-N-by-lift list, which tends
    to be dominated by near-duplicate variants of the same one or two
    patterns (e.g. several 'weekend overnight' combinations all predicting
    Low)."""
    best_per_level = (
        rules.sort_values("lift", ascending=False)
        .groupby("consequent", as_index=False).first()
    )
    best_per_level["consequent"] = pd.Categorical(best_per_level["consequent"], categories=CONGESTION_ORDER, ordered=True)
    best_per_level = best_per_level.sort_values("consequent")
    best_per_level["antecedents_readable"] = best_per_level["antecedents"].apply(
        lambda s: _antecedent_phrase(s).replace("it's ", "").replace("the weather is ", "").capitalize()
    )

    import textwrap

    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.barh(best_per_level["consequent"].astype(str), best_per_level["lift"], color=CHART_COLOR)
    for i, (lift, label) in enumerate(zip(best_per_level["lift"], best_per_level["antecedents_readable"])):
        wrapped = "\n".join(textwrap.wrap(f"lift {lift:.2f} -- {label}", width=38))
        ax.text(lift + 0.06, i, wrapped, va="center", fontsize=9)
    ax.set_xlabel("Lift")
    ax.set_title("Highest-Lift Rule Predicting Each Congestion Level")
    ax.set_xlim(0, best_per_level["lift"].max() * 3.0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "association_rules_by_congestion_level.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Saved figure: %s", path)
    return path


def run(input_path: Path, min_support: float, min_confidence: float, top_n: int, output_dir: Path) -> pd.DataFrame:
    df = load_dataset(input_path)
    df = prepare_ml_dataset(df)
    df = add_weather_severity(df)
    df = add_time_of_day(df)

    basket = build_transactions(df)
    rules = mine_rules(basket, min_support, min_confidence)
    if rules.empty:
        raise UserInputError(
            f"No congestion-predicting rules found at min_support={min_support}. Try a lower --min-support."
        )

    top = save_rules_table(rules, output_dir, top_n)
    plot_top_rule_per_congestion_level(rules, output_dir)
    return top


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Mine association rules predicting congestion level from time/weekday/weather."
    )
    parser.add_argument("--input", type=Path, default=here.parent / "pipeline" / "traffic_features.csv",
                         help="Path to Part 2's feature-engineered CSV.")
    parser.add_argument("--min-support", type=float, default=0.01,
                         help="Minimum itemset support for apriori (default: 0.01).")
    parser.add_argument("--min-confidence", type=float, default=0.5,
                         help="Minimum rule confidence to keep (default: 0.5).")
    parser.add_argument("--top-n", type=int, default=15,
                         help="How many top-by-lift rules to report in detail (default: 15).")
    parser.add_argument("--output-dir", type=Path, default=here / "artifacts",
                         help="Directory to write the rule tables and chart.")
    parser.add_argument("--log-file", type=Path, default=here / "association_rules.log",
                         help="Path to write the log file.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable DEBUG-level logging.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging(args.log_file, verbose=args.verbose)
    logger.info("Command invoked: association_rules with arguments: %s", vars(args))

    if not (0 < args.min_support < 1):
        logger.error("Invalid input: --min-support must be between 0 and 1, got %s.", args.min_support)
        return 1
    if not (0 < args.min_confidence <= 1):
        logger.error("Invalid input: --min-confidence must be between 0 and 1, got %s.", args.min_confidence)
        return 1

    try:
        run(args.input, args.min_support, args.min_confidence, args.top_n, args.output_dir)
    except UserInputError as e:
        logger.error("Invalid input: %s", e)
        return 1
    except (FileNotFoundError, pd.errors.EmptyDataError, pd.errors.ParserError,
            OSError, UnicodeDecodeError):
        logger.error("Association rule mining aborted: could not load the dataset", exc_info=True)
        return 1
    except Exception:
        logger.error("Association rule mining aborted due to an unexpected error", exc_info=True)
        return 1

    logger.info(
        "Association rule mining completed successfully -- see %s for the top rules",
        args.output_dir / "association_rules_top.csv",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
