#!/usr/bin/env python3
"""
explain_shap.py
Part 3, Task 3 (explainability half): computes SHAP values for the
RandomForestRegressor that lstm_forecast.py trained (the "comparable
tree-based model trained on the same problem" the brief allows using in
place of direct LSTM explainability), and reports the result as a beeswarm
plot, a mean-|SHAP| feature-importance chart, and a plain-language summary
of the top drivers of the forecast, generated from the SHAP values
themselves rather than hand-written.

Why SHAP is applied to the Random Forest, not the LSTM, directly:
    shap.TreeExplainer computes EXACT Shapley values for tree ensembles in
    polynomial time. The LSTM's own explainability routes (DeepExplainer /
    GradientExplainer) are approximate, meaningfully slower on this CPU-only
    environment, and version-fragile against PyTorch autograd internals --
    and even where they run, their output is a (24 timesteps x 11 features)
    attribution tensor PER PREDICTION, which has no clean single summary:
    attributing across 264 (timestep, feature) pairs does not reduce to a
    short, readable "top features" list the way attributing across 14 named
    features does. The brief explicitly allows explaining "a comparable
    tree-based or linear model trained on the same problem" instead, as long
    as the choice is documented here -- so this script explains the Random
    Forest that lstm_forecast.py already trained on the SAME forecasting
    target (next-hour traffic_volume), from hand-engineered lag/rolling/
    calendar/weather features built from the identical 24-hour window the
    LSTM consumes (see sequence_common.py). The two models are close enough
    in test-set accuracy (see lstm_regression_comparison.csv -- Random
    Forest and LSTM MAE differ by under 1%) that the Random Forest's
    attributions are a reasonable, if imperfect, stand-in for what generally
    drives the forecast -- though they describe the tree model's own
    reasoning, not literally the LSTM's internal computation.

Usage:
    python explain_shap.py [--model PATH] [--test-features PATH] [--output-dir PATH]
                            [--top-n N] [--log-file PATH] [-v]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from ml_common import UserInputError, configure_logging
from sequence_common import TREE_FEATURES

logger = logging.getLogger(__name__)

CHART_COLOR = "#2a78d6"  # reused from the palette used throughout Parts 1-3

# Plain-language label + optional grouping for each raw feature column.
# hour_sin/hour_cos and dow_sin/dow_cos are two halves of one cyclical
# encoding (Part 2's design for time-of-day / day-of-week) -- neither half
# has a meaningful standalone "higher value -> higher forecast" direction on
# its own, so they're grouped and reported together as "hour of day" /
# "day of week", with combined magnitude but no single direction claim.
FEATURE_GROUPS = {
    "lag_1": {"label": "traffic volume one hour earlier", "columns": ["lag_1"], "cyclical": False},
    "lag_24": {"label": "traffic volume 24 hours earlier (same hour, previous day)", "columns": ["lag_24"], "cyclical": False},
    "roll_mean_6": {"label": "average traffic over the past 6 hours", "columns": ["roll_mean_6"], "cyclical": False},
    "roll_mean_24": {"label": "average traffic over the past 24 hours", "columns": ["roll_mean_24"], "cyclical": False},
    "hour_of_day": {"label": "hour of day (cyclical encoding)", "columns": ["hour_sin", "hour_cos"], "cyclical": True},
    "day_of_week": {"label": "day of week (cyclical encoding)", "columns": ["dow_sin", "dow_cos"], "cyclical": True},
    "is_weekend": {"label": "whether it's a weekend", "columns": ["is_weekend"], "cyclical": False},
    "holiday_flag": {"label": "whether it's a recognized holiday", "columns": ["holiday_flag"], "cyclical": False},
    "temp_zscore": {"label": "temperature (standardized)", "columns": ["temp_zscore"], "cyclical": False},
    "is_precipitating": {"label": "whether it's raining/snowing", "columns": ["is_precipitating"], "cyclical": False},
    "is_severe_weather": {"label": "whether there's severe weather (thunderstorm/squall)", "columns": ["is_severe_weather"], "cyclical": False},
    "is_low_visibility": {"label": "whether visibility is reduced (fog/mist/haze/smoke)", "columns": ["is_low_visibility"], "cyclical": False},
}


def load_model_and_data(model_path: Path, test_features_path: Path) -> tuple:
    import joblib

    if not model_path.exists():
        raise UserInputError(
            f"Model file not found: {model_path}. Run lstm_forecast.py first -- it saves this file."
        )
    if not test_features_path.exists():
        raise UserInputError(
            f"Test feature file not found: {test_features_path}. Run lstm_forecast.py first -- it saves this file."
        )

    model = joblib.load(model_path)
    table = pd.read_csv(test_features_path)

    missing = [c for c in TREE_FEATURES if c not in table.columns]
    if missing:
        raise UserInputError(
            f"Test feature file {test_features_path} is missing required column(s): {missing}. "
            f"It may be from an older run -- re-run lstm_forecast.py to regenerate it."
        )

    X = table[TREE_FEATURES].copy()
    logger.info("Loaded Random Forest model from %s and %d test rows from %s", model_path, len(X), test_features_path)
    return model, X, table


def compute_shap_values(model, X: pd.DataFrame) -> shap.Explanation:
    logger.info("Computing SHAP values for %d rows x %d features via TreeExplainer (exact, may take a minute) ...", *X.shape)
    explainer = shap.TreeExplainer(model)
    explanation = explainer(X)
    logger.info(
        "Computed SHAP values. Base value (mean model output over training data): %.1f veh/hr",
        float(np.mean(explanation.base_values)),
    )
    return explanation


def plot_beeswarm(explanation: shap.Explanation, output_dir: Path) -> Path:
    fig = plt.figure(figsize=(9, 6))
    shap.plots.beeswarm(explanation, show=False, max_display=len(TREE_FEATURES))
    fig = plt.gcf()
    fig.suptitle("SHAP Beeswarm: Feature Impact on Next-Hour Traffic Forecast", y=1.02)
    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "shap_beeswarm.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved figure: %s", path)
    return path


def build_group_importance(explanation: shap.Explanation, X: pd.DataFrame) -> pd.DataFrame:
    """Combines hour_sin/hour_cos and dow_sin/dow_cos into single 'hour of
    day' / 'day of week' entries (their contributions are additive under
    SHAP's efficiency property, so summing per-row is a valid combined
    contribution), and computes, for every group, its mean |SHAP| magnitude
    and -- for non-cyclical, single-column groups -- the Pearson correlation
    between the raw feature value and its SHAP value, as a data-driven proxy
    for "higher value pushes the forecast up or down"."""
    shap_df = pd.DataFrame(explanation.values, columns=TREE_FEATURES)
    rows = []
    for group_id, spec in FEATURE_GROUPS.items():
        combined_shap = shap_df[spec["columns"]].sum(axis=1)
        mean_abs = combined_shap.abs().mean()

        direction, corr = None, None
        if not spec["cyclical"]:
            col = spec["columns"][0]
            if X[col].std() > 0:
                corr = float(np.corrcoef(X[col], combined_shap)[0, 1])
                direction = "higher" if corr > 0 else "lower"

        rows.append({
            "feature_group": group_id,
            "label": spec["label"],
            "mean_abs_shap": mean_abs,
            "correlation_with_shap": corr,
            "direction": direction,
            "cyclical": spec["cyclical"],
        })

    result = pd.DataFrame(rows).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    return result


def plot_group_importance(importance: pd.DataFrame, output_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(9, 6))
    ordered = importance.sort_values("mean_abs_shap")
    ax.barh(ordered["label"], ordered["mean_abs_shap"], color=CHART_COLOR)
    ax.set_xlabel("Mean |SHAP value| (veh/hr impact on forecast)")
    ax.set_title("Feature Importance by Mean Absolute SHAP Value")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "shap_feature_importance.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Saved figure: %s", path)
    return path


def write_plain_language_summary(importance: pd.DataFrame, top_n: int, output_dir: Path) -> Path:
    lines = ["# SHAP Explainability -- Top Drivers of the Next-Hour Traffic Forecast", ""]
    top = importance.head(top_n)
    for rank, row in enumerate(top.itertuples(), start=1):
        if row.cyclical:
            sentence = (
                f"{rank}. **{row.label}** -- mean |SHAP| impact of {row.mean_abs_shap:,.0f} veh/hr. "
                f"This is a cyclical (sine/cosine) encoding, so its effect on the forecast depends on "
                f"which specific hour/day it represents rather than a single 'higher is more' direction -- "
                f"consistent with the rush-hour and weekday/weekend patterns found throughout this project."
            )
        else:
            direction_phrase = {
                "higher": "push the forecast UP",
                "lower": "push the forecast DOWN",
            }.get(row.direction, "have an unclear directional effect")
            sentence = (
                f"{rank}. **{row.label}** -- mean |SHAP| impact of {row.mean_abs_shap:,.0f} veh/hr. "
                f"Higher values of this feature {direction_phrase} "
                f"(Pearson correlation between feature value and SHAP value: r={row.correlation_with_shap:+.2f})."
            )
        lines.append(sentence)
        logger.info("Top feature #%d: %s", rank, sentence)

    text = "\n".join(lines) + "\n"
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "shap_top_features.md"
    path.write_text(text)
    logger.info("Wrote plain-language SHAP summary to %s", path)
    return path


def save_importance_table(importance: pd.DataFrame, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "shap_feature_importance.csv"
    importance.to_csv(path, index=False)
    logger.info("Wrote feature importance table to %s", path)
    return path


def run(model_path: Path, test_features_path: Path, top_n: int, output_dir: Path) -> pd.DataFrame:
    model, X, _table = load_model_and_data(model_path, test_features_path)
    explanation = compute_shap_values(model, X)

    plot_beeswarm(explanation, output_dir)
    importance = build_group_importance(explanation, X)
    plot_group_importance(importance, output_dir)
    save_importance_table(importance, output_dir)
    write_plain_language_summary(importance, top_n, output_dir)
    return importance


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Compute and report SHAP explainability for the Random Forest forecasting model."
    )
    parser.add_argument("--model", type=Path, default=here / "artifacts" / "lstm_task_random_forest.joblib",
                         help="Path to the trained Random Forest model (from lstm_forecast.py).")
    parser.add_argument("--test-features", type=Path, default=here / "artifacts" / "lstm_task_test_features.csv",
                         help="Path to the saved test-set feature table (from lstm_forecast.py).")
    parser.add_argument("--top-n", type=int, default=6,
                         help="How many top feature groups to summarize in plain language (default: 6).")
    parser.add_argument("--output-dir", type=Path, default=here / "artifacts",
                         help="Directory to write charts, tables, and the plain-language summary.")
    parser.add_argument("--log-file", type=Path, default=here / "explain_shap.log",
                         help="Path to write the log file.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable DEBUG-level logging.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging(args.log_file, verbose=args.verbose)
    logger.info("Command invoked: explain_shap with arguments: %s", vars(args))

    if args.top_n < 1:
        logger.error("Invalid input: --top-n must be at least 1, got %d.", args.top_n)
        return 1

    try:
        run(args.model, args.test_features, args.top_n, args.output_dir)
    except UserInputError as e:
        logger.error("Invalid input: %s", e)
        return 1
    except (FileNotFoundError, pd.errors.EmptyDataError, pd.errors.ParserError,
            OSError, UnicodeDecodeError):
        logger.error("SHAP explainability aborted: could not load model or data", exc_info=True)
        return 1
    except Exception:
        logger.error("SHAP explainability aborted due to an unexpected error", exc_info=True)
        return 1

    logger.info(
        "SHAP explainability completed successfully -- see %s for the plain-language summary",
        args.output_dir / "shap_top_features.md",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
