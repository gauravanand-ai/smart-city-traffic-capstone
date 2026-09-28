#!/usr/bin/env python3
"""
train_regression.py
Part 3, Task 1 (regression half): train two supervised regressors to predict
traffic_volume, on the same common feature set used by train_classification.py.

Models trained:
    - Linear Regression (baseline, scaled features)
    - Random Forest Regressor (tree-based ensemble)

Usage:
    python train_regression.py [--input PATH] [--output-dir PATH] [--log-file PATH] [-v]
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
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ml_common import (
    COMMON_FEATURE_COLUMNS, UserInputError, chronological_split,
    configure_logging, get_feature_matrix, load_dataset, prepare_ml_dataset,
)

logger = logging.getLogger(__name__)

COLOR_LINEAR = "#2a78d6"   # blue -- same pair as train_classification.py, for consistency
COLOR_FOREST = "#eb6834"   # orange
COLOR_IDEAL = "#9a9a9a"    # neutral gray (perfect-prediction reference, not a series)

TARGET_COLUMN = "traffic_volume"


def build_models() -> dict:
    """A linear baseline (in a scaling Pipeline) and a tree-based ensemble --
    the same two-algorithms-per-model requirement as the classifier, applied
    to the regression task."""
    return {
        "linear_regression": Pipeline([
            ("scaler", StandardScaler()),
            ("reg", LinearRegression()),
        ]),
        "random_forest": RandomForestRegressor(
            n_estimators=150, max_depth=10, min_samples_leaf=5, random_state=42, n_jobs=-1,
        ),
    }


def evaluate_model(name: str, model, X_test: pd.DataFrame, y_test: pd.Series) -> tuple[dict, pd.Series]:
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    r2 = r2_score(y_test, y_pred)

    metrics = {"model": name, "mae": mae, "r2": r2}
    logger.info("[%s] MAE=%.1f veh/hr  R2=%.4f  (RMSE=%.1f veh/hr, DEBUG-level extra)", name, mae, r2, rmse)
    logger.debug("[%s] RMSE=%.4f", name, rmse)
    return metrics, pd.Series(y_pred, index=y_test.index)


def plot_predicted_vs_actual(y_test: pd.Series, pred_by_model: dict, output_dir: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), sharex=True, sharey=True)
    colors = {"linear_regression": COLOR_LINEAR, "random_forest": COLOR_FOREST}
    titles = {"linear_regression": "Linear Regression", "random_forest": "Random Forest"}

    lims = [0, max(y_test.max(), max(p.max() for p in pred_by_model.values())) * 1.02]

    for ax, (name, y_pred) in zip(axes, pred_by_model.items()):
        ax.scatter(y_test, y_pred, s=6, alpha=0.15, color=colors[name], linewidths=0)
        ax.plot(lims, lims, color=COLOR_IDEAL, linewidth=1.5, linestyle="--")
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        ax.set_title(f"{titles[name]}\nMAE = {mae:,.0f} veh/hr, R² = {r2:.3f}")
        ax.set_xlabel("Actual traffic volume (veh/hr)")
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel("Predicted traffic volume (veh/hr)")
    fig.suptitle("Predicted vs. Actual Traffic Volume (test set)")
    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "regression_predicted_vs_actual.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Saved figure: %s", path)
    return path


def save_comparison_table(rows: list[dict], output_dir: Path) -> pd.DataFrame:
    comparison = pd.DataFrame(rows).set_index("model")
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "regression_model_comparison.csv"
    comparison.to_csv(path)
    logger.info("Wrote regression model comparison table to %s", path)
    return comparison


def save_models(models: dict, output_dir: Path) -> None:
    import joblib
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, model in models.items():
        path = output_dir / f"regression_{name}.joblib"
        joblib.dump(model, path)
        logger.info("Saved trained model: %s", path)


def run(input_path: Path, output_dir: Path) -> pd.DataFrame:
    df = load_dataset(input_path)
    df = prepare_ml_dataset(df)

    train_df, test_df = chronological_split(df)
    X_train, y_train = get_feature_matrix(train_df), train_df[TARGET_COLUMN]
    X_test, y_test = test_df[COMMON_FEATURE_COLUMNS], test_df[TARGET_COLUMN]

    logger.info(
        "Training set traffic_volume mean=%.1f, std=%.1f | Test set mean=%.1f, std=%.1f",
        y_train.mean(), y_train.std(), y_test.mean(), y_test.std(),
    )

    models = build_models()
    results, pred_by_model = [], {}
    for name, model in models.items():
        logger.info("Fitting %s ...", name)
        model.fit(X_train, y_train)
        metrics, y_pred = evaluate_model(name, model, X_test, y_test)
        results.append(metrics)
        pred_by_model[name] = y_pred

    comparison = save_comparison_table(results, output_dir)
    plot_predicted_vs_actual(y_test, pred_by_model, output_dir)
    save_models(models, output_dir)
    return comparison


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Train and compare regression models for traffic_volume."
    )
    parser.add_argument("--input", type=Path, default=here.parent / "pipeline" / "traffic_features.csv",
                         help="Path to Part 2's feature-engineered CSV.")
    parser.add_argument("--output-dir", type=Path, default=here / "artifacts",
                         help="Directory to write the comparison table, chart, and saved models.")
    parser.add_argument("--log-file", type=Path, default=here / "train_regression.log",
                         help="Path to write the log file.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable DEBUG-level logging.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging(args.log_file, verbose=args.verbose)
    logger.info("Command invoked: train_regression with arguments: %s", vars(args))

    try:
        run(args.input, args.output_dir)
    except UserInputError as e:
        logger.error("Invalid input: %s", e)
        return 1
    except (FileNotFoundError, pd.errors.EmptyDataError, pd.errors.ParserError,
            OSError, UnicodeDecodeError):
        logger.error("Regression training aborted: could not load the dataset", exc_info=True)
        return 1
    except Exception:
        logger.error("Regression training aborted due to an unexpected error", exc_info=True)
        return 1

    logger.info(
        "Regression training completed successfully -- see %s for the full model comparison table",
        args.output_dir / "regression_model_comparison.csv",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
