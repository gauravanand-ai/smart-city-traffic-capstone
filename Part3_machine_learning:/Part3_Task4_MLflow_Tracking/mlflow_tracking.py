#!/usr/bin/env python3
"""
mlflow_tracking.py
Part 3, Task 4 (Advanced AI Technique): MLflow experiment tracking applied to
every model already built in this capstone -- two classifiers and two
regressors (Task 1), plus three forecasting models (Task 3: naive baseline,
Random Forest, LSTM). See RESULTS_TASK4.md for the full "why/how/value/
limitations" write-up the brief asks for; this docstring covers the how.

Why MLflow, briefly: eight-plus models have been trained across this
capstone so far, each previously compared only within its own task's ad hoc
CSV/chart. MLflow adds one queryable, model-agnostic record of every run's
hyperparameters, metrics, and artifacts -- letting any model from any task be
compared side by side, reloaded, or audited later without re-running scripts
or re-reading multiple files.

How this script is built: it does NOT duplicate any training logic. It
imports train_classification.py, train_regression.py, and lstm_forecast.py
as modules and calls their existing build_models() / evaluate_model() /
train_random_forest() / train_lstm() / predict_lstm() / naive_persistence_
predictions() functions directly -- the exact same code paths (same
hyperparameters, same feature sets) already verified in Tasks 1 and 3. This
script's own job is narrow: wrap each already-defined training/evaluation
step in an mlflow.start_run() context and log params/metrics/artifacts. The
one production-code change this required was a minimal, backward-compatible
epoch_callback hook added to lstm_forecast.train_lstm() (default None, zero
effect on that script's own CLI), used here to log the LSTM's per-epoch
train/val loss as MLflow step metrics -- without adding an MLflow dependency
to lstm_forecast.py itself.

Runs are grouped under one MLflow experiment as three parent runs
(classification / regression / forecasting comparisons), each with one
nested child run per model -- mirroring how the three tasks are already
organized in this project. After training, the script queries the tracking
store itself (mlflow.search_runs(), not the in-memory results) to build a
cross-model leaderboard chart, demonstrating that MLflow's own record is a
complete, sufficient source for that comparison.

Tracking store: local SQLite database (mlflow.db) for run metadata, with
artifacts (models, charts) under the experiment's default local artifact
directory (mlruns/), both under ml/ by default. Browse interactively with
`mlflow ui --backend-store-uri sqlite:///mlflow.db` from that directory
(see RESULTS_TASK4.md).

Usage:
    python mlflow_tracking.py [--input PATH] [--tracking-uri URI] [--experiment NAME]
                               [--epochs N] [--patience N] [--output-dir PATH]
                               [--log-file PATH] [-v]
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

# Set before importing mlflow: keeps this run fully local (no anonymous usage
# ping to MLflow's telemetry endpoint) and silences an unrelated startup hint
# about MLflow's separate LLM-tracing feature, which this script doesn't use
# -- this is classic ML experiment tracking (params/metrics/artifacts), not
# LLM/agent tracing.
os.environ.setdefault("MLFLOW_DISABLE_TELEMETRY", "1")
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.pytorch
import mlflow.sklearn
import pandas as pd
from mlflow.models import infer_signature

import lstm_forecast
import sequence_common
import train_classification
import train_regression
from ml_common import (
    COMMON_FEATURE_COLUMNS, UserInputError, chronological_split,
    configure_logging, get_feature_matrix, load_dataset, prepare_ml_dataset,
)

logger = logging.getLogger(__name__)

COLOR_A = "#2a78d6"  # blue -- consistent with the palette used throughout Parts 1-3
COLOR_B = "#eb6834"  # orange
COLOR_C = "#1baf7a"  # green


def flatten_sklearn_params(params: dict) -> dict:
    """sklearn's estimator.get_params(deep=True) includes both nested step
    objects (e.g. a Pipeline's 'clf' key -> the LogisticRegression instance
    itself) and their individual dotted hyperparameters (e.g. 'clf__C' ->
    1.0). Keeps only scalar, genuinely loggable values, so a Pipeline's own
    step objects don't get logged as noisy stringified reprs alongside their
    real hyperparameters."""
    return {k: v for k, v in params.items() if isinstance(v, (str, int, float, bool, type(None)))}


def track_classification(df: pd.DataFrame, output_dir: Path) -> None:
    train_df, test_df = chronological_split(df)
    X_train, y_train = get_feature_matrix(train_df), train_df[train_classification.TARGET_COLUMN]
    X_test, y_test = test_df[COMMON_FEATURE_COLUMNS], test_df[train_classification.TARGET_COLUMN]

    models = train_classification.build_models()
    with mlflow.start_run(run_name="classification_comparison") as parent:
        mlflow.set_tag("task", "classification")
        mlflow.set_tag("target", train_classification.TARGET_COLUMN)
        mlflow.log_param("n_train", len(X_train))
        mlflow.log_param("n_test", len(X_test))
        logger.info("Started parent MLflow run 'classification_comparison' (run_id=%s)", parent.info.run_id)

        for name, model in models.items():
            with mlflow.start_run(run_name=name, nested=True) as child:
                mlflow.set_tag("task", "classification")
                mlflow.set_tag("algorithm", name)
                mlflow.log_params(flatten_sklearn_params(model.get_params()))

                logger.info("Fitting %s under MLflow run %s ...", name, child.info.run_id)
                model.fit(X_train, y_train)
                metrics, _y_proba = train_classification.evaluate_model(name, model, X_test, y_test)
                mlflow.log_metrics({k: v for k, v in metrics.items() if k != "model"})

                signature = infer_signature(X_test, model.predict(X_test))
                mlflow.sklearn.log_model(
                    model, name="model", signature=signature,
                    input_example=X_test.head(3), serialization_format="pickle",
                )
                logger.info("Logged MLflow run for classification/%s (run_id=%s)", name, child.info.run_id)

        roc_path = output_dir / "classification_roc_curves.png"
        if roc_path.exists():
            mlflow.log_artifact(str(roc_path))


def track_regression(df: pd.DataFrame, output_dir: Path) -> None:
    train_df, test_df = chronological_split(df)
    X_train, y_train = get_feature_matrix(train_df), train_df[train_regression.TARGET_COLUMN]
    X_test, y_test = test_df[COMMON_FEATURE_COLUMNS], test_df[train_regression.TARGET_COLUMN]

    models = train_regression.build_models()
    with mlflow.start_run(run_name="regression_comparison") as parent:
        mlflow.set_tag("task", "regression")
        mlflow.set_tag("target", train_regression.TARGET_COLUMN)
        mlflow.log_param("n_train", len(X_train))
        mlflow.log_param("n_test", len(X_test))
        logger.info("Started parent MLflow run 'regression_comparison' (run_id=%s)", parent.info.run_id)

        for name, model in models.items():
            with mlflow.start_run(run_name=name, nested=True) as child:
                mlflow.set_tag("task", "regression")
                mlflow.set_tag("algorithm", name)
                mlflow.log_params(flatten_sklearn_params(model.get_params()))

                logger.info("Fitting %s under MLflow run %s ...", name, child.info.run_id)
                model.fit(X_train, y_train)
                metrics, _y_pred = train_regression.evaluate_model(name, model, X_test, y_test)
                mlflow.log_metrics({k: v for k, v in metrics.items() if k != "model"})

                signature = infer_signature(X_test, model.predict(X_test))
                mlflow.sklearn.log_model(
                    model, name="model", signature=signature,
                    input_example=X_test.head(3), serialization_format="pickle",
                )
                logger.info("Logged MLflow run for regression/%s (run_id=%s)", name, child.info.run_id)

        chart_path = output_dir / "regression_predicted_vs_actual.png"
        if chart_path.exists():
            mlflow.log_artifact(str(chart_path))


def track_forecasting(df: pd.DataFrame, output_dir: Path, hidden_size: int, num_layers: int,
                       batch_size: int, lr: float, epochs: int, patience: int) -> None:
    windowed = sequence_common.build_windowed_dataset(df)
    if len(windowed["seq_X_train"]) == 0 or len(windowed["seq_X_val"]) == 0 or len(windowed["seq_X_test"]) == 0:
        raise UserInputError(
            "One or more of the train/val/test splits has zero windowed samples -- the input dataset "
            "does not have enough contiguous hourly coverage to build a 24-hour window in every split."
        )
    y_test = windowed["y_test"]

    with mlflow.start_run(run_name="forecasting_comparison") as parent:
        mlflow.set_tag("task", "forecasting")
        mlflow.set_tag("target", "traffic_volume (next hour)")
        mlflow.log_param("window_hours", sequence_common.WINDOW)
        mlflow.log_param("n_train", len(windowed["seq_X_train"]))
        mlflow.log_param("n_val", len(windowed["seq_X_val"]))
        mlflow.log_param("n_test", len(windowed["seq_X_test"]))
        logger.info("Started parent MLflow run 'forecasting_comparison' (run_id=%s)", parent.info.run_id)

        # -- Naive persistence baseline: no training, no model artifact, but
        # still tracked so it's comparable in the same store as the two
        # trained models, exactly as its role in RESULTS_TASK3.md intends.
        with mlflow.start_run(run_name="naive_persistence", nested=True) as child:
            mlflow.set_tag("task", "forecasting")
            mlflow.set_tag("algorithm", "naive_persistence")
            mlflow.set_tag("model_type", "baseline")
            y_pred = lstm_forecast.naive_persistence_predictions(windowed["tree_X_test"])
            metrics = lstm_forecast.evaluate("naive_persistence", y_test, y_pred)
            mlflow.log_metrics({k: v for k, v in metrics.items() if k != "model"})
            logger.info("Logged MLflow run for forecasting/naive_persistence (run_id=%s)", child.info.run_id)

        # -- Random Forest (lag/rolling/calendar/weather features)
        with mlflow.start_run(run_name="random_forest", nested=True) as child:
            mlflow.set_tag("task", "forecasting")
            mlflow.set_tag("algorithm", "random_forest")
            logger.info("Fitting random_forest under MLflow run %s ...", child.info.run_id)
            rf_model = lstm_forecast.train_random_forest(windowed["tree_X_train"], windowed["y_train"])
            mlflow.log_params(flatten_sklearn_params(rf_model.get_params()))
            rf_pred = rf_model.predict(windowed["tree_X_test"])
            metrics = lstm_forecast.evaluate("random_forest", y_test, rf_pred)
            mlflow.log_metrics({k: v for k, v in metrics.items() if k != "model"})

            signature = infer_signature(windowed["tree_X_test"], rf_pred)
            mlflow.sklearn.log_model(
                rf_model, name="model", signature=signature,
                input_example=windowed["tree_X_test"].head(3), serialization_format="pickle",
            )
            logger.info("Logged MLflow run for forecasting/random_forest (run_id=%s)", child.info.run_id)

        # -- LSTM: per-epoch train/val loss logged live as MLflow step
        # metrics via the epoch_callback hook, giving the same loss-curve
        # view as lstm_training_curve.png but inside MLflow's own UI.
        with mlflow.start_run(run_name="lstm", nested=True) as child:
            mlflow.set_tag("task", "forecasting")
            mlflow.set_tag("algorithm", "lstm")
            mlflow.log_params({
                "hidden_size": hidden_size, "num_layers": num_layers, "batch_size": batch_size,
                "lr": lr, "max_epochs": epochs, "patience": patience,
                "window_hours": sequence_common.WINDOW, "input_size": len(sequence_common.SEQUENCE_FEATURES),
            })
            logger.info("Training lstm under MLflow run %s ...", child.info.run_id)

            def epoch_callback(epoch: int, train_loss: float, val_loss: float) -> None:
                mlflow.log_metric("train_loss_scaled", train_loss, step=epoch)
                mlflow.log_metric("val_loss_scaled", val_loss, step=epoch)

            lstm_model, history = lstm_forecast.train_lstm(
                windowed, hidden_size, num_layers, batch_size, lr, epochs, patience,
                epoch_callback=epoch_callback,
            )
            best_epoch = int(history.loc[history["val_loss"].idxmin(), "epoch"])
            mlflow.log_metric("best_epoch", best_epoch)
            mlflow.log_metric("epochs_run", len(history))

            lstm_pred = lstm_forecast.predict_lstm(
                lstm_model, windowed["seq_X_test"], windowed["scaler_mean"], windowed["scaler_std"],
            )
            metrics = lstm_forecast.evaluate("lstm", y_test, lstm_pred)
            mlflow.log_metrics({k: v for k, v in metrics.items() if k != "model"})

            mlflow.pytorch.log_model(lstm_model, name="model", serialization_format="pickle")
            logger.info("Logged MLflow run for forecasting/lstm (run_id=%s)", child.info.run_id)

        chart_path = output_dir / "lstm_forecast_comparison.png"
        if chart_path.exists():
            mlflow.log_artifact(str(chart_path))


def build_leaderboard(experiment_id: str, output_dir: Path) -> pd.DataFrame:
    """Queries the MLflow tracking store itself (mlflow.search_runs) rather
    than any in-memory result from this script's own execution -- the point
    being that this leaderboard could be regenerated by anyone, at any later
    time, from the tracking store alone."""
    runs = mlflow.search_runs(experiment_ids=[experiment_id], order_by=["start_time ASC"])
    runs = runs[runs["tags.algorithm"].notna()].copy()  # drop the 3 parent/comparison runs

    keep_cols = ["tags.task", "tags.algorithm", "run_id"]
    metric_cols = sorted(c for c in runs.columns if c.startswith("metrics."))
    leaderboard = runs[keep_cols + metric_cols].rename(
        columns={"tags.task": "task", "tags.algorithm": "algorithm", **{c: c.removeprefix("metrics.") for c in metric_cols}}
    )
    leaderboard = leaderboard.sort_values(["task", "algorithm"]).reset_index(drop=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "mlflow_runs_leaderboard.csv"
    leaderboard.to_csv(path, index=False)
    logger.info(
        "Queried %d model run(s) from the MLflow tracking store and wrote leaderboard to %s",
        len(leaderboard), path,
    )
    return leaderboard


def plot_leaderboard(leaderboard: pd.DataFrame, output_dir: Path) -> Path:
    """Three panels, one per task, each showing the headline metric already
    used to pick a winner in that task's own RESULTS doc -- rebuilt here
    entirely from the MLflow-queried leaderboard, not from the original
    per-task CSVs."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    cls = leaderboard[leaderboard["task"] == "classification"].set_index("algorithm")
    axes[0].bar(cls.index, cls["f1_score"], color=[COLOR_A, COLOR_B][: len(cls)])
    axes[0].set_title("Classification\n(F1, high_risk)")
    axes[0].set_ylabel("F1 score")
    axes[0].tick_params(axis="x", rotation=20)

    reg = leaderboard[leaderboard["task"] == "regression"].set_index("algorithm")
    axes[1].bar(reg.index, reg["r2"], color=[COLOR_A, COLOR_B][: len(reg)])
    axes[1].set_title("Regression\n(R², traffic_volume)")
    axes[1].set_ylabel("R²")
    axes[1].tick_params(axis="x", rotation=20)

    fc = leaderboard[leaderboard["task"] == "forecasting"].set_index("algorithm")
    fc_order = [a for a in ["naive_persistence", "random_forest", "lstm"] if a in fc.index]
    axes[2].bar(fc_order, fc.loc[fc_order, "r2"], color=[COLOR_C, COLOR_B, COLOR_A][: len(fc_order)])
    axes[2].set_title("Forecasting\n(R², next-hour traffic_volume)")
    axes[2].set_ylabel("R²")
    axes[2].tick_params(axis="x", rotation=20)

    for ax in axes:
        ax.set_ylim(0, 1.05)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("All Part 3 Models, Queried from the MLflow Tracking Store")
    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "mlflow_runs_leaderboard.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Saved figure: %s", path)
    return path


def run(
    input_path: Path, tracking_uri: str, experiment_name: str, output_dir: Path,
    hidden_size: int, num_layers: int, batch_size: int, lr: float, epochs: int, patience: int,
) -> pd.DataFrame:
    mlflow.set_tracking_uri(tracking_uri)
    experiment = mlflow.set_experiment(experiment_name)
    logger.info(
        "MLflow tracking URI: %s | experiment: %s (experiment_id=%s)",
        tracking_uri, experiment_name, experiment.experiment_id,
    )

    df = load_dataset(input_path)
    df = prepare_ml_dataset(df)

    track_classification(df, output_dir)
    track_regression(df, output_dir)
    track_forecasting(df, output_dir, hidden_size, num_layers, batch_size, lr, epochs, patience)

    leaderboard = build_leaderboard(experiment.experiment_id, output_dir)
    plot_leaderboard(leaderboard, output_dir)
    return leaderboard


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Track every Part 3 model (classification, regression, forecasting) with MLflow."
    )
    parser.add_argument("--input", type=Path, default=here.parent / "pipeline" / "traffic_features.csv",
                         help="Path to Part 2's feature-engineered CSV.")
    parser.add_argument("--tracking-uri", type=str, default=f"sqlite:///{here / 'mlflow.db'}",
                         help="MLflow tracking store URI (default: local SQLite database under ml/).")
    parser.add_argument("--experiment", type=str, default="smart-city-traffic-intelligence",
                         help="MLflow experiment name.")
    parser.add_argument("--hidden-size", type=int, default=64, help="LSTM hidden state size (default: 64).")
    parser.add_argument("--num-layers", type=int, default=1, help="Number of stacked LSTM layers (default: 1).")
    parser.add_argument("--batch-size", type=int, default=128, help="LSTM training batch size (default: 128).")
    parser.add_argument("--lr", type=float, default=1e-3, help="LSTM Adam learning rate (default: 0.001).")
    parser.add_argument("--epochs", type=int, default=40, help="LSTM maximum training epochs (default: 40).")
    parser.add_argument("--patience", type=int, default=6,
                         help="LSTM early-stopping patience in epochs (default: 6).")
    parser.add_argument("--output-dir", type=Path, default=here / "artifacts",
                         help="Directory to read existing task charts from and write the leaderboard to.")
    parser.add_argument("--log-file", type=Path, default=here / "mlflow_tracking.log",
                         help="Path to write the log file.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable DEBUG-level logging.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging(args.log_file, verbose=args.verbose)
    logger.info("Command invoked: mlflow_tracking with arguments: %s", vars(args))

    if args.hidden_size < 1:
        logger.error("Invalid input: --hidden-size must be at least 1, got %d.", args.hidden_size)
        return 1
    if args.num_layers < 1:
        logger.error("Invalid input: --num-layers must be at least 1, got %d.", args.num_layers)
        return 1
    if args.batch_size < 1:
        logger.error("Invalid input: --batch-size must be at least 1, got %d.", args.batch_size)
        return 1
    if args.lr <= 0:
        logger.error("Invalid input: --lr must be positive, got %s.", args.lr)
        return 1
    if args.epochs < 1:
        logger.error("Invalid input: --epochs must be at least 1, got %d.", args.epochs)
        return 1
    if args.patience < 1:
        logger.error("Invalid input: --patience must be at least 1, got %d.", args.patience)
        return 1

    try:
        run(
            args.input, args.tracking_uri, args.experiment, args.output_dir,
            args.hidden_size, args.num_layers, args.batch_size, args.lr, args.epochs, args.patience,
        )
    except UserInputError as e:
        logger.error("Invalid input: %s", e)
        return 1
    except (FileNotFoundError, pd.errors.EmptyDataError, pd.errors.ParserError,
            OSError, UnicodeDecodeError):
        logger.error("MLflow tracking run aborted: could not load the dataset", exc_info=True)
        return 1
    except Exception:
        logger.error("MLflow tracking run aborted due to an unexpected error", exc_info=True)
        return 1

    logger.info(
        "MLflow tracking completed successfully -- see %s for the leaderboard queried back from the "
        "tracking store, or run `mlflow ui --backend-store-uri %s` from %s to browse interactively",
        args.output_dir / "mlflow_runs_leaderboard.csv", args.tracking_uri, args.output_dir.parent,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
