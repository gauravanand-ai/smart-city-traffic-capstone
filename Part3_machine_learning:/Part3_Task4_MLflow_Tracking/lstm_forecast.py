#!/usr/bin/env python3
"""
lstm_forecast.py
Part 3, Task 3 (LSTM half): forecasts next-hour traffic_volume from a 24-hour
window of history using an LSTM (PyTorch), and compares it against two
baselines on the same held-out test split -- a naive persistence baseline
(predict next hour = last observed hour) and a RandomForestRegressor trained
on hand-engineered lag/rolling features built from the same window (see
sequence_common.py for how the window/features are built, and why the tree
model's calendar/weather features are read "as of" the last observed hour
rather than the target hour).

This script also saves the trained RandomForest, its test-set feature table,
and the test-set actual/predicted values to disk, so explain_shap.py can load
them directly rather than recomputing the windowed dataset and retraining.
Why SHAP is applied to the RandomForest instead of to the LSTM directly is
explained in explain_shap.py's own docstring, per the brief's explicit
allowance to do this as long as the choice is documented.

Usage:
    python lstm_forecast.py [--input PATH] [--epochs N] [--hidden-size N]
                             [--num-layers N] [--batch-size N] [--lr F]
                             [--patience N] [--output-dir PATH] [--log-file PATH] [-v]
"""

from __future__ import annotations

import argparse
import copy
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ml_common import UserInputError, configure_logging, load_dataset, prepare_ml_dataset
from sequence_common import SEQUENCE_FEATURES, TREE_FEATURES, WINDOW, build_windowed_dataset

logger = logging.getLogger(__name__)

SEED = 42  # matches the random_state=42 used throughout Parts 1-3

COLOR_NAIVE = "#9a9a9a"   # neutral gray -- a non-learned reference, not a model
COLOR_FOREST = "#eb6834"  # orange -- same as train_regression.py's Random Forest
COLOR_LSTM = "#2a78d6"    # blue -- same slot train_regression.py used for its other model
COLOR_ACTUAL = "#1baf7a"  # green -- reused from cluster_traffic.py's palette


# ---------------------------------------------------------------------------
# LSTM model
# ---------------------------------------------------------------------------
class LSTMForecaster(nn.Module):
    """A standard many-to-one LSTM: consumes the WINDOW-length sequence of
    SEQUENCE_FEATURES and predicts a single scaled traffic_volume value for
    the next hour, from the hidden state at the final timestep. Kept as a
    plain sequence-to-one architecture (no separate future-calendar input)
    since the window's own trailing hour_sin/cos already implicitly encodes
    what hour comes next."""

    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 1, dropout: float = 0.0):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        last_timestep = out[:, -1, :]
        return self.head(last_timestep).squeeze(-1)


def train_lstm(
    windowed: dict, hidden_size: int, num_layers: int, batch_size: int, lr: float,
    epochs: int, patience: int, epoch_callback=None,
) -> tuple[LSTMForecaster, pd.DataFrame]:
    """Trains with early stopping on the validation split's (scaled) MSE loss,
    restoring the best-epoch weights before returning -- so a run that starts
    overfitting late doesn't hand back its worst-of-the-run model.

    epoch_callback, if given, is called as epoch_callback(epoch, train_loss,
    val_loss) after every epoch -- purely an observation hook (this function's
    own control flow, early stopping included, never depends on it). It exists
    so an external experiment tracker (see mlflow_tracking.py, Part 3 Task 4)
    can record the per-epoch loss curve as it happens, without this module
    taking on a tracking-library dependency itself: this stays a plain PyTorch
    training function, usable standalone exactly as before when the argument
    is omitted."""
    torch.manual_seed(SEED)
    mean, std = windowed["scaler_mean"], windowed["scaler_std"]

    def to_tensors(split: str) -> tuple[torch.Tensor, torch.Tensor]:
        X = torch.from_numpy(windowed[f"seq_X_{split}"])
        y_scaled = (windowed[f"y_{split}"] - mean) / std
        y = torch.from_numpy(y_scaled.astype(np.float32))
        return X, y

    X_train, y_train = to_tensors("train")
    X_val, y_val = to_tensors("val")

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)

    model = LSTMForecaster(input_size=len(SEQUENCE_FEATURES), hidden_size=hidden_size, num_layers=num_layers)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    logger.info(
        "Training LSTM: %d train / %d val sequences, hidden_size=%d, num_layers=%d, "
        "batch_size=%d, lr=%g, max_epochs=%d, early-stopping patience=%d",
        len(X_train), len(X_val), hidden_size, num_layers, batch_size, lr, epochs, patience,
    )

    best_val_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_total = 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            pred = model(X_batch)
            loss = loss_fn(pred, y_batch)
            loss.backward()
            optimizer.step()
            train_loss_total += loss.item() * len(X_batch)
        train_loss = train_loss_total / len(X_train)

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(X_val), y_val).item()

        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        logger.debug("Epoch %d: train_loss=%.5f val_loss=%.5f", epoch, train_loss, val_loss)
        if epoch_callback is not None:
            epoch_callback(epoch, train_loss, val_loss)

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            logger.info(
                "Early stopping at epoch %d (no val_loss improvement for %d epochs; best val_loss=%.5f at epoch %d)",
                epoch, patience, best_val_loss, epoch - epochs_without_improvement,
            )
            break
    else:
        logger.info("Completed all %d epochs without triggering early stopping (best val_loss=%.5f)", epochs, best_val_loss)

    model.load_state_dict(best_state)
    logger.info("Restored best-epoch LSTM weights (val_loss=%.5f)", best_val_loss)
    return model, pd.DataFrame(history)


def predict_lstm(model: LSTMForecaster, seq_X: np.ndarray, mean: float, std: float) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        pred_scaled = model(torch.from_numpy(seq_X)).numpy()
    return pred_scaled * std + mean


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------
def naive_persistence_predictions(tree_X: pd.DataFrame) -> np.ndarray:
    """Predicts the next hour's traffic_volume as simply equal to the last
    observed hour's (lag_1) -- the standard "no model" reference point any
    real forecaster must beat to be worth using."""
    return tree_X["lag_1"].to_numpy()


def train_random_forest(tree_X_train: pd.DataFrame, y_train: np.ndarray) -> RandomForestRegressor:
    model = RandomForestRegressor(
        n_estimators=150, max_depth=10, min_samples_leaf=5, random_state=SEED, n_jobs=-1,
    )
    logger.info("Fitting Random Forest on %d rows x %d lag/rolling features ...", *tree_X_train.shape)
    model.fit(tree_X_train, y_train)
    return model


# ---------------------------------------------------------------------------
# Evaluation and plotting
# ---------------------------------------------------------------------------
def evaluate(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    logger.info("[%s] MAE=%.1f veh/hr  R2=%.4f", name, mae, r2)
    return {"model": name, "mae": mae, "r2": r2}


def plot_training_curve(history: pd.DataFrame, output_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.plot(history["epoch"], history["train_loss"], color=COLOR_FOREST, label="Train loss")
    ax.plot(history["epoch"], history["val_loss"], color=COLOR_LSTM, label="Validation loss")
    best_epoch = int(history.loc[history["val_loss"].idxmin(), "epoch"])
    ax.axvline(best_epoch, color=COLOR_NAIVE, linestyle="--", linewidth=1, label=f"Best epoch ({best_epoch})")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE loss (scaled traffic_volume)")
    ax.set_title("LSTM Training Curve")
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "lstm_training_curve.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Saved figure: %s", path)
    return path


def plot_forecast_comparison(
    target_datetime: pd.Series, y_test: np.ndarray, preds: dict, output_dir: Path,
) -> Path:
    """Two panels: (left) a two-week slice of actual vs. predicted traffic so
    the models' forecasting behaviour is visible over time -- a scatter alone
    can't show whether a model tracks rises and falls or just regresses to
    the mean; (right) the scatter/R2 view matching train_regression.py's
    chart style, for direct visual comparison with Task 1's regression
    results."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    slice_hours = 24 * 14
    start = len(y_test) // 3  # an arbitrary interior slice, not the very start/end of the test period
    end = start + slice_hours
    ax = axes[0]
    ax.plot(target_datetime.iloc[start:end], y_test[start:end], color=COLOR_ACTUAL, linewidth=1.3, label="Actual")
    ax.plot(target_datetime.iloc[start:end], preds["random_forest"][start:end], color=COLOR_FOREST, linewidth=1, alpha=0.85, label="Random Forest")
    ax.plot(target_datetime.iloc[start:end], preds["lstm"][start:end], color=COLOR_LSTM, linewidth=1, alpha=0.85, label="LSTM")
    ax.set_ylabel("Traffic volume (veh/hr)")
    ax.set_title("Two-Week Forecast Slice (test set)")
    ax.tick_params(axis="x", rotation=30)
    ax.legend(frameon=False, fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax2 = axes[1]
    lims = [0, max(y_test.max(), preds["random_forest"].max(), preds["lstm"].max()) * 1.02]
    ax2.scatter(y_test, preds["random_forest"], s=6, alpha=0.12, color=COLOR_FOREST, linewidths=0, label="Random Forest")
    ax2.scatter(y_test, preds["lstm"], s=6, alpha=0.12, color=COLOR_LSTM, linewidths=0, label="LSTM")
    ax2.plot(lims, lims, color=COLOR_NAIVE, linewidth=1.5, linestyle="--")
    ax2.set_xlabel("Actual traffic volume (veh/hr)")
    ax2.set_ylabel("Predicted traffic volume (veh/hr)")
    ax2.set_xlim(lims)
    ax2.set_ylim(lims)
    ax2.set_title("Predicted vs. Actual (test set)")
    leg = ax2.legend(frameon=False, fontsize=9, markerscale=3)
    for lh in leg.legend_handles:
        lh.set_alpha(1)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    fig.suptitle("Next-Hour Traffic Volume Forecast: Random Forest vs. LSTM")
    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "lstm_forecast_comparison.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Saved figure: %s", path)
    return path


def save_comparison_table(rows: list[dict], output_dir: Path) -> pd.DataFrame:
    comparison = pd.DataFrame(rows).set_index("model")
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "lstm_regression_comparison.csv"
    comparison.to_csv(path)
    logger.info("Wrote forecast model comparison table to %s", path)
    return comparison


def save_artifacts_for_shap(
    rf_model: RandomForestRegressor, tree_X_test: pd.DataFrame, y_test: np.ndarray,
    rf_pred: np.ndarray, target_datetime: pd.Series, output_dir: Path,
) -> None:
    """Saves exactly what explain_shap.py needs so it never has to reload
    traffic_features.csv or rebuild the windowed dataset: the fitted RF model
    and its test-set feature table (with the actual/predicted values attached
    for reference in the SHAP write-up)."""
    import joblib
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / "lstm_task_random_forest.joblib"
    joblib.dump(rf_model, model_path)
    logger.info("Saved trained model: %s", model_path)

    test_table = tree_X_test.copy()
    test_table["target_datetime"] = target_datetime.values
    test_table["actual_traffic_volume"] = y_test
    test_table["rf_predicted_traffic_volume"] = rf_pred
    test_path = output_dir / "lstm_task_test_features.csv"
    test_table.to_csv(test_path, index=False)
    logger.info("Saved test-set feature table (for SHAP) to %s", test_path)


def save_lstm_model(model: LSTMForecaster, hidden_size: int, num_layers: int, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "lstm_forecast_model.pt"
    torch.save(
        {"state_dict": model.state_dict(), "hidden_size": hidden_size, "num_layers": num_layers,
         "input_size": len(SEQUENCE_FEATURES)},
        path,
    )
    logger.info("Saved trained LSTM weights to %s", path)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def run(
    input_path: Path, hidden_size: int, num_layers: int, batch_size: int, lr: float,
    epochs: int, patience: int, output_dir: Path,
) -> pd.DataFrame:
    np.random.seed(SEED)

    df = load_dataset(input_path)
    df = prepare_ml_dataset(df)
    windowed = build_windowed_dataset(df)

    if len(windowed["seq_X_train"]) == 0 or len(windowed["seq_X_val"]) == 0 or len(windowed["seq_X_test"]) == 0:
        raise UserInputError(
            "One or more of the train/val/test splits has zero windowed samples -- the input dataset "
            "does not have enough contiguous hourly coverage to build a 24-hour window in every split."
        )

    lstm_model, history = train_lstm(windowed, hidden_size, num_layers, batch_size, lr, epochs, patience)
    plot_training_curve(history, output_dir)
    save_lstm_model(lstm_model, hidden_size, num_layers, output_dir)

    rf_model = train_random_forest(windowed["tree_X_train"], windowed["y_train"])

    y_test = windowed["y_test"]
    preds = {
        "naive_persistence": naive_persistence_predictions(windowed["tree_X_test"]),
        "random_forest": rf_model.predict(windowed["tree_X_test"]),
        "lstm": predict_lstm(lstm_model, windowed["seq_X_test"], windowed["scaler_mean"], windowed["scaler_std"]),
    }

    results = [evaluate(name, y_test, y_pred) for name, y_pred in preds.items()]
    comparison = save_comparison_table(results, output_dir)

    plot_forecast_comparison(windowed["target_datetime_test"], y_test, preds, output_dir)
    save_artifacts_for_shap(
        rf_model, windowed["tree_X_test"], y_test, preds["random_forest"],
        windowed["target_datetime_test"], output_dir,
    )
    return comparison


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Train and compare an LSTM, a lag-feature Random Forest, and a naive "
                     "persistence baseline for next-hour traffic volume forecasting."
    )
    parser.add_argument("--input", type=Path, default=here.parent / "pipeline" / "traffic_features.csv",
                         help="Path to Part 2's feature-engineered CSV.")
    parser.add_argument("--hidden-size", type=int, default=64, help="LSTM hidden state size (default: 64).")
    parser.add_argument("--num-layers", type=int, default=1, help="Number of stacked LSTM layers (default: 1).")
    parser.add_argument("--batch-size", type=int, default=128, help="Training batch size (default: 128).")
    parser.add_argument("--lr", type=float, default=1e-3, help="Adam learning rate (default: 0.001).")
    parser.add_argument("--epochs", type=int, default=40, help="Maximum training epochs (default: 40).")
    parser.add_argument("--patience", type=int, default=6,
                         help="Stop early after this many epochs with no validation-loss improvement (default: 6).")
    parser.add_argument("--output-dir", type=Path, default=here / "artifacts",
                         help="Directory to write comparison table, charts, and saved models.")
    parser.add_argument("--log-file", type=Path, default=here / "lstm_forecast.log",
                         help="Path to write the log file.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable DEBUG-level logging.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging(args.log_file, verbose=args.verbose)
    logger.info("Command invoked: lstm_forecast with arguments: %s", vars(args))

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
            args.input, args.hidden_size, args.num_layers, args.batch_size, args.lr,
            args.epochs, args.patience, args.output_dir,
        )
    except UserInputError as e:
        logger.error("Invalid input: %s", e)
        return 1
    except (FileNotFoundError, pd.errors.EmptyDataError, pd.errors.ParserError,
            OSError, UnicodeDecodeError):
        logger.error("LSTM forecasting aborted: could not load the dataset", exc_info=True)
        return 1
    except Exception:
        logger.error("LSTM forecasting aborted due to an unexpected error", exc_info=True)
        return 1

    logger.info(
        "LSTM forecasting completed successfully -- see %s for the full model comparison table",
        args.output_dir / "lstm_regression_comparison.csv",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
