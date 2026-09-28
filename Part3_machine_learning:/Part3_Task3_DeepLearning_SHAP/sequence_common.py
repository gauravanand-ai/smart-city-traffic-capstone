#!/usr/bin/env python3
"""
sequence_common.py
Shared code for Part 3, Task 3 (LSTM forecasting + SHAP explainability):
building strictly-contiguous hourly windows from traffic_features.csv, and
deriving two *aligned* views of the same windows -- a 3D sequence array for
the LSTM, and a hand-engineered lag/rolling-feature table for the comparable
tree-based model used for SHAP.

Kept separate from ml_common.py (used by Tasks 1-2) because this task's data
representation (fixed-length hourly sequences, not one-row-per-hour) is
genuinely different, and because it deliberately has a leaner dependency
footprint (no torch/shap import here -- those stay in lstm_forecast.py /
explain_shap.py so Tasks 1-2's scripts never need them installed).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

WINDOW = 24  # hours of history used to forecast the next hour

# Per-timestep features the LSTM sees across the whole window. traffic_volume
# itself is included (scaled) -- this is a forecasting problem, so using past
# traffic_volume to predict a *future* traffic_volume is the entire point,
# not the leakage Task 1 had to guard against there.
SEQUENCE_FEATURES = [
    "traffic_volume_scaled", "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    "is_weekend", "temp_zscore", "is_precipitating", "is_severe_weather",
    "is_low_visibility", "holiday_flag",
]

# Hand-engineered features for the tree-based comparison model: lag/rolling
# summaries of the SAME window the LSTM sees, describing the same underlying
# signals (traffic history, calendar position, weather) in the form a
# non-sequential model needs -- not literally the same 24x11 tensor, since
# the whole point of the comparison is that the LSTM learns temporal
# structure directly while the tree model needs it engineered by hand.
#
# Split in two so build_windowed_dataset() can read each half from the right
# place: LAG_FEATURES (lag_1/lag_24/roll_mean_6/roll_mean_24) are already
# shifted by construction (add_lag_features uses .shift(1)/.shift(WINDOW)),
# so reading them at the target row still only reflects hours up to the last
# *observed* hour. ASOF_FEATURES are plain columns with no built-in shift, so
# they must be read from the window's last row (the last observed hour, "hour
# t") rather than the target row (the forecast hour, "hour t+1") -- otherwise
# the tree model would see the true future weather at t+1, which isn't
# knowable in reality and which the LSTM never sees either, making the
# comparison unfair.
LAG_FEATURES = ["lag_1", "lag_24", "roll_mean_6", "roll_mean_24"]
ASOF_FEATURES = [
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_weekend", "holiday_flag",
    "temp_zscore", "is_precipitating", "is_severe_weather", "is_low_visibility",
]
TREE_FEATURES = LAG_FEATURES + ASOF_FEATURES


def add_contiguous_blocks(df: pd.DataFrame) -> pd.DataFrame:
    """Marks runs of truly-consecutive hourly readings (gap <= 1h) with a
    block_id. The raw dataset has real coverage gaps (Part 1 found whole
    months missing in some years) -- a naive 'row N back' window would
    silently splice unrelated time periods together whenever a gap falls
    inside it, so windows are only built within a block, never across one."""
    df = df.sort_values("date_time").reset_index(drop=True).copy()
    gap_hours = df["date_time"].diff().dt.total_seconds().div(3600)
    df["block_id"] = (gap_hours.fillna(1) > 1).cumsum()

    block_sizes = df["block_id"].value_counts()
    n_usable_blocks = int((block_sizes > WINDOW).sum())
    logger.info(
        "Split %d rows into %d contiguous hourly block(s) (gap > 1h starts a new block); "
        "%d block(s) are long enough (> %d hours) to contribute at least one window",
        len(df), df["block_id"].nunique(), n_usable_blocks, WINDOW,
    )
    return df


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """lag_1/lag_24/rolling means computed WITHIN each contiguous block only
    (groupby block_id), so a lag never reaches back across a real time gap
    either."""
    df = df.copy()
    grouped = df.groupby("block_id")["traffic_volume"]
    df["lag_1"] = grouped.shift(1)
    df["lag_24"] = grouped.shift(WINDOW)
    df["roll_mean_6"] = grouped.transform(lambda s: s.shift(1).rolling(6).mean())
    df["roll_mean_24"] = grouped.transform(lambda s: s.shift(1).rolling(WINDOW).mean())
    return df


def build_windowed_dataset(
    df: pd.DataFrame, train_frac: float = 0.7, val_frac: float = 0.1,
) -> dict:
    """Builds every valid (window -> next-hour target) sample from strictly
    contiguous blocks, then splits chronologically into train/val/test
    (val is for the LSTM's early stopping; test is held out for both models).
    traffic_volume is scaled using TRAIN-set mean/std only, to avoid leaking
    test-period statistics into the scaling.

    Returns a dict with:
        seq_X_{train,val,test}: (n, WINDOW, len(SEQUENCE_FEATURES)) float32 arrays
        tree_X_{train,val,test}: DataFrames with TREE_FEATURES columns
        y_{train,val,test}: raw (unscaled) traffic_volume targets, float32
        target_datetime_{train,val,test}: for reference/plotting
        scaler_mean, scaler_std: the traffic_volume scaling parameters (fit on train)
    """
    df = add_contiguous_blocks(df)
    df = add_lag_features(df)

    samples = []  # list of (target_row_idx, block_rows_start, block_rows_end)
    for block_id, block in df.groupby("block_id"):
        block = block.reset_index()  # keep original index as a column
        if len(block) <= WINDOW:
            continue
        for target_pos in range(WINDOW, len(block)):
            window_positions = range(target_pos - WINDOW, target_pos)
            samples.append((block.loc[target_pos, "index"], block.loc[list(window_positions), "index"].tolist()))

    logger.info("Built %d valid window(s) (window=%dh) from contiguous blocks", len(samples), WINDOW)

    samples.sort(key=lambda s: df.loc[s[0], "date_time"])
    n = len(samples)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    splits = {
        "train": samples[:n_train],
        "val": samples[n_train:n_train + n_val],
        "test": samples[n_train + n_val:],
    }
    for name, s in splits.items():
        if s:
            logger.info(
                "%s split: %d samples (%s to %s)",
                name, len(s), df.loc[s[0][0], "date_time"], df.loc[s[-1][0], "date_time"],
            )

    train_volumes = df.loc[[s[0] for s in splits["train"]], "traffic_volume"]
    # Use the volumes actually IN the training windows (not just targets) for scaling stats
    train_window_idx = sorted({idx for s in splits["train"] for idx in s[1]} | {s[0] for s in splits["train"]})
    scaler_mean = df.loc[train_window_idx, "traffic_volume"].mean()
    scaler_std = df.loc[train_window_idx, "traffic_volume"].std()
    logger.info("traffic_volume scaling (fit on train only): mean=%.1f, std=%.1f", scaler_mean, scaler_std)

    df["traffic_volume_scaled"] = (df["traffic_volume"] - scaler_mean) / scaler_std

    result = {"scaler_mean": scaler_mean, "scaler_std": scaler_std}
    for name, s in splits.items():
        if not s:
            result[f"seq_X_{name}"] = np.zeros((0, WINDOW, len(SEQUENCE_FEATURES)), dtype=np.float32)
            result[f"tree_X_{name}"] = pd.DataFrame(columns=TREE_FEATURES)
            result[f"y_{name}"] = np.zeros((0,), dtype=np.float32)
            result[f"target_datetime_{name}"] = pd.Series(dtype="datetime64[ns]")
            continue

        target_idx = [t for t, _ in s]
        # The last hour actually observed before the target (the final row of
        # each window) -- calendar/weather TREE_FEATURES are read from here,
        # not from the target row itself, so the tree model sees exactly the
        # same "as of hour t" information the LSTM's last timestep sees. Using
        # the target row's own weather would leak the true future weather
        # (not knowable in reality) into the tree model but not the LSTM,
        # breaking the fair comparison between the two.
        last_observed_idx = [window_idx[-1] for _, window_idx in s]
        seq_arr = np.stack([
            df.loc[window_idx, SEQUENCE_FEATURES].to_numpy(dtype=np.float32)
            for _, window_idx in s
        ])
        result[f"seq_X_{name}"] = seq_arr

        lag_df = df.loc[target_idx, LAG_FEATURES].reset_index(drop=True)
        asof_df = df.loc[last_observed_idx, ASOF_FEATURES].reset_index(drop=True)
        result[f"tree_X_{name}"] = pd.concat([lag_df, asof_df], axis=1)[TREE_FEATURES]

        result[f"y_{name}"] = df.loc[target_idx, "traffic_volume"].to_numpy(dtype=np.float32)
        result[f"target_datetime_{name}"] = df.loc[target_idx, "date_time"].reset_index(drop=True)

    return result
