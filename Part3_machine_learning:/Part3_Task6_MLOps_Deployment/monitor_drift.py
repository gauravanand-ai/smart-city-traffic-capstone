#!/usr/bin/env python3
"""
monitor_drift.py
Part 3, Task 6.4/6.5 (MLOps: monitoring simulation + alerting).

Simulates what a production monitoring job would do periodically: compare
whatever traffic/weather data has come in against the statistics the
deployed model was trained on, flag both FEATURE distribution drift and
PREDICTION ERROR drift, and roll everything up into one PASS/ALERT status
per monitored window plus a small dashboard.

No live production traffic exists for this capstone, so two monitoring
windows are simulated against the same REFERENCE distribution (the
classifier's chronological training split -- what the deployed model was
actually fit on, and the actual boundary Task 1/6.1 already established):

    test_period       -- the model's own chronological TEST split (the most
                          recent 20% of real historical readings). This is
                          the most honest "what would monitoring have seen
                          right after deployment" simulation available
                          without live data: genuine, unperturbed history.
    synthetic_severe_winter
                       -- a synthetically perturbed batch (temp shifted down,
                          clouds/weather skewed toward Severe/Low-visibility)
                          standing in for a genuinely out-of-distribution
                          month, so the ALERT branch of this mechanism is
                          demonstrated on purpose, not left untested because
                          real drift happened to be mild. Clearly labeled
                          synthetic throughout -- see RESULTS_TASK6.md.

For each window, two kinds of drift are checked against the reference:

    Feature distribution drift
        - Kolmogorov-Smirnov test (scipy.stats.ks_2samp) on two continuous
          raw features (temp, clouds_all): p < 0.05 -> ALERT.
        - Population Stability Index (PSI) on two categorical-ish
          distributions (hour-of-day, weather_main): PSI > 0.25 -> ALERT,
          0.1-0.25 -> flagged "moderate" but not alerting alone (standard
          PSI convention), < 0.1 -> stable.
    Prediction error drift
        - The deployed classifier (Task 6.1's registry "champion" alias --
          the same model deploy_api.py serves) is scored on each window;
          F1 dropping more than 0.05 versus its OWN reference-window F1 ->
          ALERT.
        - Task 1's regression Random Forest is scored the same way; MAE
          rising more than 25% versus its reference-window MAE -> ALERT.

A window's overall status is ALERT if ANY individual check alerts, else
PASS -- the simple mechanism the brief asks for in 6.5.

Usage:
    python monitor_drift.py [--input PATH] [--tracking-uri URI]
                             [--registered-name NAME] [--alias champion]
                             [--output-dir PATH] [--log-file PATH] [-v]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

os.environ.setdefault("MLFLOW_DISABLE_TELEMETRY", "1")
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from scipy.stats import ks_2samp
from sklearn.metrics import f1_score, mean_absolute_error

from ml_common import (
    COMMON_FEATURE_COLUMNS, UserInputError, chronological_split,
    configure_logging, get_feature_matrix, load_dataset, prepare_ml_dataset,
)
from serving_common import WEATHER_CATEGORIES

logger = logging.getLogger(__name__)

CLASSIFICATION_TARGET = "high_risk"
REGRESSION_TARGET = "traffic_volume"

# Status palette -- reserved for PASS/ALERT only, never reused for the
# categorical (window/feature) bars elsewhere in the dashboard, per this
# project's established red/green/blue convention (route_recommender.py,
# model_registry.py).
COLOR_PASS = "#1baf7a"    # green
COLOR_ALERT = "#e34948"   # red
COLOR_MODERATE = "#e0a72e"  # amber -- PSI's "moderate" middle band only
COLOR_REFERENCE = "#9a9a9a"  # neutral gray -- the baseline itself, not a status
COLOR_WINDOW_A = "#2a78d6"   # blue
COLOR_WINDOW_B = "#eb6834"   # orange

# Thresholds -- documented, conventional choices (PSI's 0.1/0.25 bands are
# the standard industry convention; the others are this project's own
# reasoned choices, explained in RESULTS_TASK6.md).
#
# KS_ALERT_P alone is NOT used as the sole trigger: with ~32k reference rows,
# the KS test's p-value is so sensitive that even a small, expected amount of
# real drift (e.g. temp's normal year-to-year seasonal variation between the
# training years and the held-out test year) comes back astronomically
# significant (p ~1e-60) despite a small effect size. This was caught by
# actually running this script against the real chronological test split
# (not just the synthetic one) and seeing it ALERT on ordinary historical
# data -- see RESULTS_TASK6.md. KS_ALERT_STAT (the KS statistic itself, a
# genuine effect-size measure bounded [0, 1] and NOT sample-size-sensitive)
# is required ALONGSIDE the p-value, so a real but modest seasonal shift
# reads as PASS while a genuinely severe distributional change still ALERTs.
KS_ALERT_P = 0.05
KS_ALERT_STAT = 0.15
PSI_ALERT = 0.25
PSI_MODERATE = 0.10
F1_DROP_ALERT = 0.05          # absolute F1 points
MAE_INCREASE_ALERT_PCT = 0.25  # relative increase


def population_stability_index(reference: pd.Series, current: pd.Series, bins: int | list) -> float:
    """Standard PSI: sum((cur% - ref%) * ln(cur% / ref%)) over shared bins.
    A small epsilon avoids log(0)/division-by-zero when a bin is empty in
    one distribution -- a real risk with a 24-hour or 11-category bin set
    on a modestly sized synthetic window."""
    if isinstance(bins, int):
        edges = np.histogram_bin_edges(pd.concat([reference, current]), bins=bins)
        ref_counts, _ = np.histogram(reference, bins=edges)
        cur_counts, _ = np.histogram(current, bins=edges)
    else:
        edges = bins
        ref_counts = reference.value_counts().reindex(edges, fill_value=0).values
        cur_counts = current.value_counts().reindex(edges, fill_value=0).values

    eps = 1e-4
    ref_pct = ref_counts / ref_counts.sum() + eps
    cur_pct = cur_counts / cur_counts.sum() + eps
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def make_synthetic_severe_winter(reference: pd.DataFrame, n: int, rng: np.random.Generator) -> pd.DataFrame:
    """A synthetic 'severe winter month' batch: resamples real reference
    rows (so every OTHER feature stays internally consistent -- cyclical
    encodings, one-hot weather, etc. are never hand-edited into invalid
    combinations) then overwrites temp/clouds_all/weather_main to simulate
    an unusually cold, overcast, low-visibility spell -- the kind of
    genuinely out-of-distribution period this mechanism should catch."""
    sample = reference.sample(n=n, replace=True, random_state=int(rng.integers(0, 2**31))).copy()
    sample["temp"] = sample["temp"] - 18.0 + rng.normal(0, 2.0, size=n)  # ~18K colder, noisy
    sample["clouds_all"] = np.clip(sample["clouds_all"] + 35 + rng.normal(0, 5, size=n), 0, 100)

    # Skew weather_main heavily toward severe/low-visibility categories
    # instead of the reference's own mix.
    severe_weather = rng.choice(
        ["snow", "fog", "mist", "thunderstorm", "clouds"], size=n,
        p=[0.35, 0.25, 0.15, 0.10, 0.15],
    )
    sample["weather_main"] = severe_weather
    # Only the known one-hot dummy columns -- NOT "weather_main" itself,
    # which also (confusingly) starts with "weather_" and would otherwise
    # get matched and overwritten by this same loop (caught by actually
    # running this function and finding weather_main silently turned into
    # an all-zero int column; see RESULTS_TASK6.md).
    for category in WEATHER_CATEGORIES:
        sample[f"weather_{category}"] = (sample["weather_main"] == category).astype(int)
    sample["is_severe_weather"] = sample["weather_main"].isin({"thunderstorm", "squall"}).astype(int)
    sample["is_low_visibility"] = sample["weather_main"].isin({"fog", "mist", "haze", "smoke"}).astype(int)
    sample["is_precipitating"] = sample["weather_main"].isin({"drizzle", "rain", "snow"}).astype(int)
    sample["temp_zscore"] = (sample["temp"] - reference["temp"].mean()) / reference["temp"].std()
    clouds_min, clouds_max = reference["clouds_all"].min(), reference["clouds_all"].max()
    sample["clouds_all_minmax"] = (sample["clouds_all"] - clouds_min) / (clouds_max - clouds_min)

    return sample.reset_index(drop=True)


def load_champion_classifier(tracking_uri: str, registered_name: str, alias: str):
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()
    try:
        mv = client.get_model_version_by_alias(registered_name, alias)
    except MlflowException as e:
        raise UserInputError(
            f"No model version aliased '{alias}' for registered model '{registered_name}' -- "
            f"run model_registry.py (Task 6.1) first."
        ) from e
    model = mlflow.sklearn.load_model(f"models:/{registered_name}@{alias}")
    logger.info("Loaded champion classifier for monitoring: %s@%s (registry v%s)", registered_name, alias, mv.version)
    return model


def load_regressor(path: Path):
    if not path.exists():
        raise UserInputError(
            f"Trained regressor not found at {path}. Run train_regression.py (Part 3 Task 1) first."
        )
    return joblib.load(path)


def check_feature_drift(reference: pd.DataFrame, current: pd.DataFrame) -> list[dict]:
    rows = []

    for feature in ["temp", "clouds_all"]:
        stat, p_value = ks_2samp(reference[feature], current[feature])
        status = "ALERT" if (p_value < KS_ALERT_P and stat > KS_ALERT_STAT) else "PASS"
        # value = the KS statistic itself (the effect-size number the ALERT
        # decision actually gates on, per the KS_ALERT_STAT comment above) --
        # NOT the p-value, which is kept in extra/p_value for reference but
        # is close to uninformative at this sample size (see comment above
        # KS_ALERT_STAT).
        rows.append({
            "check": f"ks_{feature}", "kind": "feature_drift", "value": stat, "reference_value": 0.0,
            "threshold": f"ks_stat > {KS_ALERT_STAT} (and p < {KS_ALERT_P})",
            "extra": f"p_value={p_value:.3e}", "status": status,
        })

    psi_hour = population_stability_index(reference["hour"], current["hour"], bins=list(range(25)))
    rows.append({
        "check": "psi_hour_of_day", "kind": "feature_drift", "value": psi_hour, "reference_value": 0.0,
        "threshold": f"> {PSI_ALERT}", "extra": "", "status": _psi_status(psi_hour),
    })

    weather_categories = sorted(set(reference["weather_main"]) | set(current["weather_main"]))
    psi_weather = population_stability_index(reference["weather_main"], current["weather_main"], bins=weather_categories)
    rows.append({
        "check": "psi_weather_main", "kind": "feature_drift", "value": psi_weather, "reference_value": 0.0,
        "threshold": f"> {PSI_ALERT}", "extra": "", "status": _psi_status(psi_weather),
    })
    return rows


def _psi_status(psi: float) -> str:
    if psi > PSI_ALERT:
        return "ALERT"
    if psi > PSI_MODERATE:
        return "MODERATE"
    return "PASS"


def check_prediction_error_drift(
    reference: pd.DataFrame, current: pd.DataFrame, classifier, regressor,
) -> list[dict]:
    rows = []

    X_ref, y_ref = get_feature_matrix(reference), reference[CLASSIFICATION_TARGET]
    X_cur, y_cur = get_feature_matrix(current), current[CLASSIFICATION_TARGET]
    f1_ref = f1_score(y_ref, classifier.predict(X_ref), zero_division=0)
    f1_cur = f1_score(y_cur, classifier.predict(X_cur), zero_division=0)
    f1_drop = f1_ref - f1_cur
    rows.append({
        "check": "classifier_f1_drift", "kind": "prediction_error_drift", "value": f1_cur, "reference_value": f1_ref,
        "threshold": f"reference F1={f1_ref:.3f}, drop > {F1_DROP_ALERT}",
        "extra": f"drop={f1_drop:.3f}", "status": "ALERT" if f1_drop > F1_DROP_ALERT else "PASS",
    })

    y_ref_vol, y_cur_vol = reference[REGRESSION_TARGET], current[REGRESSION_TARGET]
    mae_ref = mean_absolute_error(y_ref_vol, regressor.predict(X_ref))
    mae_cur = mean_absolute_error(y_cur_vol, regressor.predict(X_cur))
    mae_increase_pct = (mae_cur - mae_ref) / mae_ref if mae_ref else 0.0
    rows.append({
        "check": "regressor_mae_drift", "kind": "prediction_error_drift", "value": mae_cur, "reference_value": mae_ref,
        "threshold": f"reference MAE={mae_ref:.1f}, increase > {MAE_INCREASE_ALERT_PCT:.0%}",
        "extra": f"increase={mae_increase_pct:.1%}",
        "status": "ALERT" if mae_increase_pct > MAE_INCREASE_ALERT_PCT else "PASS",
    })
    return rows


def plot_dashboard(report: pd.DataFrame, window_status: dict, output_dir: Path) -> Path:
    """A 2x3 grid, one measure per panel -- deliberately NOT a dual-axis
    chart for the prediction-error panels (classifier F1 and regressor MAE
    are different units/scales, so each gets its own single-axis panel
    rather than sharing one plot with two y-axes)."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    windows = list(window_status.keys())
    window_colors = {windows[0]: COLOR_WINDOW_A, windows[1]: COLOR_WINDOW_B} if len(windows) >= 2 else {windows[0]: COLOR_WINDOW_A}
    status_color = {"PASS": COLOR_PASS, "ALERT": COLOR_ALERT}
    width = 0.35

    # Panel 1: overall status -- the actual 6.5 "PASS / ALERT" answer.
    ax = axes[0, 0]
    ax.axis("off")
    ax.set_title("Overall Monitoring Status", fontsize=12, loc="left")
    for i, window in enumerate(windows):
        y = 0.75 - i * 0.35
        status = window_status[window]
        ax.add_patch(plt.Rectangle((0.02, y - 0.1), 0.3, 0.2, color=status_color[status], transform=ax.transAxes))
        ax.text(0.17, y, status, ha="center", va="center", color="white", fontsize=13, fontweight="bold", transform=ax.transAxes)
        ax.text(0.38, y, window.replace("_", " "), ha="left", va="center", fontsize=11, transform=ax.transAxes)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # Panel 2: PSI by feature/window, with the alert/moderate threshold lines.
    ax = axes[0, 1]
    psi_rows = report[report["check"].str.startswith("psi_")]
    features = psi_rows["check"].unique()
    x = np.arange(len(features))
    for i, window in enumerate(windows):
        vals = [psi_rows[(psi_rows["window"] == window) & (psi_rows["check"] == f)]["value"].iloc[0] for f in features]
        ax.bar(x + (i - 0.5) * width, vals, width=width, color=window_colors[window], label=window.replace("_", " "))
    ax.axhline(PSI_ALERT, color=COLOR_ALERT, linestyle="--", linewidth=1.5, label=f"ALERT (PSI > {PSI_ALERT})")
    ax.axhline(PSI_MODERATE, color=COLOR_MODERATE, linestyle="--", linewidth=1.5, label=f"Moderate (PSI > {PSI_MODERATE})")
    ax.set_xticks(x)
    ax.set_xticklabels([f.replace("psi_", "") for f in features], rotation=15, ha="right")
    ax.set_ylabel("PSI")
    ax.set_title("Feature Drift: PSI", fontsize=12, loc="left")
    ax.legend(loc="upper left", frameon=False, fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Panel 3: KS statistic (effect size, bounded [0,1]) by feature/window --
    # this, not the p-value, is what the ALERT decision actually gates on
    # (see the KS_ALERT_STAT comment above); plotting the statistic directly
    # makes the chart show the same thing the threshold check uses.
    ax = axes[0, 2]
    ks_rows = report[report["check"].str.startswith("ks_")]
    features = ks_rows["check"].unique()
    x = np.arange(len(features))
    for i, window in enumerate(windows):
        vals = [ks_rows[(ks_rows["window"] == window) & (ks_rows["check"] == f)]["value"].iloc[0] for f in features]
        ax.bar(x + (i - 0.5) * width, vals, width=width, color=window_colors[window], label=window.replace("_", " "))
    ax.axhline(KS_ALERT_STAT, color=COLOR_ALERT, linestyle="--", linewidth=1.5, label=f"ALERT (ks_stat > {KS_ALERT_STAT})")
    ax.set_xticks(x)
    ax.set_xticklabels([f.replace("ks_", "") for f in features])
    ax.set_ylim(0, 1)
    ax.set_ylabel("KS statistic (effect size)")
    ax.set_title("Feature Drift: KS Test", fontsize=12, loc="left")
    ax.legend(loc="upper left", frameon=False, fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Panel 4: classifier F1, reference vs each window -- its own single-axis
    # panel (not merged with MAE, which is a different unit entirely).
    ax = axes[1, 0]
    err_rows = report[report["kind"] == "prediction_error_drift"]
    f1_rows = err_rows[err_rows["check"] == "classifier_f1_drift"]
    f1_ref = f1_rows["reference_value"].iloc[0]
    labels = ["reference"] + [w.replace("_", " ") for w in windows]
    vals = [f1_ref] + [f1_rows[f1_rows["window"] == w]["value"].iloc[0] for w in windows]
    colors = [COLOR_REFERENCE] + [window_colors[w] for w in windows]
    ax.bar(labels, vals, color=colors)
    ax.axhline(f1_ref - F1_DROP_ALERT, color=COLOR_ALERT, linestyle="--", linewidth=1.5,
               label=f"ALERT (drop > {F1_DROP_ALERT})")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Classifier F1 (high_risk)")
    ax.set_title("Prediction Error: Classifier F1", fontsize=12, loc="left")
    ax.legend(loc="lower left", frameon=False, fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Panel 5: regressor MAE, reference vs each window -- own panel, own units.
    ax = axes[1, 1]
    mae_rows = err_rows[err_rows["check"] == "regressor_mae_drift"]
    mae_ref = mae_rows["reference_value"].iloc[0]
    vals = [mae_ref] + [mae_rows[mae_rows["window"] == w]["value"].iloc[0] for w in windows]
    ax.bar(labels, vals, color=colors)
    ax.axhline(mae_ref * (1 + MAE_INCREASE_ALERT_PCT), color=COLOR_ALERT, linestyle="--", linewidth=1.5,
               label=f"ALERT (increase > {MAE_INCREASE_ALERT_PCT:.0%})")
    ax.set_ylabel("Regressor MAE (veh/hr)")
    ax.set_title("Prediction Error: Regressor MAE", fontsize=12, loc="left")
    ax.legend(loc="upper left", frameon=False, fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Panel 6: plain-language summary of exactly which checks triggered each
    # window's status -- complements the charts with the "why" in words.
    ax = axes[1, 2]
    ax.axis("off")
    ax.set_title("Alert Detail", fontsize=12, loc="left")
    y = 0.92
    for window in windows:
        w_rows = report[report["window"] == window]
        alerts = w_rows[w_rows["status"] == "ALERT"]["check"].tolist()
        ax.text(0.0, y, window.replace("_", " ") + ":", fontsize=10, fontweight="bold", transform=ax.transAxes)
        y -= 0.09
        if alerts:
            for a in alerts:
                ax.text(0.05, y, f"- {a}", fontsize=9, color=COLOR_ALERT, transform=ax.transAxes)
                y -= 0.08
        else:
            ax.text(0.05, y, "- no checks triggered", fontsize=9, color=COLOR_PASS, transform=ax.transAxes)
            y -= 0.08
        y -= 0.05
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    fig.suptitle("Model Monitoring Dashboard -- Feature & Prediction-Error Drift", fontsize=14, y=0.995)
    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "drift_monitoring_dashboard.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Saved figure: %s", path)
    return path


def run(
    input_path: Path, tracking_uri: str, registered_name: str, alias: str,
    regressor_path: Path, output_dir: Path, seed: int,
) -> pd.DataFrame:
    df = load_dataset(input_path)
    df = prepare_ml_dataset(df)
    train_df, test_df = chronological_split(df)
    reference = train_df

    rng = np.random.default_rng(seed)
    synthetic = make_synthetic_severe_winter(reference, n=len(test_df), rng=rng)

    classifier = load_champion_classifier(tracking_uri, registered_name, alias)
    regressor = load_regressor(regressor_path)

    windows = {"test_period": test_df, "synthetic_severe_winter": synthetic}
    all_rows = []
    window_status = {}

    for window_name, current in windows.items():
        feature_rows = check_feature_drift(reference, current)
        error_rows = check_prediction_error_drift(reference, current, classifier, regressor)
        for row in feature_rows + error_rows:
            row["window"] = window_name
            all_rows.append(row)

        statuses = {row["status"] for row in feature_rows + error_rows}
        overall = "ALERT" if "ALERT" in statuses else "PASS"
        window_status[window_name] = overall
        logger.info(
            "Window '%s': overall status = %s (%d check(s), %d alert(s), %d moderate)",
            window_name, overall, len(feature_rows) + len(error_rows),
            sum(1 for r in feature_rows + error_rows if r["status"] == "ALERT"),
            sum(1 for r in feature_rows + error_rows if r["status"] == "MODERATE"),
        )
        if overall == "ALERT":
            alerting_checks = [r["check"] for r in feature_rows + error_rows if r["status"] == "ALERT"]
            logger.warning("Window '%s' ALERT triggered by: %s", window_name, ", ".join(alerting_checks))

    report = pd.DataFrame(all_rows)[["window", "kind", "check", "value", "reference_value", "threshold", "extra", "status"]]
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "drift_monitoring_report.csv"
    report.to_csv(report_path, index=False)
    logger.info("Wrote drift monitoring report to %s", report_path)

    summary = {
        "reference_window": {
            "start": str(reference["date_time"].min()), "end": str(reference["date_time"].max()), "n_rows": len(reference),
        },
        "windows": {
            name: {"overall_status": status, "n_rows": len(windows[name])}
            for name, status in window_status.items()
        },
    }
    summary_path = output_dir / "drift_monitoring_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    logger.info("Wrote drift monitoring summary to %s", summary_path)

    plot_dashboard(report, window_status, output_dir)
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Simulate feature/prediction-error drift monitoring and PASS/ALERT status reporting."
    )
    parser.add_argument("--input", type=Path, default=here.parent / "pipeline" / "traffic_features.csv",
                         help="Path to Part 2's feature-engineered CSV.")
    parser.add_argument("--tracking-uri", type=str, default=f"sqlite:///{here}/mlflow.db",
                         help="MLflow tracking store URI (default: local SQLite, same store Task 6.1 uses).")
    parser.add_argument("--registered-name", type=str, default="traffic-risk-classifier",
                         help="Registered model name to monitor (default: traffic-risk-classifier).")
    parser.add_argument("--alias", type=str, default="champion",
                         help="Registry alias to monitor (default: champion -- the version deploy_api.py serves).")
    parser.add_argument("--regressor-path", type=Path, default=here / "artifacts" / "regression_random_forest.joblib",
                         help="Path to Task 1's trained regression model.")
    parser.add_argument("--output-dir", type=Path, default=here / "artifacts",
                         help="Directory to write the report/summary/dashboard.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for the synthetic drifted batch.")
    parser.add_argument("--log-file", type=Path, default=here / "monitor_drift.log",
                         help="Path to write the log file.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable DEBUG-level logging.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging(args.log_file, verbose=args.verbose)
    logger.info("Command invoked: monitor_drift with arguments: %s", vars(args))

    try:
        run(
            args.input, args.tracking_uri, args.registered_name, args.alias,
            args.regressor_path, args.output_dir, args.seed,
        )
    except UserInputError as e:
        logger.error("Invalid input: %s", e)
        return 1
    except (FileNotFoundError, pd.errors.EmptyDataError, pd.errors.ParserError,
            OSError, UnicodeDecodeError):
        logger.error("Drift monitoring aborted: could not load the dataset", exc_info=True)
        return 1
    except Exception:
        logger.error("Drift monitoring aborted due to an unexpected error", exc_info=True)
        return 1

    logger.info(
        "Drift monitoring completed successfully -- see %s for the full report",
        args.output_dir / "drift_monitoring_report.csv",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
