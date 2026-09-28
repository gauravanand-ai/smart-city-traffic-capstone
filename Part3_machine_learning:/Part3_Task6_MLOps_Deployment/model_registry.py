#!/usr/bin/env python3
"""
model_registry.py
Part 3, Task 6.1/6.2 (MLOps: model versioning + MLflow experiment tracking).

This script demonstrates genuine model versioning, not just a written
description of it: it trains TWO real, differently-configured versions of
the high_risk Random Forest classifier already built in Task 1, tracks both
with MLflow (params/metrics/model artifact -- exactly Task 4's tracking
approach, reused here rather than reinvented), and then REGISTERS both into
MLflow's Model Registry under one named model ("traffic-risk-classifier"),
so each gets a real, queryable version number:

    v1 "baseline"  -- Task 1's exact original configuration (150 trees,
                       depth 10, min_samples_leaf 5). Reproduced here (not
                       re-imported from train_classification.py's saved
                       .joblib) so both versions are trained, evaluated, and
                       logged through the identical code path in this
                       script -- an apples-to-apples versioning comparison.
    v2 "candidate" -- a deliberately different hyperparameter configuration
                       (300 trees, depth 16, min_samples_leaf 2) -- a
                       genuine second version, not a relabeled copy of v1.

Whichever version scores higher F1 on the same chronological test split is
aliased "champion" in the registry; v1 always keeps the "baseline" alias.
deploy_api.py (Task 6.3) loads whichever version is currently aliased
"champion" -- i.e. the deployment API is wired to the registry, not to a
hardcoded file path, which is the actual point of a model registry in a real
MLOps pipeline: promoting a new version to production is a registry alias
change, not a code change.

Re-running this script is idempotent: it deletes any previously registered
"traffic-risk-classifier" model (and its versions) first, so version numbers
stay meaningful (v1/v2) across repeated runs rather than climbing forever.
The underlying MLflow *runs* are never deleted (registry deletion only
un-registers a version; it does not touch the run) -- so the run history in
mlflow_runs_leaderboard.csv from Task 4 keeps growing across script re-runs
by design, and only the registry's own version list is reset here.

Usage:
    python model_registry.py [--input PATH] [--tracking-uri URI]
                              [--experiment NAME] [--registered-name NAME]
                              [--output-dir PATH] [--log-file PATH] [-v]
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

# Same rationale as mlflow_tracking.py (Task 4): keep this run fully local
# and silence the unrelated LLM-tracing startup hint. Set before importing
# mlflow.
os.environ.setdefault("MLFLOW_DISABLE_TELEMETRY", "1")
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from mlflow.models import infer_signature
from sklearn.ensemble import RandomForestClassifier

from ml_common import (
    COMMON_FEATURE_COLUMNS, UserInputError, chronological_split,
    configure_logging, get_feature_matrix, load_dataset, prepare_ml_dataset,
)
from train_classification import evaluate_model

logger = logging.getLogger(__name__)

TARGET_COLUMN = "high_risk"
REGISTERED_MODEL_NAME_DEFAULT = "traffic-risk-classifier"

COLOR_BASELINE = "#2a78d6"   # blue -- consistent with Task 1/4's palette
COLOR_CANDIDATE = "#eb6834"  # orange

# Two genuinely different configurations -- not the same model relabeled.
# v1 reproduces Task 1's exact Random Forest. v2 is a real candidate: more
# trees, deeper, smaller leaves -- a plausible "someone tried to improve it"
# retraining, not a strawman.
MODEL_VERSIONS = {
    "v1_baseline": dict(n_estimators=150, max_depth=10, min_samples_leaf=5),
    "v2_candidate": dict(n_estimators=300, max_depth=16, min_samples_leaf=2),
}


def build_versions() -> dict[str, RandomForestClassifier]:
    return {
        name: RandomForestClassifier(
            **params, class_weight="balanced", random_state=42, n_jobs=-1,
        )
        for name, params in MODEL_VERSIONS.items()
    }


def reset_registered_model(client: MlflowClient, name: str) -> None:
    """Deletes any existing registered model of this name (and all its
    versions) so re-running this script produces a clean v1/v2 pair instead
    of version numbers climbing indefinitely. Does not touch the underlying
    MLflow runs -- only the registry's pointer to them."""
    try:
        client.delete_registered_model(name)
        logger.info("Removed existing registered model '%s' before re-registering", name)
    except MlflowException:
        logger.debug("No existing registered model named '%s' to remove", name)


def plot_version_comparison(rows: list[dict], output_dir: Path) -> Path:
    metrics = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
    labels = {"v1_baseline": "v1 (baseline)", "v2_candidate": "v2 (candidate)"}
    colors = {"v1_baseline": COLOR_BASELINE, "v2_candidate": COLOR_CANDIDATE}

    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = range(len(metrics))
    width = 0.35
    for i, row in enumerate(rows):
        offsets = [xi + (i - 0.5) * width for xi in x]
        values = [row[m] for m in metrics]
        bars = ax.bar(offsets, values, width=width, color=colors[row["version"]],
                       label=labels[row["version"]])
        for b, v in zip(bars, values):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=8)

    ax.set_xticks(list(x))
    ax.set_xticklabels(["Accuracy", "Precision", "Recall", "F1", "ROC AUC"])
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Score")
    ax.set_title("Model Version Comparison -- traffic-risk-classifier")
    ax.legend(loc="lower right", frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "model_version_comparison.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Saved figure: %s", path)
    return path


def run(
    input_path: Path, tracking_uri: str, experiment_name: str,
    registered_name: str, output_dir: Path,
) -> pd.DataFrame:
    df = load_dataset(input_path)
    df = prepare_ml_dataset(df)
    train_df, test_df = chronological_split(df)
    X_train, y_train = get_feature_matrix(train_df), train_df[TARGET_COLUMN]
    X_test, y_test = test_df[COMMON_FEATURE_COLUMNS], test_df[TARGET_COLUMN]

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    client = MlflowClient()
    reset_registered_model(client, registered_name)

    versions = build_versions()
    rows = []
    version_infos = {}

    with mlflow.start_run(run_name="model_versioning_traffic_risk_classifier") as parent_run:
        mlflow.set_tag("task", "model_versioning")
        for version_name, model in versions.items():
            with mlflow.start_run(run_name=version_name, nested=True) as child_run:
                logger.info("Fitting %s (%s) ...", version_name, MODEL_VERSIONS[version_name])
                model.fit(X_train, y_train)
                metrics, _ = evaluate_model(version_name, model, X_test, y_test)

                mlflow.log_params(MODEL_VERSIONS[version_name])
                mlflow.log_metrics({
                    "accuracy": metrics["accuracy"], "precision": metrics["precision"],
                    "recall": metrics["recall"], "f1_score": metrics["f1_score"],
                    "roc_auc": metrics["roc_auc"],
                })
                mlflow.set_tags({"task": "model_versioning", "algorithm": "random_forest", "version": version_name})

                signature = infer_signature(X_train, model.predict(X_train))
                mlflow.sklearn.log_model(
                    model, name="model", signature=signature,
                    input_example=X_train.head(3), serialization_format="pickle",
                )

                row = {"version": version_name, "run_id": child_run.info.run_id, **MODEL_VERSIONS[version_name], **{
                    k: v for k, v in metrics.items() if k != "model"
                }}
                rows.append(row)

                model_uri = f"runs:/{child_run.info.run_id}/model"
                mv = mlflow.register_model(model_uri, registered_name)
                version_infos[version_name] = mv
                logger.info(
                    "Registered '%s' as %s version %s (run_id=%s)",
                    version_name, registered_name, mv.version, child_run.info.run_id,
                )

    # v1 always keeps the "baseline" alias -- it's the reference point,
    # regardless of who wins. Whichever version scores the higher F1 gets
    # "champion" -- the alias deploy_api.py actually serves.
    client.set_registered_model_alias(registered_name, "baseline", version_infos["v1_baseline"].version)
    champion_name = max(rows, key=lambda r: r["f1_score"])["version"]
    client.set_registered_model_alias(registered_name, "champion", version_infos[champion_name].version)
    logger.info(
        "'%s' aliased 'champion' (F1=%.3f) -- this is the version deploy_api.py will serve",
        champion_name, next(r["f1_score"] for r in rows if r["version"] == champion_name),
    )

    for row in rows:
        row["alias"] = "champion" if row["version"] == champion_name else "baseline"
        row["registered_version"] = version_infos[row["version"]].version

    comparison = pd.DataFrame(rows).set_index("version")
    output_dir.mkdir(parents=True, exist_ok=True)
    table_path = output_dir / "model_registry_versions.csv"
    comparison.to_csv(table_path)
    logger.info("Wrote model version comparison table to %s", table_path)

    plot_version_comparison(rows, output_dir)
    return comparison


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Train, track, and register two versions of the high_risk classifier in MLflow's Model Registry."
    )
    parser.add_argument("--input", type=Path, default=here.parent / "pipeline" / "traffic_features.csv",
                         help="Path to Part 2's feature-engineered CSV.")
    parser.add_argument("--tracking-uri", type=str, default=f"sqlite:///{here}/mlflow.db",
                         help="MLflow tracking store URI (default: local SQLite, same store Task 4 uses).")
    parser.add_argument("--experiment", type=str, default="smart-city-traffic-intelligence",
                         help="MLflow experiment name (default: same experiment Task 4 uses).")
    parser.add_argument("--registered-name", type=str, default=REGISTERED_MODEL_NAME_DEFAULT,
                         help="Name to register the model under in MLflow's Model Registry.")
    parser.add_argument("--output-dir", type=Path, default=here / "artifacts",
                         help="Directory to write the version comparison table/chart.")
    parser.add_argument("--log-file", type=Path, default=here / "model_registry.log",
                         help="Path to write the log file.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable DEBUG-level logging.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging(args.log_file, verbose=args.verbose)
    logger.info("Command invoked: model_registry with arguments: %s", vars(args))

    try:
        run(args.input, args.tracking_uri, args.experiment, args.registered_name, args.output_dir)
    except UserInputError as e:
        logger.error("Invalid input: %s", e)
        return 1
    except (FileNotFoundError, pd.errors.EmptyDataError, pd.errors.ParserError,
            OSError, UnicodeDecodeError):
        logger.error("Model versioning aborted: could not load the dataset", exc_info=True)
        return 1
    except Exception:
        logger.error("Model versioning aborted due to an unexpected error", exc_info=True)
        return 1

    logger.info(
        "Model versioning completed successfully -- see %s for the version comparison table",
        args.output_dir / "model_registry_versions.csv",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
