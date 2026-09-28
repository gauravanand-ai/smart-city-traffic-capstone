#!/usr/bin/env python3
"""
train_classification.py
Part 3, Task 1 (classification half): train two supervised classifiers to
estimate accident likelihood from the proxy 'high_risk' label defined in the
Part 3 brief, on the common feature set shared with train_regression.py.

No accident dataset was provided for this capstone -- high_risk is a PROXY
built from congestion + weather, meant only to demonstrate the classification
workflow (train/test split, model comparison, evaluation metrics), not to
predict real accidents.

Models trained:
    - Logistic Regression (baseline, linear, scaled features)
    - Random Forest Classifier (tree-based ensemble)

Usage:
    python train_classification.py [--input PATH] [--output-dir PATH] [--log-file PATH] [-v]
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, roc_curve, confusion_matrix,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ml_common import (
    COMMON_FEATURE_COLUMNS, UserInputError, chronological_split,
    configure_logging, get_feature_matrix, load_dataset, prepare_ml_dataset,
)

logger = logging.getLogger(__name__)

# Reused from the palette already validated and used throughout Part 1/2
# (dashboard, visualize.py) -- kept consistent rather than introducing a new
# ad hoc pair.
COLOR_LOGISTIC = "#2a78d6"   # blue
COLOR_FOREST = "#eb6834"     # orange
COLOR_CHANCE = "#9a9a9a"     # neutral gray (reference diagonal, not a series)

TARGET_COLUMN = "high_risk"


def build_models() -> dict:
    """Two algorithms, as required: a linear baseline (Logistic Regression,
    inside a scaling Pipeline since linear models are scale-sensitive) and a
    tree-based ensemble (Random Forest, which needs no scaling). Both use
    class_weight='balanced' because high_risk is imbalanced (~5% positive --
    see ml_common.add_proxy_labels' WARNING)."""
    return {
        "logistic_regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)),
        ]),
        "random_forest": RandomForestClassifier(
            n_estimators=150, max_depth=10, min_samples_leaf=5, class_weight="balanced",
            random_state=42, n_jobs=-1,
        ),
    }


def evaluate_model(name: str, model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "model": name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    logger.info(
        "[%s] accuracy=%.3f precision=%.3f recall=%.3f f1=%.3f roc_auc=%.3f "
        "(confusion matrix: tn=%d fp=%d fn=%d tp=%d)",
        name, metrics["accuracy"], metrics["precision"], metrics["recall"],
        metrics["f1_score"], metrics["roc_auc"], tn, fp, fn, tp,
    )
    return metrics, y_proba


def plot_roc_curves(y_test: pd.Series, proba_by_model: dict, output_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7, 6))
    colors = {"logistic_regression": COLOR_LOGISTIC, "random_forest": COLOR_FOREST}
    labels = {"logistic_regression": "Logistic Regression", "random_forest": "Random Forest"}

    for name, y_proba in proba_by_model.items():
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc = roc_auc_score(y_test, y_proba)
        ax.plot(fpr, tpr, color=colors[name], linewidth=2,
                 label=f"{labels[name]} (AUC = {auc:.3f})")

    ax.plot([0, 1], [0, 1], color=COLOR_CHANCE, linewidth=1.5, linestyle="--", label="Chance (AUC = 0.500)")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve Comparison -- Proxy High-Risk Classification")
    ax.legend(loc="lower right", frameon=False)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "classification_roc_curves.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Saved figure: %s", path)
    return path


def save_comparison_table(rows: list[dict], output_dir: Path) -> Path:
    comparison = pd.DataFrame(rows).set_index("model")
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "classification_model_comparison.csv"
    comparison.to_csv(path)
    logger.info("Wrote classification model comparison table to %s", path)
    return comparison


def save_models(models: dict, output_dir: Path) -> None:
    import joblib
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, model in models.items():
        path = output_dir / f"classification_{name}.joblib"
        joblib.dump(model, path)
        logger.info("Saved trained model: %s", path)


def run(input_path: Path, output_dir: Path) -> pd.DataFrame:
    df = load_dataset(input_path)
    df = prepare_ml_dataset(df)

    train_df, test_df = chronological_split(df)
    X_train, y_train = get_feature_matrix(train_df), train_df[TARGET_COLUMN]
    X_test, y_test = test_df[COMMON_FEATURE_COLUMNS], test_df[TARGET_COLUMN]

    logger.info(
        "Training set high_risk positive rate: %.2f%% | Test set: %.2f%%",
        100 * y_train.mean(), 100 * y_test.mean(),
    )

    models = build_models()
    results, proba_by_model = [], {}
    for name, model in models.items():
        logger.info("Fitting %s ...", name)
        model.fit(X_train, y_train)
        metrics, y_proba = evaluate_model(name, model, X_test, y_test)
        results.append(metrics)
        proba_by_model[name] = y_proba

    comparison = save_comparison_table(results, output_dir)
    plot_roc_curves(y_test, proba_by_model, output_dir)
    save_models(models, output_dir)
    return comparison


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Train and compare classification models for the proxy accident-risk label."
    )
    parser.add_argument("--input", type=Path, default=here.parent / "pipeline" / "traffic_features.csv",
                         help="Path to Part 2's feature-engineered CSV.")
    parser.add_argument("--output-dir", type=Path, default=here / "artifacts",
                         help="Directory to write the comparison table, ROC chart, and saved models.")
    parser.add_argument("--log-file", type=Path, default=here / "train_classification.log",
                         help="Path to write the log file.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable DEBUG-level logging.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging(args.log_file, verbose=args.verbose)
    logger.info("Command invoked: train_classification with arguments: %s", vars(args))

    try:
        comparison = run(args.input, args.output_dir)
    except UserInputError as e:
        logger.error("Invalid input: %s", e)
        return 1
    except (FileNotFoundError, pd.errors.EmptyDataError, pd.errors.ParserError,
            OSError, UnicodeDecodeError):
        logger.error("Classification training aborted: could not load the dataset", exc_info=True)
        return 1
    except Exception:
        logger.error("Classification training aborted due to an unexpected error", exc_info=True)
        return 1

    # Per-model metrics were already logged at INFO inside evaluate_model();
    # this script is a batch training job (not an interactive CLI like
    # app.py), so its result is the saved comparison CSV/chart/models on
    # disk -- not console output -- and it follows pipeline.py/
    # feature_engineering.py/visualize.py's no-print(), logging-only
    # convention rather than app.py's print()-for-results one.
    logger.info(
        "Classification training completed successfully -- see %s for the "
        "full model comparison table", args.output_dir / "classification_model_comparison.csv",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
