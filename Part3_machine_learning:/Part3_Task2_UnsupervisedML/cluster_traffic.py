#!/usr/bin/env python3
"""
cluster_traffic.py
Part 3, Task 2 (K-means half): cluster traffic *conditions* -- hour of day,
weather severity, and traffic volume -- since no accident dataset exists to
cluster incidents against. Each cluster is interpreted in plain language from
its own centroid statistics (never hardcoded), the same "compute, don't
narrate" approach used for every interpretation elsewhere in this project.

Usage:
    python cluster_traffic.py [--input PATH] [--k N] [--output-dir PATH] [--log-file PATH] [-v]
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
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from ml_common import (
    UserInputError, add_weather_severity, configure_logging, load_dataset,
    prepare_ml_dataset,
)

logger = logging.getLogger(__name__)

CLUSTER_FEATURES = ["hour_sin", "hour_cos", "weather_severity", "traffic_volume"]

# Categorical hue order, reused from the palette already validated and used
# throughout Parts 1-2 (dashboard, visualize.py) -- fixed order, never cycled.
CLUSTER_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#4a3aa7", "#e34948", "#008300"]


def evaluate_k_range(X_scaled: np.ndarray, k_range: range) -> pd.DataFrame:
    """Elbow (inertia) + silhouette score across a range of k, logged so the
    chosen k is a documented decision, not an arbitrary constant."""
    rows = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X_scaled)
        sil = silhouette_score(X_scaled, km.labels_, sample_size=5000, random_state=42)
        rows.append({"k": k, "inertia": km.inertia_, "silhouette": sil})
        logger.debug("k=%d: inertia=%.0f, silhouette=%.4f", k, km.inertia_, sil)
    result = pd.DataFrame(rows)
    logger.info(
        "Evaluated k=%d..%d for K-means (elbow/silhouette): %s",
        k_range.start, k_range.stop - 1,
        ", ".join(f"k={r.k}:sil={r.silhouette:.3f}" for r in result.itertuples()),
    )
    return result


def circular_mean_hour(hours: pd.Series) -> float:
    """Mean hour that respects the 23->0 wraparound (a plain arithmetic mean
    of, say, [23, 1] would wrongly give 12 instead of 0)."""
    radians = 2 * np.pi * hours / 24
    mean_angle = np.arctan2(np.sin(radians).mean(), np.cos(radians).mean())
    return (mean_angle / (2 * np.pi) * 24) % 24


def _time_of_day_phrase(hour: float) -> str:
    if hour < 5 or hour >= 22:
        return "overnight"
    if hour < 10:
        return "morning"
    if hour < 14:
        return "midday"
    if hour < 18:
        return "afternoon/evening rush hour"
    return "evening"


def _traffic_phrase(volume: float, q1: float, q2: float, q3: float) -> str:
    if volume <= q1:
        return "low traffic"
    if volume <= q2:
        return "moderate traffic"
    if volume <= q3:
        return "high traffic"
    return "very high traffic"


def build_cluster_profile(df: pd.DataFrame, quartiles: tuple[float, float, float]) -> pd.DataFrame:
    q1, q2, q3 = quartiles
    rows = []
    for cluster_id, group in df.groupby("cluster"):
        avg_hour = circular_mean_hour(group["hour"])
        avg_severity = group["weather_severity"].mean()
        avg_volume = group["traffic_volume"].mean()
        dominant_weather = group["weather_severity_label"].mode().iat[0]
        dominant_congestion = group["congestion_category_p3"].mode().iat[0]
        pct_weekend = 100 * group["is_weekend"].mean()

        interpretation = (
            f"{_time_of_day_phrase(avg_hour).capitalize()} conditions "
            f"(avg. hour {avg_hour:04.1f}h), predominantly '{dominant_weather.lower()}' weather, "
            f"with {_traffic_phrase(avg_volume, q1, q2, q3)} "
            f"(avg. {avg_volume:,.0f} veh/hr, mostly '{dominant_congestion}' congestion); "
            f"{pct_weekend:.0f}% of hours in this cluster are on a weekend."
        )

        rows.append({
            "cluster": cluster_id,
            "n_rows": len(group),
            "pct_of_data": 100 * len(group) / len(df),
            "avg_hour": round(avg_hour, 1),
            "avg_weather_severity": round(avg_severity, 2),
            "dominant_weather": dominant_weather,
            "avg_traffic_volume": round(avg_volume, 0),
            "dominant_congestion": dominant_congestion,
            "pct_weekend": round(pct_weekend, 1),
            "interpretation": interpretation,
        })
        logger.info("Cluster %d (%d rows, %.1f%%): %s", cluster_id, len(group), 100 * len(group) / len(df), interpretation)

    return pd.DataFrame(rows).set_index("cluster")


def plot_clusters(df: pd.DataFrame, profile: pd.DataFrame, output_dir: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    ax = axes[0]
    for cluster_id in sorted(df["cluster"].unique()):
        sub = df[df["cluster"] == cluster_id]
        ax.scatter(
            sub["hour"] + np.random.default_rng(42).uniform(-0.3, 0.3, len(sub)),  # jitter for readability
            sub["traffic_volume"],
            s=5, alpha=0.15, color=CLUSTER_COLORS[cluster_id % len(CLUSTER_COLORS)], linewidths=0,
            label=f"Cluster {cluster_id}",
        )
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Traffic volume (veh/hr)")
    ax.set_title("Traffic condition clusters: hour vs. volume")
    ax.set_xticks(range(0, 24, 3))
    leg = ax.legend(loc="upper left", frameon=False, markerscale=4)
    for lh in leg.legend_handles:
        lh.set_alpha(1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax2 = axes[1]
    clusters = profile.index.tolist()
    colors = [CLUSTER_COLORS[c % len(CLUSTER_COLORS)] for c in clusters]
    ax2.barh(
        [f"Cluster {c}\n({profile.loc[c, 'pct_of_data']:.0f}% of hours)" for c in clusters],
        profile["avg_weather_severity"], color=colors,
    )
    ax2.set_xlabel("Avg. weather severity (0=Clear .. 4=Severe)")
    ax2.set_title("Weather severity by cluster")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    fig.suptitle("K-Means Traffic Condition Clusters (k=%d)" % len(clusters))
    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "cluster_profile.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Saved figure: %s", path)
    return path


def run(input_path: Path, k: int, output_dir: Path) -> pd.DataFrame:
    df = load_dataset(input_path)
    df = prepare_ml_dataset(df)  # holiday_flag, month/year, congestion_category_p3, high_risk
    df = add_weather_severity(df)

    X = df[CLUSTER_FEATURES].copy()
    X_scaled = StandardScaler().fit_transform(X)
    logger.info("Built K-means feature matrix: %d rows x %d columns (%s)", *X.shape, ", ".join(CLUSTER_FEATURES))

    evaluate_k_range(X_scaled, range(2, 9))

    logger.info("Fitting final K-means with k=%d ...", k)
    km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X_scaled)
    df["cluster"] = km.labels_
    final_sil = silhouette_score(X_scaled, km.labels_, sample_size=5000, random_state=42)
    logger.info("Final model: k=%d, inertia=%.0f, silhouette=%.4f", k, km.inertia_, final_sil)

    quartiles = tuple(df["traffic_volume"].quantile([0.25, 0.5, 0.75]).values)
    profile = build_cluster_profile(df, quartiles)

    output_dir.mkdir(parents=True, exist_ok=True)
    profile_path = output_dir / "cluster_profile.csv"
    profile.to_csv(profile_path)
    logger.info("Wrote cluster profile table to %s", profile_path)

    plot_clusters(df, profile, output_dir)
    return profile


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="K-means clustering of traffic conditions (hour, weather severity, traffic volume)."
    )
    parser.add_argument("--input", type=Path, default=here.parent / "pipeline" / "traffic_features.csv",
                         help="Path to Part 2's feature-engineered CSV.")
    parser.add_argument("--k", type=int, default=5,
                         help="Number of clusters (default: 5, chosen via elbow/silhouette analysis -- see README).")
    parser.add_argument("--output-dir", type=Path, default=here / "artifacts",
                         help="Directory to write the cluster profile table and chart.")
    parser.add_argument("--log-file", type=Path, default=here / "cluster_traffic.log",
                         help="Path to write the log file.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable DEBUG-level logging.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging(args.log_file, verbose=args.verbose)
    logger.info("Command invoked: cluster_traffic with arguments: %s", vars(args))

    if args.k < 2:
        logger.error("Invalid input: --k must be at least 2, got %d.", args.k)
        return 1

    try:
        run(args.input, args.k, args.output_dir)
    except UserInputError as e:
        logger.error("Invalid input: %s", e)
        return 1
    except (FileNotFoundError, pd.errors.EmptyDataError, pd.errors.ParserError,
            OSError, UnicodeDecodeError):
        logger.error("Clustering aborted: could not load the dataset", exc_info=True)
        return 1
    except Exception:
        logger.error("Clustering aborted due to an unexpected error", exc_info=True)
        return 1

    logger.info(
        "Clustering completed successfully -- see %s for the full cluster profile table",
        args.output_dir / "cluster_profile.csv",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
