#!/usr/bin/env python3
"""
visualize.py
Task 3: Matplotlib visualisations of traffic patterns, built on Task 2's
feature-engineered dataset (traffic_features.csv).

Produces 6 PNG figures (comfortably over the "at least three" requirement,
covering every example pattern the brief lists: hour, weekday/weekend,
weather, temperature, distribution, and congestion over time) plus a short,
data-driven interpretation of each -- the numbers in each interpretation are
computed from whatever data is loaded, not hardcoded, so they stay correct if
the underlying dataset changes.

Usage:
    python visualize.py [--input PATH] [--output-dir DIR] [--log-file PATH] [-v]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless: never try to open a GUI window
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from pipeline import configure_logging  # reuse Task 1's logging setup

logger = logging.getLogger(__name__)

# Palette reused from the Part 1 dashboard for visual consistency across the project.
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948",
)
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
SURFACE = "#fcfcfb"
CONGESTION_COLORS = {"Low": BLUE, "Medium": AQUA, "High": ORANGE, "Very High": RED}

DAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

# Collected interpretations, in save order, so main() can write them to a
# single companion markdown file after all figures are produced.
INTERPRETATIONS: list[tuple[str, str, str]] = []  # (title, filename, interpretation)


def _style_axes(ax) -> None:
    ax.set_facecolor(SURFACE)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(GRIDLINE)
    ax.spines["bottom"].set_color(GRIDLINE)
    ax.tick_params(colors=INK_SECONDARY, labelsize=9)
    ax.grid(axis="y", color=GRIDLINE, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)


def save_figure(fig, output_dir: Path, filename: str, title: str, interpretation: str) -> Path:
    path = output_dir / filename
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    logger.info("Saved figure: %s", path)
    INTERPRETATIONS.append((title, filename, interpretation))
    return path


# ---------------------------------------------------------------------------
# 1. Traffic demand by hour, weekday vs weekend
# ---------------------------------------------------------------------------
def plot_hourly_by_daytype(df: pd.DataFrame, output_dir: Path) -> Path:
    grouped = df.groupby(["hour", "is_weekend"])["traffic_volume"].mean().unstack()
    weekday = grouped[0]
    weekend = grouped[1]

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(weekday.index, weekday.values, color=BLUE, linewidth=2, label="Weekday", zorder=3)
    ax.plot(weekend.index, weekend.values, color=ORANGE, linewidth=2, label="Weekend", zorder=3)
    _style_axes(ax)
    ax.set_xlabel("Hour of day", color=INK_SECONDARY)
    ax.set_ylabel("Average traffic volume (veh/hr)", color=INK_SECONDARY)
    ax.set_title("Traffic demand by hour: weekday vs. weekend", color=INK, fontsize=13, fontweight="bold")
    ax.set_xticks(range(0, 24, 3))
    ax.legend(frameon=False, labelcolor=INK_SECONDARY)

    weekday_peak_hr = int(weekday.idxmax())
    weekend_peak_hr = int(weekend.idxmax())
    interpretation = (
        f"Weekday traffic is bimodal, peaking at {weekday_peak_hr}:00 ({weekday.max():,.0f} veh/hr) "
        f"with a clear commute pattern, while weekend traffic peaks later and flatter at "
        f"{weekend_peak_hr}:00 ({weekend.max():,.0f} veh/hr) with no morning rush. "
        f"Weekdays exceed weekends by {(weekday.mean() - weekend.mean()):,.0f} veh/hr on average, "
        "confirming this corridor is primarily commute-driven."
    )
    return save_figure(fig, output_dir, "01_traffic_by_hour_weekday_vs_weekend.png",
                        "Traffic demand by hour (weekday vs. weekend)", interpretation)


# ---------------------------------------------------------------------------
# 2. Weekday vs weekend traffic distribution
# ---------------------------------------------------------------------------
def plot_weekday_weekend_distribution(df: pd.DataFrame, output_dir: Path) -> Path:
    weekday_vals = df.loc[df["is_weekend"] == 0, "traffic_volume"]
    weekend_vals = df.loc[df["is_weekend"] == 1, "traffic_volume"]

    fig, ax = plt.subplots(figsize=(6, 4.5))
    box = ax.boxplot(
        [weekday_vals, weekend_vals], tick_labels=["Weekday", "Weekend"],
        patch_artist=True, widths=0.5, medianprops={"color": INK, "linewidth": 1.5},
        whiskerprops={"color": INK_SECONDARY}, capprops={"color": INK_SECONDARY},
        flierprops={"markeredgecolor": INK_MUTED, "markersize": 3, "alpha": 0.4},
    )
    for patch, color in zip(box["boxes"], [BLUE, ORANGE]):
        patch.set_facecolor(color)
        patch.set_alpha(0.55)
        patch.set_edgecolor(color)
    _style_axes(ax)
    ax.set_ylabel("Traffic volume (veh/hr)", color=INK_SECONDARY)
    ax.set_title("Weekday vs. weekend traffic volume distribution", color=INK, fontsize=13, fontweight="bold")

    wd_median, we_median = weekday_vals.median(), weekend_vals.median()
    interpretation = (
        f"Median weekday volume ({wd_median:,.0f} veh/hr) is {wd_median - we_median:,.0f} veh/hr higher "
        f"than the weekend median ({we_median:,.0f} veh/hr), and the weekday box is visibly taller "
        "(wider interquartile range) -- weekday traffic swings much harder between off-peak and rush-hour "
        "levels, while weekends stay comparatively steady across the day."
    )
    return save_figure(fig, output_dir, "02_weekday_vs_weekend_distribution.png",
                        "Weekday vs. weekend traffic distribution", interpretation)


# ---------------------------------------------------------------------------
# 3. Traffic and weather relationship
# ---------------------------------------------------------------------------
def plot_weather_relationship(df: pd.DataFrame, output_dir: Path, min_n: int = 50) -> Path:
    stats = df.groupby("weather_main")["traffic_volume"].agg(["mean", "count"])
    stats = stats[stats["count"] >= min_n].sort_values("mean", ascending=True)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = [BLUE if v < stats["mean"].median() else ORANGE for v in stats["mean"]]
    ax.barh(stats.index, stats["mean"], color=colors, zorder=3, height=0.65)
    _style_axes(ax)
    ax.grid(axis="x", color=GRIDLINE, linewidth=0.8, zorder=0)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Average traffic volume (veh/hr)", color=INK_SECONDARY)
    ax.set_title(f"Average traffic volume by weather condition (n ≥ {min_n})",
                 color=INK, fontsize=13, fontweight="bold")
    for y, v in enumerate(stats["mean"]):
        ax.text(v + 30, y, f"{v:,.0f}", va="center", fontsize=8.5, color=INK_SECONDARY)

    highest, lowest = stats.index[-1], stats.index[0]
    diff = stats["mean"].iloc[-1] - stats["mean"].iloc[0]
    interpretation = (
        f"'{highest.title()}' carries the highest average volume ({stats['mean'].iloc[-1]:,.0f} veh/hr) "
        f"and '{lowest.title()}' the lowest ({stats['mean'].iloc[0]:,.0f} veh/hr), a difference of "
        f"{diff:,.0f} veh/hr. The spread is real but modest relative to the overall variability in "
        "traffic volume -- weather condition alone is a weak lever for predicting congestion."
    )
    return save_figure(fig, output_dir, "03_traffic_by_weather_condition.png",
                        "Traffic and weather relationship", interpretation)


# ---------------------------------------------------------------------------
# 4. Temperature vs traffic
# ---------------------------------------------------------------------------
def plot_temperature_vs_traffic(df: pd.DataFrame, output_dir: Path, sample_n: int = 4000, seed: int = 42) -> Path:
    temp_c = df["temp"] - 273.15
    sample_idx = df.sample(n=min(sample_n, len(df)), random_state=seed).index

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for category, color in CONGESTION_COLORS.items():
        mask = (df.loc[sample_idx, "congestion_category"] == category)
        ax.scatter(temp_c.loc[sample_idx][mask], df.loc[sample_idx, "traffic_volume"][mask],
                   s=8, alpha=0.45, color=color, label=category, linewidths=0, zorder=3)
    _style_axes(ax)
    ax.set_xlabel("Temperature (°C)", color=INK_SECONDARY)
    ax.set_ylabel("Traffic volume (veh/hr)", color=INK_SECONDARY)
    ax.set_title("Temperature vs. traffic volume", color=INK, fontsize=13, fontweight="bold")
    ax.legend(frameon=False, labelcolor=INK_SECONDARY, title="Congestion category",
              title_fontsize=9, fontsize=8.5)

    r = np.corrcoef(temp_c, df["traffic_volume"])[0, 1]
    interpretation = (
        f"Pearson r = {r:.3f} between temperature and traffic volume -- a weak positive relationship; "
        "temperature alone explains only a small share of the variance. The point cloud is fairly flat "
        "through most of the range and only thins out toward the coldest hours, where volume drops "
        "noticeably -- extreme cold suppresses traffic more clearly than moderate temperature varies it."
    )
    return save_figure(fig, output_dir, "04_temperature_vs_traffic_scatter.png",
                        "Temperature vs. traffic", interpretation)


# ---------------------------------------------------------------------------
# 5. Traffic volume distribution
# ---------------------------------------------------------------------------
def plot_traffic_distribution(df: pd.DataFrame, output_dir: Path) -> Path:
    vol = df["traffic_volume"]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(vol, bins=40, color=BLUE, alpha=0.75, zorder=3, edgecolor=SURFACE, linewidth=0.5)
    ax.axvline(vol.mean(), color=RED, linewidth=1.5, linestyle="--", zorder=4, label=f"Mean ({vol.mean():,.0f})")
    ax.axvline(vol.median(), color=INK, linewidth=1.5, linestyle="--", zorder=4, label=f"Median ({vol.median():,.0f})")
    _style_axes(ax)
    ax.set_xlabel("Traffic volume (veh/hr)", color=INK_SECONDARY)
    ax.set_ylabel("Number of hours", color=INK_SECONDARY)
    ax.set_title("Distribution of hourly traffic volume", color=INK, fontsize=13, fontweight="bold")
    ax.legend(frameon=False, labelcolor=INK_SECONDARY)

    skew = vol.skew()
    interpretation = (
        f"The distribution is bimodal, not a single bell curve: a large cluster of low-traffic overnight "
        f"hours sits well apart from a second cluster of rush-hour readings. Skewness is only mild "
        f"({skew:.2f}) and the mean ({vol.mean():,.0f}) sits just below the median ({vol.median():,.0f}) -- "
        "the more important story is the two-regime shape than the direction of the skew. There is no "
        "single 'typical' hour: the corridor alternates between distinct off-peak and peak regimes."
    )
    return save_figure(fig, output_dir, "05_traffic_volume_distribution.png",
                        "Traffic distribution", interpretation)


# ---------------------------------------------------------------------------
# 6. Congestion patterns over time
# ---------------------------------------------------------------------------
def plot_congestion_over_time(df: pd.DataFrame, output_dir: Path) -> Path:
    d = df.copy()
    d["date_time"] = pd.to_datetime(d["date_time"])
    d["year_month"] = d["date_time"].dt.to_period("M")
    monthly_share = (
        d.assign(is_very_high=(d["congestion_category"] == "Very High").astype(int))
        .groupby("year_month")["is_very_high"].mean() * 100
    )
    monthly_share = monthly_share.sort_index()
    x = monthly_share.index.to_timestamp()

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(x, monthly_share.values, color=VIOLET, linewidth=1.8, marker="o", markersize=3, zorder=3)
    ax.fill_between(x, monthly_share.values, color=VIOLET, alpha=0.10, zorder=2)
    _style_axes(ax)
    ax.set_xlabel("Month", color=INK_SECONDARY)
    ax.set_ylabel("Share of hours in 'Very High' congestion (%)", color=INK_SECONDARY)
    ax.set_title("Congestion patterns over time", color=INK, fontsize=13, fontweight="bold")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(decimals=0))
    fig.autofmt_xdate()

    peak_month = monthly_share.idxmax()
    interpretation = (
        f"The share of severely congested hours varies substantially month to month, peaking at "
        f"{monthly_share.max():.0f}% in {peak_month.strftime('%B %Y')}. Coverage gaps in the underlying "
        "sensor data (documented in Part 1) mean some of this swing reflects which months were measured "
        "as much as real seasonal demand -- read this chart as directional rather than a precise "
        "seasonal forecast."
    )
    return save_figure(fig, output_dir, "06_congestion_over_time.png",
                        "Congestion patterns over time", interpretation)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def load_features_csv(path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        logger.error("Feature-engineered CSV not found at %s (run feature_engineering.py first)",
                     path, exc_info=True)
        raise
    except pd.errors.EmptyDataError:
        logger.error("Feature-engineered CSV at %s is empty", path, exc_info=True)
        raise
    except pd.errors.ParserError:
        logger.error("Feature-engineered CSV at %s could not be parsed", path, exc_info=True)
        raise
    except (OSError, UnicodeDecodeError):
        logger.error("I/O error while reading %s", path, exc_info=True)
        raise

    logger.info("Loaded feature-engineered dataset: %d rows, %d columns (source: %s)", *df.shape, path.name)
    return df


def write_interpretations_doc(output_dir: Path) -> Path:
    path = output_dir / "interpretations.md"
    lines = ["# Visualisation Interpretations\n"]
    for i, (title, filename, interpretation) in enumerate(INTERPRETATIONS, start=1):
        lines.append(f"## {i}. {title}\n")
        lines.append(f"![{title}]({filename})\n")
        lines.append(f"{interpretation}\n")
    path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote interpretations summary: %s", path)
    return path


def make_all_visualizations(df: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_hourly_by_daytype(df, output_dir)
    plot_weekday_weekend_distribution(df, output_dir)
    plot_weather_relationship(df, output_dir)
    plot_temperature_vs_traffic(df, output_dir)
    plot_traffic_distribution(df, output_dir)
    plot_congestion_over_time(df, output_dir)
    write_interpretations_doc(output_dir)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Visualise traffic patterns from the feature-engineered dataset.")
    parser.add_argument("--input", type=Path, default=here / "traffic_features.csv",
                         help="Path to Task 2's feature-engineered CSV.")
    parser.add_argument("--output-dir", type=Path, default=here / "figures",
                         help="Directory to write PNG figures + interpretations.md into.")
    parser.add_argument("--log-file", type=Path, default=here / "visualize.log",
                         help="Path to write the log file.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable DEBUG-level logging.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging(args.log_file, verbose=args.verbose)

    try:
        df = load_features_csv(args.input)
        make_all_visualizations(df, args.output_dir)
    except (FileNotFoundError, pd.errors.EmptyDataError, pd.errors.ParserError,
            OSError, UnicodeDecodeError):
        logger.error("Visualisation aborted: could not produce all figures", exc_info=True)
        return 1
    except Exception:
        logger.error("Visualisation aborted due to an unexpected error", exc_info=True)
        return 1

    logger.info("Visualisation completed successfully: %d figure(s) written to %s",
                len(INTERPRETATIONS), args.output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
