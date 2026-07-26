#!/usr/bin/env python3

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
print("modules imported")

SCRIPT_DIR = Path(__file__).resolve().parent
FIGDATA_DIR = SCRIPT_DIR / "figdata"

GRID_METRICS_FILE = FIGDATA_DIR / "fig6_layer1_grid_metrics.csv"
MONTHLY_CLIMATOLOGY_FILE = FIGDATA_DIR / "fig6_monthly_climatology.csv"
OUTPUT_FILE = SCRIPT_DIR / "Fig6.png"

REGION_ORDER = [
    "global",
    "africa",
    "asia",
    "europe",
    "north_america",
    "south_america",
    "oceania",
]

REGION_LABELS = {
    "global": "Global",
    "africa": "Africa",
    "asia": "Asia",
    "europe": "Europe",
    "north_america": "N. America",
    "south_america": "S. America",
    "oceania": "Oceania",
}

TARGET_COLOR = "#2166ac"
ERA_COLOR = "#4d4d4d"
LIGHT_TARGET = "#92c5de"
LIGHT_GRAY = "#bdbdbd"

GLOBAL_BOX_COLOR = "#B5B5B5"
REGIONAL_BOX_COLOR = "#E3E3E3"


def read_grid_metrics() -> pd.DataFrame:
    """Read released grid-level Figure 6 metrics."""
    metrics = pd.read_csv(GRID_METRICS_FILE, na_values=-9999)

    required = {
        "region",
        "daily_corr",
        "difference_corr",
    }
    missing = required.difference(metrics.columns)
    if missing:
        raise ValueError(
            f"{GRID_METRICS_FILE} missing required columns: "
            f"{sorted(missing)}"
        )

    if metrics.empty:
        raise RuntimeError(f"Empty metric file: {GRID_METRICS_FILE}")

    return metrics


def read_monthly_climatology() -> pd.DataFrame:
    """Read released monthly climatology summaries."""
    climatology = pd.read_csv(
        MONTHLY_CLIMATOLOGY_FILE,
        na_values=-9999,
    )

    required = {
        "month",
        "target_median",
        "target_q25",
        "target_q75",
        "era_median",
        "era_q25",
        "era_q75",
    }
    missing = required.difference(climatology.columns)
    if missing:
        raise ValueError(
            f"{MONTHLY_CLIMATOLOGY_FILE} missing required columns: "
            f"{sorted(missing)}"
        )

    climatology = climatology.sort_values("month")

    if len(climatology) != 12:
        raise ValueError(
            f"{MONTHLY_CLIMATOLOGY_FILE} must contain 12 monthly rows."
        )

    return climatology


def prepare_regional_boxplots(
    metrics: pd.DataFrame,
    value_column: str,
):
    """Prepare global and regional correlation arrays."""
    arrays = []
    labels = []
    sample_sizes = []

    for region in REGION_ORDER:
        subset = (
            metrics
            if region == "global"
            else metrics.loc[metrics["region"] == region]
        )

        values = subset[value_column].dropna().to_numpy()

        arrays.append(values)
        labels.append(REGION_LABELS[region])
        sample_sizes.append(len(values))

    positions = np.arange(len(REGION_ORDER), 0, -1)

    return arrays, labels, sample_sizes, positions


def draw_horizontal_boxplot(
    axis,
    arrays,
    labels,
    sample_sizes,
    positions,
    panel_label,
    show_sample_sizes=True,
):
    """Draw one regional correlation boxplot."""
    boxplot = axis.boxplot(
        arrays,
        vert=False,
        positions=positions,
        widths=0.6,
        showfliers=False,
        patch_artist=True,
        medianprops={
            "color": "black",
            "linewidth": 1.5,
        },
        boxprops={
            "color": "0.35",
            "linewidth": 1.0,
        },
        whiskerprops={
            "color": "0.35",
            "linewidth": 1.0,
        },
        capprops={
            "color": "0.35",
            "linewidth": 1.0,
        },
    )

    for box_index, box in enumerate(boxplot["boxes"]):
        box.set_facecolor(
            GLOBAL_BOX_COLOR
            if box_index == 0
            else REGIONAL_BOX_COLOR
        )

    axis.set_yticks(positions)
    axis.set_yticklabels(labels)
    axis.set_xlim(-0.2, 1.0)
    axis.set_xticks(np.arange(-0.2, 1.01, 0.2))
    axis.set_xlabel("Correlation", fontsize=10)
    axis.tick_params(axis="both", labelsize=9)
    axis.grid(alpha=0.2, linewidth=0.6, axis="x")
    axis.set_title(
        panel_label,
        fontsize=14,
        weight="bold",
        x=0.02,
        y=1.02,
    )

    if show_sample_sizes:
        for y_position, sample_size in zip(
            positions,
            sample_sizes,
        ):
            axis.text(
                -0.17,
                y_position,
                f"n={sample_size:,}",
                va="center",
                ha="left",
                fontsize=8,
                color="0.35",
                bbox={
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.8,
                    "pad": 0.3,
                },
            )


def plot_figure() -> None:
    """Create and save Figure 6."""
    metrics = read_grid_metrics()
    climatology = read_monthly_climatology()

    daily_arrays, region_labels, daily_n, positions = (
        prepare_regional_boxplots(
            metrics,
            value_column="daily_corr",
        )
    )

    difference_arrays, _, difference_n, _ = (
        prepare_regional_boxplots(
            metrics,
            value_column="difference_corr",
        )
    )

    figure = plt.figure(
        figsize=(7, 7),
        constrained_layout=True,
    )

    grid = figure.add_gridspec(
        2,
        2,
        height_ratios=[0.95, 1.20],
        wspace=0.03,
        hspace=0.05,
    )

    climatology_axis = figure.add_subplot(grid[0, :])
    months = climatology["month"].to_numpy()

    climatology_axis.plot(
        months,
        climatology["era_median"],
        color=ERA_COLOR,
        linewidth=2.2,
        label="ERA5",
    )
    climatology_axis.fill_between(
        months,
        climatology["era_q25"],
        climatology["era_q75"],
        color=LIGHT_GRAY,
        alpha=0.3,
    )

    climatology_axis.plot(
        months,
        climatology["target_median"],
        color=TARGET_COLOR,
        linewidth=2.2,
        label="SoMoBench",
    )
    climatology_axis.fill_between(
        months,
        climatology["target_q25"],
        climatology["target_q75"],
        color=LIGHT_TARGET,
        alpha=0.3,
    )

    climatology_axis.set_xlim(1, 12)
    climatology_axis.set_ylim(0.15, 0.4)
    climatology_axis.set_xticks(range(1, 13))
    climatology_axis.set_xticklabels(
        [
            "Jan", "Feb", "Mar", "Apr", "May", "Jun",
            "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
        ]
    )
    climatology_axis.set_xlabel("Month", fontsize=10)
    climatology_axis.set_ylabel(
        r"Soil moisture [$\mathrm{m^3\,m^{-3}}$]",
        fontsize=10,
    )
    climatology_axis.tick_params(axis="both", labelsize=9)
    climatology_axis.set_title(
        "(a)",
        fontsize=14,
        weight="bold",
        x=0.005,
        y=1.02,
    )
    climatology_axis.grid(alpha=0.2, linewidth=0.6)
    climatology_axis.legend(
        loc="lower left",
        frameon=False,
        ncol=2,
        fontsize=9,
    )
    climatology_axis.text(
        0.88,
        0.90,
        f"n={len(metrics):,}",
        transform=climatology_axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
    )

    daily_axis = figure.add_subplot(grid[1, 0])
    draw_horizontal_boxplot(
        axis=daily_axis,
        arrays=daily_arrays,
        labels=region_labels,
        sample_sizes=daily_n,
        positions=positions,
        panel_label="(b)",
        show_sample_sizes=True,
    )

    difference_axis = figure.add_subplot(grid[1, 1])
    draw_horizontal_boxplot(
        axis=difference_axis,
        arrays=difference_arrays,
        labels=[""] * len(region_labels),
        sample_sizes=difference_n,
        positions=positions,
        panel_label="(c)",
        show_sample_sizes=False,
    )

    figure.savefig(
        OUTPUT_FILE,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(figure)

    print(f"Saved: {OUTPUT_FILE}")


def main() -> None:
    plot_figure()


if __name__ == "__main__":
    main()
