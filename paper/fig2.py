#!/usr/bin/env python
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
print("modules imported)")

# ==========================================================
# User settings
# Input metadata file (included in the benchmark dataset).
# Place the metadata CSV in the same folder as this script.
# ==========================================================
METADATA_FILE = "grid_metadata.csv"


# ==========================================================
# Plot parameters
# ==========================================================

LAYER_LABELS = {
    1: "Layer1 (0–10cm)",
    2: "Layer2 (10–30cm)",
    3: "Layer3 (30–50cm)",
}

REGION_ORDER = [
    "africa",
    "asia",
    "europe",
    "north_america",
    "south_america",
    "oceania",
]

REGION_COLORS = {
    "north_america": "black",
    "europe": "goldenrod",
    "asia": "royalblue",
    "oceania": "crimson",
    "africa": "sienna",
    "south_america": "forestgreen",
}

REGION_LABELS = {
    "north_america": "N. America",
    "south_america": "S. America",
    "europe": "Europe",
    "asia": "Asia",
    "oceania": "Oceania",
    "africa": "Africa",
}

REGION_ALIASES = {
    "namerica": "north_america",
    "northamerica": "north_america",
    "north_america": "north_america",
    "samerica": "south_america",
    "southamerica": "south_america",
    "south_america": "south_america",
}


def clean_region(region):
    region = str(region).strip().lower()
    region = region.replace("-", "_").replace(" ", "_")
    region = "".join(char for char in region if not char.isdigit())
    return REGION_ALIASES.get(region, region)


def load_metadata():
    path = os.path.join(os.path.dirname(__file__), METADATA_FILE)
    df = pd.read_csv(path)

    df["region"] = df["region"].apply(clean_region)
    df["layer_name"] = df["layer"].map(LAYER_LABELS)
    df["t_mean_c"] = df["temp_mean_k"] - 273.15
    df["dryness"] = df["dryness_index"]

    return df


def plot_map(ax, all_points, counts):
    ax.set_global()
    ax.add_feature(cfeature.LAND, facecolor="#F2F2F2", edgecolor="none")
    ax.add_feature(cfeature.OCEAN, facecolor="white", edgecolor="none")
    ax.add_feature(cfeature.COASTLINE, linewidth=0.6, edgecolor="0.25")
    ax.add_feature(cfeature.BORDERS, linewidth=0.4, edgecolor="0.35")

    gridlines = ax.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=False,
        linewidth=0.4,
        color="0.6",
        alpha=0.5,
        linestyle="--",
        zorder=5,
    )
    gridlines.xlocator = mticker.FixedLocator([-180, -120, -60, 0, 60, 120, 180])
    gridlines.ylocator = mticker.FixedLocator([-60, -30, 0, 30, 60])

    ax.scatter(
        all_points["lon"].to_numpy(),
        all_points["lat"].to_numpy(),
        transform=ccrs.PlateCarree(),
        s=3,
        marker="s",
        color="k",
        alpha=0.95,
        linewidths=0,
        zorder=2,
    )

    handles = [
        Line2D(
            [], [], linestyle="none", marker="s", markersize=5,
            markerfacecolor="black", markeredgecolor="none", alpha=0.9,
            label="All grid pixels (any layer): n={:,d}".format(len(all_points)),
        ),
        Line2D([], [], linestyle="none", marker=None,
               label="Layer1: n={:,d}".format(counts["Layer1 (0–10cm)"])),
        Line2D([], [], linestyle="none", marker=None,
               label="Layer2: n={:,d}".format(counts["Layer2 (10–30cm)"])),
        Line2D([], [], linestyle="none", marker=None,
               label="Layer3: n={:,d}".format(counts["Layer3 (30–50cm)"])),
    ]

    ax.legend(
        handles=handles,
        loc="lower right",
        bbox_to_anchor=(1.1, -0.1),
        frameon=True,
        framealpha=1.0,
        facecolor="white",
        edgecolor="0.4",
        fontsize=9,
        fancybox=True,
        labelspacing=0.5,
    )

    ax.set_title("(a)", fontsize=14, fontweight="bold")
    ax.title.set_position((-.2, 1.05))


def plot_scatter(ax, df, layer_name, panel_label=False):
    for region in REGION_ORDER:
        sub = df[df["region"] == region]
        if sub.empty:
            continue

        zorder = 4
        if region == "north_america":
            zorder = 2
        if region == "africa":
            zorder = 6

        ax.scatter(
            sub["dryness"],
            sub["t_mean_c"],
            s=2,
            alpha=0.9,
            marker="x",
            color=REGION_COLORS[region],
            label="{} ({})".format(REGION_LABELS[region], len(sub)),
            zorder=zorder,
        )

    ax.set_xscale("log")
    ax.set_xlabel("Dryness (Rad$_{net}$/Precip)", fontsize=10)
    ax.set_ylim(-15, 35)
    ax.set_yticks(np.arange(-10, 31, 10))
    ax.tick_params(axis="both", labelsize=9)
    ax.set_xticks([0.1, 1, 10, 100])
    ax.xaxis.set_major_formatter(lambda x, pos: "{:g}".format(x))

    if panel_label:
        ax.text(
            -.13, 1.2, "(b)", transform=ax.transAxes,
            ha="left", va="top", fontsize=14, fontweight="bold",
        )
        ax.set_ylabel("Temperature (°C)", fontsize=10)
    else:
        ax.set_ylabel("")
        ax.tick_params(labelleft=False)

    ax.legend(
        loc="lower right",
        bbox_to_anchor=(1.20, 0.00),
        fontsize=8,
        frameon=True,
        framealpha=1.0,
        facecolor="white",
        edgecolor="0.4",
        borderpad=0.4,
        labelspacing=0.3,
        handletextpad=0.4,
        markerscale=3.0,
    )

    ax.set_title(layer_name, fontsize=10)
    ax.grid(True, linewidth=0.4, alpha=0.4)


def make_figure(df):
    layer_names = list(LAYER_LABELS.values())
    layer_data = {}

    for layer_name in layer_names:
        layer_data[layer_name] = df[df["layer_name"] == layer_name].copy()

    counts = {}
    for layer_name in layer_names:
        counts[layer_name] = len(layer_data[layer_name].drop_duplicates(subset=["lat", "lon"]))

    all_points = df[["lat", "lon"]].drop_duplicates()

    fig = plt.figure(figsize=(10, 8))
    grid = fig.add_gridspec(nrows=2, ncols=3, height_ratios=[1.25, 1.0], hspace=0.35)

    ax_map = fig.add_subplot(grid[0, :], projection=ccrs.Robinson())
    plot_map(ax_map, all_points, counts)

    for i, layer_name in enumerate(layer_names):
        ax = fig.add_subplot(grid[1, i])
        plot_scatter(ax, layer_data[layer_name], layer_name, panel_label=(i == 0))

    plt.show()


def main():
    metadata = load_metadata()
    make_figure(metadata)


if __name__ == "__main__":
    main()
