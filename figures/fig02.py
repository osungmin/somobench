#!/usr/bin/env python3

from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
print("modules imported")


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_ROOT = SCRIPT_DIR / "SoMoBench_v1.0"
METADATA_CSV = DATA_ROOT / "SoMoBench_global_metadata_v1.csv"
OUTPUT_FILE = SCRIPT_DIR / "Fig2.png"

REGION_ORDER = ["africa", "asia", "europe", "namerica", "samerica", "oceania"]

REGION_COLORS = {
    "namerica": "black",
    "europe": "goldenrod",
    "asia": "royalblue",
    "oceania": "crimson",
    "africa": "sienna",
    "samerica": "forestgreen",
}

REGION_LABELS = {
    "namerica": "N. America",
    "samerica": "S. America",
    "europe": "Europe",
    "asia": "Asia",
    "oceania": "Oceania",
    "africa": "Africa",
}


def normalize_region_name(series: pd.Series) -> pd.Series:
    """Lowercase region names and remove digits."""
    return series.astype(str).str.lower().str.replace(r"\d+", "", regex=True)


def load_layer_points(metadata: pd.DataFrame, layer_num: int) -> pd.DataFrame:
    """Load grid cells for one soil layer."""
    df = metadata.loc[metadata["layer"] == f"layer{layer_num}"].copy()

    df["region"] = normalize_region_name(df["region"])
    df["lat"] = df["lat"].astype(float)
    df["lon"] = df["lon"].astype(float)
    df["t_mean_c"] = df["temp_mean_c"].astype(float)
    df["dryness"] = df["dryness_index"].astype(float)

    return df


def compute_all_points(layer_dfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return unique grid cells represented in any layer."""
    return (
        pd.concat(layer_dfs.values(), ignore_index=True)
        .drop_duplicates(subset=["lat", "lon"])
        [["lat", "lon"]]
    )


def main() -> None:
    metadata = pd.read_csv(METADATA_CSV)

    layer_dfs = {
        "Layer1 (0–10cm)": load_layer_points(metadata, 1),
        "Layer2 (10–30cm)": load_layer_points(metadata, 2),
        "Layer3 (30–50cm)": load_layer_points(metadata, 3),
    }

    counts = {
        name: len(df.drop_duplicates(subset=["lat", "lon"]))
        for name, df in layer_dfs.items()
    }

    all_points = compute_all_points(layer_dfs)

    fig = plt.figure(figsize=(10, 8))
    grid = fig.add_gridspec(
        nrows=2,
        ncols=3,
        height_ratios=[1.25, 1.0],
        hspace=0.35,
    )

    map_axis = fig.add_subplot(grid[0, :], projection=ccrs.Robinson())
    map_axis.set_global()

    map_axis.add_feature(cfeature.LAND, facecolor="#F2F2F2", edgecolor="none")
    map_axis.add_feature(cfeature.OCEAN, facecolor="white", edgecolor="none")
    map_axis.add_feature(cfeature.COASTLINE, linewidth=0.6, edgecolor="0.25")
    map_axis.add_feature(cfeature.BORDERS, linewidth=0.4, edgecolor="0.35")

    gridlines = map_axis.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=False,
        linewidth=0.4,
        color="0.6",
        alpha=0.5,
        linestyle="--",
        zorder=5,
    )
    gridlines.xlocator = mticker.FixedLocator(
        [-180, -120, -60, 0, 60, 120, 180]
    )
    gridlines.ylocator = mticker.FixedLocator([-60, -30, 0, 30, 60])

    map_axis.scatter(
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

    legend_handles = [
        Line2D(
            [],
            [],
            linestyle="none",
            marker="s",
            markersize=5,
            markerfacecolor="black",
            markeredgecolor="none",
            alpha=0.9,
            label=f"All grid pixels (any layer): n={len(all_points):,d}",
        ),
        Line2D(
            [],
            [],
            linestyle="none",
            marker=None,
            label=f"Layer1: n={counts['Layer1 (0–10cm)']:,d}",
        ),
        Line2D(
            [],
            [],
            linestyle="none",
            marker=None,
            label=f"Layer2: n={counts['Layer2 (10–30cm)']:,d}",
        ),
        Line2D(
            [],
            [],
            linestyle="none",
            marker=None,
            label=f"Layer3: n={counts['Layer3 (30–50cm)']:,d}",
        ),
    ]

    map_axis.legend(
        handles=legend_handles,
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

    map_axis.set_title("(a)", fontsize=14, fontweight="bold")
    map_axis.title.set_position((-0.2, 1.05))

    for panel_index, (layer_name, layer_data) in enumerate(layer_dfs.items()):
        axis = fig.add_subplot(grid[1, panel_index])

        for region in REGION_ORDER:
            subset = layer_data[layer_data["region"] == region]

            if subset.empty:
                continue

            zorder = 2 if region == "namerica" else 6 if region == "africa" else 4

            axis.scatter(
                subset["dryness"],
                subset["t_mean_c"],
                s=2,
                alpha=0.9,
                marker="x",
                color=REGION_COLORS[region],
                label=f"{REGION_LABELS[region]} ({len(subset)})",
                zorder=zorder,
            )

        axis.set_xscale("log")
        axis.set_xlabel("Dryness (Rad$_{net}$/Precip)", fontsize=10)
        axis.set_ylim(-15, 35)
        axis.set_yticks(np.arange(-10, 31, 10))
        axis.tick_params(axis="both", labelsize=9)

        if panel_index == 0:
            axis.text(
                -0.13,
                1.2,
                "(b)",
                transform=axis.transAxes,
                ha="left",
                va="top",
                fontsize=14,
                fontweight="bold",
            )
            axis.set_ylabel("Temperature (°C)", fontsize=10)
        else:
            axis.tick_params(labelleft=False)

        axis.set_xticks([0.1, 1, 10, 100])
        axis.xaxis.set_major_formatter(lambda value, _: f"{value:g}")

        axis.legend(
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

        axis.set_title(layer_name, fontsize=10)
        axis.grid(True, linewidth=0.4, alpha=0.4)

    fig.savefig(OUTPUT_FILE, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
