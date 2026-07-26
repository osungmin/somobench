#!/usr/bin/env python3

import calendar
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colors
from matplotlib.colors import Normalize
print("modules imported")


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_ROOT = SCRIPT_DIR / "SoMoBench_v1.0"
METADATA_CSV = DATA_ROOT / "SoMoBench_global_metadata_v1.csv"
TARGET_ROOT = DATA_ROOT / "target"
OUTPUT_FILE = SCRIPT_DIR / "Fig3.png"

LAYER_SETTINGS = {
    "layer1": {
        "label": "Layer 1 (0–10 cm)",
        "short_label": "Layer 1",
        "data_dir": TARGET_ROOT / "layer1",
        "suffix": "l1",
    },
    "layer2": {
        "label": "Layer 2 (10–30 cm)",
        "short_label": "Layer 2",
        "data_dir": TARGET_ROOT / "layer2",
        "suffix": "l2",
    },
    "layer3": {
        "label": "Layer 3 (30–50 cm)",
        "short_label": "Layer 3",
        "data_dir": TARGET_ROOT / "layer3",
        "suffix": "l3",
    },
}


def read_metadata() -> pd.DataFrame:
    """Read released global metadata."""
    metadata = pd.read_csv(METADATA_CSV)

    required = {"grid_id", "layer"}
    missing = required.difference(metadata.columns)
    if missing:
        raise ValueError(
            f"Missing metadata columns: {', '.join(sorted(missing))}"
        )

    return metadata


def target_filename(grid_id: str, suffix: str) -> str:
    """Build a released target filename from its grid ID and layer."""
    return f"IDX_{grid_id}_{suffix}.dat"


def read_target_series(
    path: Path,
    start_year: int,
    end_year: int,
) -> pd.Series:
    """Read valid daily soil-moisture values from one target file."""
    data = pd.read_csv(
        path,
        index_col=0,
        parse_dates=True,
        na_values=-9999,
    )

    if "soilm" not in data.columns:
        raise ValueError(f"'soilm' column missing: {path}")

    soilm = pd.to_numeric(data["soilm"], errors="coerce")
    soilm = soilm[
        (soilm.index.year >= start_year)
        & (soilm.index.year <= end_year)
    ].sort_index()

    if soilm.index.duplicated().any():
        duplicated = soilm.index[soilm.index.duplicated()].unique()
        raise ValueError(
            f"Duplicate dates found in {path}: "
            f"{duplicated[:5].astype(str).tolist()}"
        )

    soilm = soilm.dropna()

    if (soilm < 0).any():
        raise ValueError(f"Negative soil moisture found in: {path}")

    return soilm


def build_temporal_availability(
    metadata: pd.DataFrame,
    layer: str,
    data_dir: Path,
    suffix: str,
    start_year: int,
    end_year: int,
) -> tuple[np.ndarray, np.ma.MaskedArray, np.ndarray]:
    """Calculate monthly completeness and annual active-grid counts."""
    layer_metadata = metadata.loc[metadata["layer"] == layer]

    years = np.arange(start_year, end_year + 1)
    observed_days = np.zeros((12, len(years)), dtype=np.int64)
    possible_days = np.zeros((12, len(years)), dtype=np.int64)
    yearly_active_grids = np.zeros(len(years), dtype=np.int32)

    for file_index, grid_id in enumerate(
        layer_metadata["grid_id"].astype(str)
    ):
        if file_index % 250 == 0:
            print(f"  {file_index}/{len(layer_metadata)} | {layer}")

        target_file = data_dir / target_filename(grid_id, suffix)

        if not target_file.exists():
            raise FileNotFoundError(f"Missing target file: {target_file}")

        soilm = read_target_series(
            target_file,
            start_year=start_year,
            end_year=end_year,
        )

        if soilm.empty:
            continue

        valid_dates = pd.DatetimeIndex(soilm.index).normalize()
        first_valid = valid_dates.min()
        last_valid = valid_dates.max()

        for year in np.unique(valid_dates.year):
            yearly_active_grids[year - start_year] += 1

        observed_counts = (
            pd.Series(1, index=valid_dates)
            .groupby([valid_dates.year, valid_dates.month])
            .sum()
        )

        first_year = max(start_year, first_valid.year)
        last_year = min(end_year, last_valid.year)

        for year in range(first_year, last_year + 1):
            for month in range(1, 13):
                month_start = pd.Timestamp(year=year, month=month, day=1)
                month_end = pd.Timestamp(
                    year=year,
                    month=month,
                    day=calendar.monthrange(year, month)[1],
                )

                overlap_start = max(first_valid, month_start)
                overlap_end = min(last_valid, month_end)

                if overlap_start > overlap_end:
                    continue

                year_index = year - start_year
                month_index = month - 1

                possible_days[month_index, year_index] += (
                    overlap_end - overlap_start
                ).days + 1
                observed_days[month_index, year_index] += int(
                    observed_counts.get((year, month), 0)
                )

    with np.errstate(divide="ignore", invalid="ignore"):
        completeness = observed_days / possible_days.astype(float)

    completeness = np.ma.masked_where(possible_days == 0, completeness)

    return years, completeness, yearly_active_grids


def make_truncated_viridis() -> colors.LinearSegmentedColormap:
    """Create the lightened viridis colormap used in the original figure."""
    base = plt.get_cmap("viridis")
    cmap = colors.LinearSegmentedColormap.from_list(
        "viridis_truncated",
        base(np.linspace(0.3, 1.0, 256)),
    )
    cmap.set_bad("white")
    return cmap


def choose_active_grid_ymax(
    all_yearly_counts: dict[str, np.ndarray],
) -> int:
    """Choose a rounded upper y-axis limit from the plotted data."""
    maximum = max(
        int(np.max(values))
        for values in all_yearly_counts.values()
    )

    if maximum <= 0:
        return 1

    return int(np.ceil((maximum * 1.08) / 100.0) * 100)


def plot_figure(
    start_year: int = 2001,
    end_year: int = 2024,
    active_grid_ymax: int | None = 1200,
) -> None:
    """Create and save Figure 3."""
    metadata = read_metadata()

    month_labels = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ]

    fig = plt.figure(figsize=(10, 8))
    grid = fig.add_gridspec(
        nrows=2,
        ncols=3,
        height_ratios=[1.25, 1.0],
        hspace=0.35,
    )

    heatmap_axes = [
        fig.add_subplot(grid[0, column])
        for column in range(3)
    ]
    timeseries_axis = fig.add_subplot(grid[1, :])

    cmap = make_truncated_viridis()
    colorbar_image = None
    all_years = None
    all_yearly_counts = {}

    for panel_index, (layer, settings) in enumerate(
        LAYER_SETTINGS.items()
    ):
        years, completeness, yearly_active_grids = (
            build_temporal_availability(
                metadata=metadata,
                layer=layer,
                data_dir=settings["data_dir"],
                suffix=settings["suffix"],
                start_year=start_year,
                end_year=end_year,
            )
        )

        all_years = years
        all_yearly_counts[layer] = yearly_active_grids
        axis = heatmap_axes[panel_index]

        image = axis.imshow(
            np.ma.masked_invalid(completeness),
            aspect="auto",
            origin="upper",
            extent=[
                start_year - 0.5,
                end_year + 0.5,
                12.5,
                0.5,
            ],
            norm=Normalize(vmin=0.0, vmax=1.0),
            cmap=cmap,
        )

        if colorbar_image is None:
            colorbar_image = image

        axis.set_xlim(start_year - 0.5, end_year + 0.5)
        axis.set_ylim(12.5, 0.5)
        axis.set_xticks([2005, 2010, 2015, 2020, 2024])
        axis.set_xticklabels(
            ["2005", "2010", "2015", "2020", "2024"],
            fontsize=9,
        )
        axis.set_yticks(np.arange(1, 13))

        if panel_index == 0:
            axis.set_yticklabels(month_labels, fontsize=9)
            axis.set_ylabel("Month", fontsize=10)
            axis.text(
                -0.2,
                1.15,
                "(a)",
                transform=axis.transAxes,
                ha="left",
                va="top",
                fontsize=14,
                fontweight="bold",
            )
        else:
            axis.set_yticklabels([])

        for year in range(
            ((start_year + 4) // 5) * 5,
            end_year + 1,
            5,
        ):
            axis.axvline(year + 0.5, linewidth=0.5, alpha=0.25)

        for month_line in np.arange(0.5, 12.6, 1.0):
            axis.axhline(month_line, linewidth=0.5, alpha=0.18)

        axis.set_title(settings["label"], fontsize=10)
        axis.set_xlabel("Year", fontsize=10)
        axis.tick_params(axis="both", labelsize=9)

    colorbar_axis = fig.add_axes([0.93, 0.55, 0.015, 0.3])
    colorbar = fig.colorbar(colorbar_image, cax=colorbar_axis)
    colorbar.ax.tick_params(labelsize=9)
    colorbar.set_label("Data completeness", fontsize=10)
    colorbar.set_ticks([0.0, 0.25, 0.5, 0.75, 1.0])
    colorbar.set_ticklabels(["0", "0.25", "0.5", "0.75", "1"])

    line_styles = {
        "layer1": {"color": "black", "linewidth": 2.1},
        "layer2": {"color": "0.4", "linewidth": 1.8},
        "layer3": {"color": "0.65", "linewidth": 1.2},
    }

    for layer, yearly_counts in all_yearly_counts.items():
        timeseries_axis.plot(
            all_years,
            yearly_counts,
            label=LAYER_SETTINGS[layer]["short_label"],
            marker="o",
            markersize=3,
            **line_styles[layer],
        )

    timeseries_axis.text(
        -0.06,
        1.1,
        "(b)",
        transform=timeseries_axis.transAxes,
        fontsize=14,
        fontweight="bold",
    )
    timeseries_axis.set_xlabel("Year", fontsize=10)
    timeseries_axis.set_ylabel(
        "Grid cells with valid targets",
        fontsize=10,
    )
    timeseries_axis.tick_params(axis="both", labelsize=9)
    timeseries_axis.tick_params(axis="x", which="major", length=6)
    timeseries_axis.set_xlim(start_year, end_year)

    if active_grid_ymax is None:
        active_grid_ymax = choose_active_grid_ymax(all_yearly_counts)

    timeseries_axis.set_ylim(0, active_grid_ymax)
    timeseries_axis.legend(
        fontsize=10,
        ncols=1,
        frameon=False,
        loc="upper left",
    )

    fig.savefig(OUTPUT_FILE, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {OUTPUT_FILE}")


def main() -> None:
    plot_figure()


if __name__ == "__main__":
    main()
