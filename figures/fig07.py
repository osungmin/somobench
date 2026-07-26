#!/usr/bin/env python3

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
print("modules imported")


SCRIPT_DIR = Path(__file__).resolve().parent
CSV_PATH = (
    SCRIPT_DIR
    / "SoMoBench_v1.0"
    / "SoMoBench_baseline_performance_v1.csv"
)
OUTPUT_FILE = SCRIPT_DIR / "Fig7.png"

METRIC = "corr"

FONTSIZE_NOTE = 9

BOX_WIDTH = 0.45
LINE_WIDTH = 1.2
MEDIAN_WIDTH = 1.8

BOX_EDGE = "#2F3A4A"
MEDIAN_COLOR = "#2F3A4A"

GRID_ALPHA = 0.35
SHADE_ALPHA = 0.05

NOTE_BBOX = {
    "boxstyle": "round,pad=0.2",
    "facecolor": "white",
    "edgecolor": "none",
    "alpha": 0.8,
}

MODEL_COLUMNS = {
    "climatology": f"clim_{METRIC}",
    "ridge_forcing": f"ridge_{METRIC}",
    "ridge_memory": f"ridge_mem_{METRIC}",
    "ridge_state": f"ridge_state_{METRIC}",
}


def read_results() -> pd.DataFrame:
    """Read released baseline performance results."""
    data = pd.read_csv(CSV_PATH)

    required = {"grid_id", "layer", "model", METRIC}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(
            f"{CSV_PATH} missing required columns: {sorted(missing)}"
        )

    unknown_models = set(data["model"].dropna()) - set(MODEL_COLUMNS)
    if unknown_models:
        raise ValueError(
            f"Unexpected model names: {sorted(unknown_models)}"
        )

    return (
        data.pivot(
            index=["grid_id", "layer"],
            columns="model",
            values=METRIC,
        )
        .reset_index()
        .rename(columns=MODEL_COLUMNS)
    )


def finite_values(series: pd.Series) -> np.ndarray:
    """Return finite numeric values."""
    values = np.asarray(series, dtype=float)
    return values[np.isfinite(values)]


def draw_boxplots(
    axis,
    series_list,
    positions,
    labels,
) -> None:
    """Draw one layer panel."""
    axis.boxplot(
        [finite_values(series) for series in series_list],
        positions=positions,
        widths=BOX_WIDTH,
        showfliers=False,
        patch_artist=False,
        medianprops={
            "linewidth": MEDIAN_WIDTH,
            "color": MEDIAN_COLOR,
        },
        boxprops={
            "linewidth": LINE_WIDTH,
            "color": BOX_EDGE,
        },
        whiskerprops={
            "linewidth": LINE_WIDTH,
            "color": BOX_EDGE,
        },
        capprops={
            "linewidth": LINE_WIDTH,
            "color": BOX_EDGE,
        },
    )

    axis.set_xticks(positions)
    axis.set_xticklabels(labels)
    axis.tick_params(axis="both", labelsize=9)
    axis.set_ylim(-0.5, 1.02)
    axis.set_yticks(np.arange(-0.5, 1.01, 0.5))
    axis.grid(
        True,
        axis="y",
        linewidth=0.7,
        alpha=GRID_ALPHA,
    )
    axis.set_axisbelow(True)


def plot_figure(data: pd.DataFrame) -> None:
    """Create and save Figure 7."""
    layers = sorted(int(layer) for layer in data["layer"].dropna().unique())

    figure, axes = plt.subplots(
        nrows=3,
        ncols=1,
        figsize=(5.5, 5.5),
        sharey=True,
        constrained_layout=False,
    )

    figure.subplots_adjust(
        left=0.12,
        right=0.97,
        top=0.97,
        bottom=0.13,
        hspace=0.25,
    )

    for axis in axes:
        axis.set_visible(False)

    positions = [1, 2, 3, 4]
    labels = [
        "Climatology",
        "Forcing-only",
        "Forcing+Memory",
        "State-augmented\n(subset only)",
    ]

    for panel_index, layer in enumerate(layers[:3]):
        subset = data.loc[data["layer"] == layer]

        climatology = subset[f"clim_{METRIC}"].to_numpy(dtype=float)
        forcing = subset[f"ridge_{METRIC}"].to_numpy(dtype=float)
        memory = subset[f"ridge_mem_{METRIC}"].to_numpy(dtype=float)
        state = subset[f"ridge_state_{METRIC}"].to_numpy(dtype=float)

        axis = axes[panel_index]
        axis.set_visible(True)
        axis.axvspan(
            3.5,
            4.5,
            color="gray",
            alpha=SHADE_ALPHA,
            zorder=0,
        )

        draw_boxplots(
            axis,
            [climatology, forcing, memory, state],
            positions,
            labels,
        )

        n_all_models = len(subset)
        n_state = int(np.isfinite(state).sum())

        axis.text(
            0.25,
            0.15,
            f"n = {n_all_models} (all 3 models)",
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=FONTSIZE_NOTE,
            color="#444444",
            bbox=NOTE_BBOX,
            zorder=5,
        )

        if layer == 1 and n_state == 0:
            axis.text(
                0.87,
                0.25,
                "N/A\n (no upper layer)",
                transform=axis.transAxes,
                ha="center",
                va="top",
                fontsize=FONTSIZE_NOTE,
                color="#444444",
                bbox=NOTE_BBOX,
                linespacing=1.15,
                zorder=5,
            )
        else:
            axis.text(
                0.93,
                0.15,
                f"n={n_state}",
                transform=axis.transAxes,
                ha="right",
                va="top",
                fontsize=FONTSIZE_NOTE,
                color="#444444",
                bbox=NOTE_BBOX,
                linespacing=1.15,
                zorder=5,
            )

        axis.set_ylabel(f"L{layer}  Corr [-]", fontsize=10)

        if panel_index < 2:
            axis.tick_params(axis="x", labelbottom=False)

    figure.savefig(
        OUTPUT_FILE,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(figure)

    print(f"Saved: {OUTPUT_FILE}")


def main() -> None:
    plot_figure(read_results())


if __name__ == "__main__":
    main()
