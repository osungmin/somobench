#!/usr/bin/env python3

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
print("modules imported")


SCRIPT_DIR = Path(__file__).resolve().parent
FIGDATA_DIR = SCRIPT_DIR / "figdata"

PANEL_A_FILE = FIGDATA_DIR / "fig5_final_source_mean_spread.csv"
PANEL_B_FILE = FIGDATA_DIR / "fig5_final_source_overlap_corr.csv"
OUTPUT_FILE = SCRIPT_DIR / "Fig5.png"

LAYER_ORDER = ["layer1", "layer2", "layer3"]

LAYER_LABELS = {
    "layer1": "Layer 1\n(0–10 cm)",
    "layer2": "Layer 2\n(10–30 cm)",
    "layer3": "Layer 3\n(30–50 cm)",
}

PANEL_FONTSIZE = 14
X_TICK_FONTSIZE = 9
Y_LABEL_FONTSIZE = 10
Y_TICK_FONTSIZE = 9
N_TEXT_FONTSIZE = 9

PANEL_A_YLIM = (0, 80)
PANEL_A_YTICKS = np.arange(0, 81, 20)

PANEL_B_YLIM = (0, 1)
PANEL_B_YTICKS = np.arange(0, 1.01, 0.2)


def read_metrics(
    path: Path,
    required_columns: set[str],
) -> pd.DataFrame:
    """Read one released Figure 5 metric file."""
    data = pd.read_csv(path, na_values=-9999)

    missing = required_columns.difference(data.columns)
    if missing:
        raise ValueError(
            f"{path} missing required columns: {sorted(missing)}"
        )

    if data.empty:
        raise RuntimeError(f"Empty metric file: {path}")

    return data


def draw_boxplot(
    axis,
    data_arrays,
    panel_label,
    y_label,
    y_limits,
    y_ticks,
):
    """Draw one three-layer boxplot panel."""
    boxplot = axis.boxplot(
        data_arrays,
        positions=[1, 2, 3],
        widths=0.55,
        patch_artist=True,
        showfliers=False,
        whis=(20, 80),
        medianprops={
            "color": "black",
            "linewidth": 1.5,
        },
        whiskerprops={
            "color": "0.3",
            "linewidth": 1.0,
        },
        capprops={
            "color": "0.3",
            "linewidth": 1.0,
        },
    )

    for box in boxplot["boxes"]:
        box.set_facecolor("white")
        box.set_edgecolor("black")
        box.set_linewidth(1.2)

    axis.text(
        -0.12,
        1.07,
        panel_label,
        transform=axis.transAxes,
        fontsize=PANEL_FONTSIZE,
        fontweight="bold",
        ha="left",
        va="bottom",
    )

    axis.set_ylabel(y_label, fontsize=Y_LABEL_FONTSIZE)
    axis.set_xticks([1, 2, 3])
    axis.set_xticklabels(
        [LAYER_LABELS[layer] for layer in LAYER_ORDER],
        fontsize=X_TICK_FONTSIZE,
    )
    axis.tick_params(axis="x", labelsize=X_TICK_FONTSIZE)
    axis.tick_params(axis="y", labelsize=Y_TICK_FONTSIZE)
    axis.set_ylim(*y_limits)
    axis.set_yticks(y_ticks)
    axis.grid(True, axis="y", linewidth=0.4, alpha=0.3)


def add_sample_sizes(axis, arrays):
    """Add grid-layer sample sizes above one panel."""
    for position, values in enumerate(arrays, start=1):
        axis.text(
            position,
            0.95,
            f"n={len(values):,}",
            transform=axis.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=N_TEXT_FONTSIZE,
        )


def plot_figure() -> None:
    """Create and save Figure 5."""
    panel_a = read_metrics(
        PANEL_A_FILE,
        {
            "key",
            "layer",
            "relative_spread_pct",
        },
    )
    panel_b = read_metrics(
        PANEL_B_FILE,
        {
            "key",
            "layer",
            "overlap_corr_median",
        },
    )

    common_keys = set(panel_a["key"]) & set(panel_b["key"])

    panel_a = panel_a.loc[
        panel_a["key"].isin(common_keys)
    ].copy()
    panel_b = panel_b.loc[
        panel_b["key"].isin(common_keys)
    ].copy()

    if panel_a.empty or panel_b.empty:
        raise RuntimeError(
            "No common grid-layer targets found between Figure 5 metric files."
        )

    panel_a_arrays = [
        panel_a.loc[
            panel_a["layer"] == layer,
            "relative_spread_pct",
        ].dropna().to_numpy()
        for layer in LAYER_ORDER
    ]

    panel_b_arrays = [
        panel_b.loc[
            panel_b["layer"] == layer,
            "overlap_corr_median",
        ].dropna().to_numpy()
        for layer in LAYER_ORDER
    ]

    figure = plt.figure(figsize=(10, 4.2))
    grid = figure.add_gridspec(
        nrows=1,
        ncols=2,
        left=0.08,
        right=0.98,
        top=0.88,
        bottom=0.18,
        wspace=0.28,
    )

    panel_a_axis = figure.add_subplot(grid[0, 0])
    panel_b_axis = figure.add_subplot(grid[0, 1])

    draw_boxplot(
        axis=panel_a_axis,
        data_arrays=panel_a_arrays,
        panel_label="(a)",
        y_label="Relative spread of station means (%)",
        y_limits=PANEL_A_YLIM,
        y_ticks=PANEL_A_YTICKS,
    )
    add_sample_sizes(panel_a_axis, panel_a_arrays)

    draw_boxplot(
        axis=panel_b_axis,
        data_arrays=panel_b_arrays,
        panel_label="(b)",
        y_label="Primary–secondary\noverlap correlation",
        y_limits=PANEL_B_YLIM,
        y_ticks=PANEL_B_YTICKS,
    )
    add_sample_sizes(panel_b_axis, panel_b_arrays)

    figure.savefig(OUTPUT_FILE, dpi=300, bbox_inches="tight")
    plt.close(figure)

    print(f"Saved: {OUTPUT_FILE}")


def main() -> None:
    plot_figure()


if __name__ == "__main__":
    main()
