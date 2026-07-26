#!/usr/bin/env python3

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
print("modules imported")


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_ROOT = SCRIPT_DIR / "SoMoBench_v1.0"
METADATA_CSV = DATA_ROOT / "SoMoBench_global_metadata_v1.csv"
OUTPUT_FILE = SCRIPT_DIR / "Fig4.png"

LAYER_SETTINGS = {
    "L1": {
        "layer": "layer1",
        "label": "Layer 1 (0–10 cm)",
        "nominal_midpoint_cm": 5.0,
    },
    "L2": {
        "layer": "layer2",
        "label": "Layer 2 (10–30 cm)",
        "nominal_midpoint_cm": 20.0,
    },
    "L3": {
        "layer": "layer3",
        "label": "Layer 3 (30–50 cm)",
        "nominal_midpoint_cm": 40.0,
    },
}

BAR_FACE = "#5B7C99"
BAR_EDGE = "#2F3E4D"

PRESENT_FACE = "black"
PRESENT_EDGE = "black"
ABSENT_EDGE = "#B0B0B0"
MATRIX_LINE = "black"

VIOLIN_FACE = "#D9E2EA"
VIOLIN_EDGE = "#5B7C99"
VIOLIN_ALPHA = 0.55


def read_metadata() -> pd.DataFrame:
    """Read released global metadata."""
    metadata = pd.read_csv(METADATA_CSV)

    required = {"grid_id", "layer", "representative_depth_cm"}
    missing = required.difference(metadata.columns)
    if missing:
        raise ValueError(
            f"Missing metadata columns: {', '.join(sorted(missing))}"
        )

    metadata["grid_id"] = metadata["grid_id"].astype(str)
    metadata["representative_depth_cm"] = pd.to_numeric(
        metadata["representative_depth_cm"],
        errors="coerce",
    )

    if metadata.duplicated(subset=["grid_id", "layer"]).any():
        duplicated = metadata.loc[
            metadata.duplicated(subset=["grid_id", "layer"]),
            ["grid_id", "layer"],
        ]
        raise ValueError(
            "Duplicate grid-layer records found: "
            f"{duplicated.head().to_dict('records')}"
        )

    return metadata


def load_all_layers(metadata: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Split global metadata into the three target layers."""
    return {
        label: metadata.loc[
            metadata["layer"] == settings["layer"]
        ].copy()
        for label, settings in LAYER_SETTINGS.items()
    }


def make_intersection_counts(
    layer_sets: dict[str, set[str]],
) -> tuple[dict[str, set[str]], set[str]]:
    """Return exclusive grid-cell sets for all seven layer combinations."""
    layer1 = layer_sets["L1"]
    layer2 = layer_sets["L2"]
    layer3 = layer_sets["L3"]

    combinations = {
        "L1": layer1 - layer2 - layer3,
        "L2": layer2 - layer1 - layer3,
        "L3": layer3 - layer1 - layer2,
        "L1+L2": (layer1 & layer2) - layer3,
        "L1+L3": (layer1 & layer3) - layer2,
        "L2+L3": (layer2 & layer3) - layer1,
        "L1+L2+L3": layer1 & layer2 & layer3,
    }

    return combinations, layer1 | layer2 | layer3


def choose_bar_ymax(counts: list[int]) -> int:
    """Choose a rounded upper limit for the intersection-size bars."""
    maximum = max(counts)

    if maximum <= 0:
        return 1

    step = 100 if maximum >= 500 else 50
    return int(np.ceil((maximum * 1.12) / step) * step)


def build_depth_deviation_arrays(
    metadata_by_layer: dict[str, pd.DataFrame],
) -> list[np.ndarray]:
    """Return signed depth deviations from nominal layer midpoints."""
    return [
        (
            metadata_by_layer[layer]["representative_depth_cm"]
            - LAYER_SETTINGS[layer]["nominal_midpoint_cm"]
        )
        .dropna()
        .to_numpy()
        for layer in ["L1", "L2", "L3"]
    ]


def format_cm(value: float) -> str:
    """Format a signed depth deviation for figure annotation."""
    if abs(value - round(value)) < 1e-9:
        return f"{int(round(value)):+d}"

    return f"{value:+.1f}"


def plot_figure() -> None:
    """Create and save Figure 4."""
    metadata_by_layer = load_all_layers(read_metadata())

    layer_sets = {
        layer: set(metadata["grid_id"])
        for layer, metadata in metadata_by_layer.items()
    }

    combinations, total_union = make_intersection_counts(layer_sets)

    combination_order = [
        "L1+L2+L3",
        "L1+L2",
        "L1+L3",
        "L2+L3",
        "L1",
        "L2",
        "L3",
    ]
    combination_counts = [
        len(combinations[label])
        for label in combination_order
    ]

    total_grids = len(total_union)
    n_layer1 = len(layer_sets["L1"])
    n_layer2 = len(layer_sets["L2"])
    n_layer3 = len(layer_sets["L3"])
    n_layer1_layer2 = len(layer_sets["L1"] & layer_sets["L2"])
    n_all_layers = len(
        layer_sets["L1"]
        & layer_sets["L2"]
        & layer_sets["L3"]
    )

    def percentage(count: int) -> float:
        return 0.0 if total_grids == 0 else 100.0 * count / total_grids

    depth_deviation_arrays = build_depth_deviation_arrays(
        metadata_by_layer
    )

    figure = plt.figure(figsize=(10, 8))
    grid = figure.add_gridspec(
        nrows=2,
        ncols=1,
        height_ratios=[1.45, 0.95],
        left=0.08,
        right=0.98,
        top=0.95,
        bottom=0.08,
        hspace=0.32,
    )

    panel_a = grid[0].subgridspec(
        nrows=2,
        ncols=1,
        height_ratios=[3.0, 1.4],
        hspace=0.05,
    )
    bar_axis = figure.add_subplot(panel_a[0])
    matrix_axis = figure.add_subplot(panel_a[1], sharex=bar_axis)

    x_positions = np.arange(len(combination_order))

    bars = bar_axis.bar(
        x_positions,
        combination_counts,
        color=BAR_FACE,
        edgecolor=BAR_EDGE,
        linewidth=0.6,
        alpha=0.9,
    )

    bar_axis.set_ylabel("Number of grid cells", fontsize=10)
    bar_axis.tick_params(axis="y", labelsize=9)
    bar_axis.tick_params(
        axis="x",
        which="both",
        bottom=False,
        labelbottom=False,
    )
    bar_axis.set_ylim(0, choose_bar_ymax(combination_counts))

    label_offset = max(combination_counts) * 0.02

    for bar, count in zip(bars, combination_counts):
        bar_axis.text(
            bar.get_x() + bar.get_width() / 2,
            count + label_offset,
            f"n={count:,}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    inset_text = (
        f"Total grid cells: {total_grids:,}\n"
        f"L1: {n_layer1:,}, L2: {n_layer2:,}, L3: {n_layer3:,}\n"
        f"With both L1 and L2: {n_layer1_layer2:,} "
        f"({percentage(n_layer1_layer2):.1f}%)\n"
        f"With all three layers: {n_all_layers:,} "
        f"({percentage(n_all_layers):.1f}%)"
    )

    bar_axis.text(
        0.98,
        0.95,
        inset_text,
        transform=bar_axis.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        linespacing=1.35,
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": "white",
            "edgecolor": "black",
            "linewidth": 0.8,
        },
    )

    bar_axis.text(
        -0.08,
        1.05,
        "(a)",
        transform=bar_axis.transAxes,
        fontsize=14,
        fontweight="bold",
        ha="left",
        va="bottom",
    )

    row_labels = ["L1", "L2", "L3"]
    row_positions = {"L1": 2, "L2": 1, "L3": 0}

    matrix_axis.set_ylim(-0.7, 2.7)
    matrix_axis.set_yticks(
        [row_positions[label] for label in row_labels]
    )
    matrix_axis.set_yticklabels(row_labels, fontsize=10)
    matrix_axis.set_xticks(x_positions)
    matrix_axis.set_xticklabels(combination_order, fontsize=10)
    matrix_axis.tick_params(axis="both", labelsize=9)

    for column_index, label in enumerate(combination_order):
        present_rows = []

        for row_label in row_labels:
            y_position = row_positions[row_label]
            is_present = row_label in label

            if is_present:
                matrix_axis.scatter(
                    column_index,
                    y_position,
                    s=38,
                    facecolors=PRESENT_FACE,
                    edgecolors=PRESENT_EDGE,
                    linewidths=0.8,
                    zorder=3,
                )
                present_rows.append(y_position)
            else:
                matrix_axis.scatter(
                    column_index,
                    y_position,
                    s=38,
                    facecolors="none",
                    edgecolors=ABSENT_EDGE,
                    linewidths=0.9,
                    zorder=2,
                )

        if len(present_rows) >= 2:
            matrix_axis.plot(
                [column_index, column_index],
                [min(present_rows), max(present_rows)],
                color=MATRIX_LINE,
                linewidth=1.3,
                zorder=1,
            )

    matrix_axis.spines["top"].set_visible(False)
    matrix_axis.spines["right"].set_visible(False)

    depth_axis = figure.add_subplot(grid[1])

    violin_parts = depth_axis.violinplot(
        depth_deviation_arrays,
        positions=[1, 2, 3],
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )

    for body in violin_parts["bodies"]:
        body.set_facecolor(VIOLIN_FACE)
        body.set_edgecolor(VIOLIN_EDGE)
        body.set_alpha(VIOLIN_ALPHA)
        body.set_linewidth(0.9)

    depth_axis.axhline(
        0,
        color=VIOLIN_EDGE,
        linewidth=1.0,
        alpha=0.8,
    )

    depth_axis.set_xticks([1, 2, 3])
    depth_axis.set_xticklabels(
        [
            LAYER_SETTINGS["L1"]["label"],
            LAYER_SETTINGS["L2"]["label"],
            LAYER_SETTINGS["L3"]["label"],
        ]
    )
    depth_axis.set_ylabel(
        "Representative-depth deviation (cm)",
        fontsize=10,
    )
    depth_axis.tick_params(axis="both", labelsize=9)
    depth_axis.set_ylim(-12, 12)

    annotation_y = -6

    for position, values in enumerate(
        depth_deviation_arrays,
        start=1,
    ):
        values = np.asarray(values, dtype=float)
        values = values[np.isfinite(values)]

        if len(values) == 0:
            continue

        mode_value = float(
            pd.Series(np.round(values, 2)).value_counts().idxmax()
        )

        depth_axis.text(
            position + 0.1,
            annotation_y,
            f"n={len(values)}\nmode={format_cm(mode_value)}",
            fontsize=10,
            ha="left",
            va="top",
        )

    depth_axis.text(
        -0.08,
        1.03,
        "(b)",
        transform=depth_axis.transAxes,
        fontsize=14,
        fontweight="bold",
        ha="left",
        va="bottom",
    )

    figure.savefig(OUTPUT_FILE, dpi=300, bbox_inches="tight")
    plt.close(figure)

    print(f"Saved: {OUTPUT_FILE}")


def main() -> None:
    plot_figure()


if __name__ == "__main__":
    main()
