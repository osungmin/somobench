#!/usr/bin/env python3
"""
Build final SoMoBench soil-moisture targets for the public layer-1 demo.

Inputs
------
- demo_data/raw_daily/{layer}/{fname}_{layer}.dat
- demo_data/meta/ismn_point_daily_{region}_{layer}.lst
- demo_data/era5/{layer}_{lat}_{lon}.dat

Outputs
-------
- demo_data/target/{layer}/IDX_{lat}_{lon}_{layer_short}.dat
- demo_data/meta/target_{layer}.lst

The target files contain:
- soilm: adjusted soil moisture
- is_gapfilled: 0 for the primary station and 1 for secondary-station filling

No intermediate, internal-source, figure, or table files are written.
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd


# ======================================================
# PATHS AND DEMO SETTINGS
# ======================================================
SCRIPT_DIR = Path(__file__).resolve().parent
DEMO_DATA_DIR = SCRIPT_DIR / "demo_data"

DAILY_ROOT = DEMO_DATA_DIR / "raw_daily"
ERA5_ROOT = DEMO_DATA_DIR / "era5"
TARGET_ROOT = DEMO_DATA_DIR / "target"
META_ROOT = DEMO_DATA_DIR / "meta"

LAYERS = ["layer1"]
REGIONS = ["demo"]
START_YEAR = 2001
END_YEAR = 2024

MIN_VALID_DAYS = 90
MIN_ADJUST_OVERLAP = 90
MIN_SECONDARY_OVERLAP = 90
MIN_SECONDARY_CORR = 0.30

CLIP_NEGATIVE = True
CLEAR_OLD_TARGETS = True
NA_REP = -9999


# ======================================================
# HELPERS
# ======================================================
def normalize_region(region):
    """Remove trailing partition numbers, e.g. namerica1 -> namerica."""
    return re.sub(r"\d+$", "", str(region).strip().lower())


def grid_id(lat, lon):
    """Return the grid identifier used in the file names."""
    return f"{float(lat)}_{float(lon)}"


def layer_short_name(layer):
    return {
        "layer1": "l1",
        "layer2": "l2",
        "layer3": "l3",
    }[layer]


def nominal_midpoint(layer):
    return {
        "layer1": 5.0,
        "layer2": 20.0,
        "layer3": 40.0,
    }[layer]


def clean_depth(value):
    """Format a depth value without unnecessary decimal zeros."""
    value = float(value)

    if value.is_integer():
        return str(int(value))

    return f"{value:g}"


def depth_key(depth0, depth1):
    return f"{clean_depth(depth0)}-{clean_depth(depth1)}"


def target_name(grid, layer):
    return f"IDX_{grid}_{layer_short_name(layer)}.dat"


def safe_std(values, ddof=1):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) <= ddof:
        return np.nan

    return float(np.nanstd(values, ddof=ddof))


def safe_corr(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]

    if len(x) < 2:
        return np.nan

    if np.nanstd(x) == 0 or np.nanstd(y) == 0:
        return np.nan

    return float(np.corrcoef(x, y)[0, 1])


def join_unique(values):
    """Return sorted unique non-empty values joined by semicolons."""
    cleaned = {
        str(value).strip()
        for value in values
        if pd.notna(value)
        and str(value).strip() not in ("", "nan")
    }

    return ";".join(sorted(cleaned))


def read_series(path, column, start_year, end_year):
    """Read one dated series and restrict it to the processing period."""
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    frame = pd.read_csv(
        path,
        index_col=0,
        parse_dates=True,
        na_values=-9999,
    )

    if column not in frame.columns:
        raise ValueError(
            f"Column '{column}' is missing: {path}"
        )

    series = pd.to_numeric(
        frame[column],
        errors="coerce",
    )

    series = series[
        (series.index.year >= start_year)
        & (series.index.year <= end_year)
    ]

    return series.sort_index()


def clear_target_files(directory):
    """Remove old target files before rebuilding."""
    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    for path in directory.glob("*.dat"):
        if path.name.startswith("._"):
            continue

        path.unlink(missing_ok=True)


# ======================================================
# STEP 1: ADJUST STATIONS TO ERA5
# ======================================================
def adjust_to_era5(
    station,
    model,
    min_overlap_days,
    clip_negative,
):
    """
    Match a station series to the ERA5 mean and standard deviation.

    Statistics are calculated using dates where both series are available.
    The transformation is then applied to the complete station series.
    """
    overlap = pd.concat(
        [station, model],
        axis=1,
        sort=False,
    )

    overlap.columns = [
        "station",
        "model",
    ]

    overlap = overlap.dropna()

    if len(overlap) < min_overlap_days:
        return None

    station_mean = float(
        overlap["station"].mean()
    )

    model_mean = float(
        overlap["model"].mean()
    )

    station_std = safe_std(
        overlap["station"],
        ddof=1,
    )

    model_std = safe_std(
        overlap["model"],
        ddof=1,
    )

    if (
        not np.isfinite(station_std)
        or not np.isfinite(model_std)
    ):
        return None

    if station_std <= 0 or model_std <= 0:
        return None

    adjusted = model_mean + (
        (station - station_mean)
        * (model_std / station_std)
    )

    if clip_negative:
        adjusted = adjusted.where(
            adjusted >= 0,
            np.nan,
        )

    return adjusted


def load_adjusted_stations(layer):
    """
    Adjust eligible stations and retain the results in memory.

    No adjusted station files or adjusted metadata files are written.
    """
    metadata_rows = []
    adjusted_by_fname = {}

    daily_dir = DAILY_ROOT / layer

    for region in REGIONS:
        metadata_file = (
            META_ROOT
            / f"ismn_point_daily_{region}_{layer}.lst"
        )

        if not metadata_file.exists():
            raise FileNotFoundError(
                "Missing daily-station metadata: "
                f"{metadata_file}"
            )

        metadata = pd.read_csv(metadata_file)

        print(
            f"\nSTEP 1 | {layer} | {region} "
            f"| stations: {len(metadata)}"
        )

        kept = 0

        for row_number, row in metadata.iterrows():
            if row_number % 100 == 0:
                print(
                    f"  {row_number}/{len(metadata)}"
                )

            fname = str(row["fname"])

            grid = grid_id(
                row["lat_d25"],
                row["lon_d25"],
            )

            station_file = (
                daily_dir
                / f"{fname}_{layer}.dat"
            )

            model_file = (
                ERA5_ROOT
                / f"{layer}_{grid}.dat"
            )

            station = read_series(
                station_file,
                "soilm",
                START_YEAR,
                END_YEAR,
            )

            model = read_series(
                model_file,
                layer,
                START_YEAR,
                END_YEAR,
            )

            adjusted = adjust_to_era5(
                station=station,
                model=model,
                min_overlap_days=MIN_ADJUST_OVERLAP,
                clip_negative=CLIP_NEGATIVE,
            )

            if adjusted is None:
                continue

            valid_days = int(
                adjusted.notna().sum()
            )

            if valid_days < MIN_VALID_DAYS:
                continue

            # Match the precision previously used by
            # the intermediate adjusted-station files.
            adjusted_by_fname[fname] = (
                adjusted.round(5)
            )

            metadata_rows.append(
                row.to_dict()
            )

            kept += 1

        print(
            "  Eligible adjusted stations: "
            f"{kept} of {len(metadata)}"
        )

    adjusted_metadata = pd.DataFrame(
        metadata_rows
    )

    if adjusted_metadata.empty:
        raise RuntimeError(
            "No adjusted station records were "
            f"available for {layer}"
        )

    adjusted_metadata["grid"] = (
        adjusted_metadata.apply(
            lambda row: grid_id(
                row["lat_d25"],
                row["lon_d25"],
            ),
            axis=1,
        )
    )

    adjusted_metadata["depth_interval"] = (
        adjusted_metadata.apply(
            lambda row: depth_key(
                row["depth0"],
                row["depth1"],
            ),
            axis=1,
        )
    )

    return (
        adjusted_metadata,
        adjusted_by_fname,
    )


# ======================================================
# STEP 2: BUILD GRID-DEPTH CANDIDATES IN MEMORY
# ======================================================
def station_table(
    metadata,
    adjusted_by_fname,
):
    """
    Return station information sorted by descending valid length.

    The station with the longest record becomes the primary station.
    """
    rows = []
    series_by_fname = {}

    for _, row in metadata.iterrows():
        fname = str(row["fname"])

        series = adjusted_by_fname.get(fname)

        if series is None:
            continue

        valid_days = int(
            series.notna().sum()
        )

        if valid_days == 0:
            continue

        series_by_fname[fname] = series

        rows.append(
            {
                "fname": fname,
                "network": str(
                    row.get("network", "")
                ),
                "station": str(
                    row.get("station", "")
                ),
                "sensor": str(
                    row.get("sensor", "")
                ),
                "valid_days": valid_days,
            }
        )

    info = pd.DataFrame(rows)

    if info.empty:
        return info, series_by_fname

    info = info.sort_values(
        [
            "valid_days",
            "fname",
        ],
        ascending=[
            False,
            True,
        ],
    ).reset_index(drop=True)

    return info, series_by_fname


def match_secondary(
    primary,
    secondary,
):
    """
    Transform and validate one secondary station against the primary.
    """
    overlap = pd.concat(
        [
            primary,
            secondary,
        ],
        axis=1,
        sort=False,
    )

    overlap.columns = [
        "primary",
        "secondary",
    ]

    overlap = overlap.dropna()

    if len(overlap) < MIN_SECONDARY_OVERLAP:
        return None

    primary_mean = float(
        overlap["primary"].mean()
    )

    secondary_mean = float(
        overlap["secondary"].mean()
    )

    primary_std = safe_std(
        overlap["primary"],
        ddof=1,
    )

    secondary_std = safe_std(
        overlap["secondary"],
        ddof=1,
    )

    if (
        not np.isfinite(primary_std)
        or not np.isfinite(secondary_std)
    ):
        return None

    if primary_std <= 0 or secondary_std <= 0:
        return None

    matched = primary_mean + (
        (secondary - secondary_mean)
        * (primary_std / secondary_std)
    )

    if CLIP_NEGATIVE:
        matched = matched.where(
            matched >= 0,
            np.nan,
        )

    valid_overlap = pd.concat(
        [
            primary,
            matched,
        ],
        axis=1,
        sort=False,
    )

    valid_overlap.columns = [
        "primary",
        "matched",
    ]

    valid_overlap = (
        valid_overlap.dropna()
    )

    if (
        len(valid_overlap)
        < MIN_SECONDARY_OVERLAP
    ):
        return None

    corr = safe_corr(
        valid_overlap["primary"],
        valid_overlap["matched"],
    )

    if not np.isfinite(corr):
        return None

    if corr < MIN_SECONDARY_CORR:
        return None

    return {
        "series": matched,
        "corr": corr,
        "overlap_days": int(
            len(valid_overlap)
        ),
    }


def build_candidate(
    metadata,
    adjusted_by_fname,
    layer,
    grid,
    interval,
):
    """
    Build one completed series for a grid and sensor-depth interval.

    Candidate files and candidate metadata are retained only in memory.
    """
    info, series_by_fname = station_table(
        metadata,
        adjusted_by_fname,
    )

    if info.empty:
        return None

    primary_info = info.iloc[0]

    primary_fname = (
        primary_info["fname"]
    )

    primary = series_by_fname[
        primary_fname
    ]

    all_series = [
        series_by_fname[name]
        for name in info["fname"]
    ]

    combined_index = pd.concat(
        all_series,
        axis=1,
        sort=False,
    ).sort_index().index

    final = primary.reindex(
        combined_index
    ).copy()

    gap_flag = pd.Series(
        np.nan,
        index=combined_index,
        dtype=float,
    )

    gap_flag.loc[
        final.notna()
    ] = 0

    accepted = []

    for row_number in range(
        1,
        len(info),
    ):
        secondary_info = info.iloc[
            row_number
        ]

        fname = secondary_info[
            "fname"
        ]

        secondary = series_by_fname[
            fname
        ].reindex(combined_index)

        match = match_secondary(
            primary.reindex(
                combined_index
            ),
            secondary,
        )

        if match is None:
            continue

        accepted.append(
            {
                "fname": fname,
                "network": secondary_info[
                    "network"
                ],
                "station": secondary_info[
                    "station"
                ],
                "series": match[
                    "series"
                ].reindex(combined_index),
                "corr": match["corr"],
                "overlap_days": match[
                    "overlap_days"
                ],
                "valid_days": int(
                    secondary.notna().sum()
                ),
                "used_days": 0,
            }
        )

    accepted.sort(
        key=lambda item: (
            -item["corr"],
            -item["overlap_days"],
            -item["valid_days"],
            item["fname"],
        )
    )

    for secondary in accepted:
        fill = (
            final.isna()
            & secondary["series"].notna()
        )

        secondary["used_days"] = int(
            fill.sum()
        )

        if secondary["used_days"] == 0:
            continue

        final.loc[fill] = (
            secondary["series"].loc[fill]
        )

        gap_flag.loc[fill] = 1

    output = pd.DataFrame(
        {
            "soilm": final,
            "is_gapfilled": gap_flag,
        }
    )

    output = output.loc[
        output["soilm"].notna()
    ].copy()

    output["soilm"] = (
        output["soilm"].round(5)
    )

    output["is_gapfilled"] = (
        output["is_gapfilled"].astype(int)
    )

    used = [
        item
        for item in accepted
        if item["used_days"] > 0
    ]

    depth0 = float(
        metadata["depth0"].iloc[0]
    )

    depth1 = float(
        metadata["depth1"].iloc[0]
    )

    representative_depth = (
        depth0 + depth1
    ) / 2.0

    length = int(len(output))

    gapfilled_days = int(
        (
            output["is_gapfilled"] == 1
        ).sum()
    )

    primary_days = int(
        (
            output["is_gapfilled"] == 0
        ).sum()
    )

    summary = {
        "region": normalize_region(
            metadata["region"].iloc[0]
        ),
        "grid": grid,
        "lat_d25": float(
            metadata["lat_d25"].iloc[0]
        ),
        "lon_d25": float(
            metadata["lon_d25"].iloc[0]
        ),
        "depth0": depth0,
        "depth1": depth1,
        "representative_depth": (
            representative_depth
        ),
        "depth_deviation": abs(
            representative_depth
            - nominal_midpoint(layer)
        ),
        "length": length,
        "start_date": (
            output.index.min().strftime(
                "%Y-%m-%d"
            )
        ),
        "end_date": (
            output.index.max().strftime(
                "%Y-%m-%d"
            )
        ),
        "n_primary_days": primary_days,
        "n_gapfilled_days": gapfilled_days,
        "gapfilled_fraction": (
            gapfilled_days / length
        ),
        "primary_network": (
            primary_info["network"]
        ),
        "primary_station": (
            primary_info["station"]
        ),
        "primary_sensor": (
            primary_info["sensor"]
        ),
        "primary_fname": primary_fname,
        "n_station_candidates": int(
            len(info)
        ),
        "n_secondary_accepted": int(
            len(accepted)
        ),
        "n_secondary_used": int(
            len(used)
        ),
        "networks_used": join_unique(
            [primary_info["network"]]
            + [
                item["network"]
                for item in used
            ]
        ),
        "stations_used": join_unique(
            [primary_info["station"]]
            + [
                item["station"]
                for item in used
            ]
        ),
        "candidate_name": (
            f"IDX_{grid}_dep"
            f"{interval}.dat"
        ),
    }

    return {
        "output": output,
        "summary": summary,
    }


def choose_candidate(candidates):
    """
    Apply depth proximity, record length, and file-name tie breakers.
    """
    return min(
        candidates,
        key=lambda item: (
            item["summary"][
                "depth_deviation"
            ],
            -item["summary"]["length"],
            item["summary"][
                "candidate_name"
            ],
        ),
    )


# ======================================================
# STEP 3: WRITE FINAL TARGETS AND PUBLIC METADATA
# ======================================================
def build_targets(layer):
    """
    Build final target files and target metadata.

    All station adjustments and depth candidates remain in memory.
    """
    (
        adjusted_metadata,
        adjusted_by_fname,
    ) = load_adjusted_stations(layer)

    target_dir = (
        TARGET_ROOT / layer
    )

    target_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    META_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    if CLEAR_OLD_TARGETS:
        clear_target_files(
            target_dir
        )

    selected_rows = []

    grids = adjusted_metadata[
        "grid"
    ].unique()

    print(
        f"\nSTEP 2 | {layer} "
        f"| grid cells: {len(grids)}"
    )

    for grid_number, grid in enumerate(
        grids
    ):
        if grid_number % 25 == 0:
            print(
                f"  {grid_number}/{len(grids)}"
            )

        grid_metadata = (
            adjusted_metadata.loc[
                adjusted_metadata["grid"]
                == grid
            ]
        )

        candidates = []

        intervals = grid_metadata[
            "depth_interval"
        ].unique()

        for interval in intervals:
            metadata = (
                grid_metadata.loc[
                    grid_metadata[
                        "depth_interval"
                    ]
                    == interval
                ].copy()
            )

            candidate = build_candidate(
                metadata=metadata,
                adjusted_by_fname=(
                    adjusted_by_fname
                ),
                layer=layer,
                grid=grid,
                interval=interval,
            )

            if candidate is not None:
                candidates.append(
                    candidate
                )

        if not candidates:
            continue

        selected = choose_candidate(
            candidates
        )

        summary = selected[
            "summary"
        ]

        if (
            summary["length"]
            < MIN_VALID_DAYS
        ):
            continue

        output_name = target_name(
            grid,
            layer,
        )

        selected["output"].to_csv(
            target_dir / output_name,
            index=True,
            index_label="date",
            na_rep=NA_REP,
        )

        selected_rows.append(
            {
                "region": summary[
                    "region"
                ],
                "idx": summary["grid"],
                "lat_d25": summary[
                    "lat_d25"
                ],
                "lon_d25": summary[
                    "lon_d25"
                ],
                "layer": layer,
                "nominal_midpoint_cm": (
                    nominal_midpoint(layer)
                ),
                "depth0_cm": summary[
                    "depth0"
                ],
                "depth1_cm": summary[
                    "depth1"
                ],
                "representative_depth_cm": (
                    summary[
                        "representative_depth"
                    ]
                ),
                "depth_deviation_cm": (
                    summary[
                        "depth_deviation"
                    ]
                ),
                "length": summary[
                    "length"
                ],
                "start_date": summary[
                    "start_date"
                ],
                "end_date": summary[
                    "end_date"
                ],
                "n_primary_days": summary[
                    "n_primary_days"
                ],
                "n_gapfilled_days": (
                    summary[
                        "n_gapfilled_days"
                    ]
                ),
                "gapfilled_fraction": (
                    summary[
                        "gapfilled_fraction"
                    ]
                ),
                "primary_network": (
                    summary[
                        "primary_network"
                    ]
                ),
                "primary_station": (
                    summary[
                        "primary_station"
                    ]
                ),
                "primary_sensor": (
                    summary[
                        "primary_sensor"
                    ]
                ),
                "primary_fname": (
                    summary[
                        "primary_fname"
                    ]
                ),
                "n_station_candidates": (
                    summary[
                        "n_station_candidates"
                    ]
                ),
                "n_secondary_accepted": (
                    summary[
                        "n_secondary_accepted"
                    ]
                ),
                "n_secondary_used": (
                    summary[
                        "n_secondary_used"
                    ]
                ),
                "networks_used": (
                    summary[
                        "networks_used"
                    ]
                ),
                "stations_used": (
                    summary[
                        "stations_used"
                    ]
                ),
                "fpath": output_name,
            }
        )

    target_metadata = pd.DataFrame(
        selected_rows
    )

    metadata_file = (
        META_ROOT
        / f"target_{layer}.lst"
    )

    target_metadata.to_csv(
        metadata_file,
        index=False,
        na_rep=NA_REP,
    )

    print(f"\nSTEP 3 | {layer}")

    print(
        "  Final targets:",
        len(target_metadata),
    )

    print(
        "  Target directory:",
        target_dir,
    )

    print(
        "  Target metadata:",
        metadata_file,
    )


# ======================================================
# MAIN
# ======================================================
def main():
    print(
        "========================================"
    )

    print(
        " SoMoBench ISMN target construction"
    )

    print(
        "========================================"
    )

    print(
        "Daily station root:",
        DAILY_ROOT,
    )

    print(
        "ERA5 soil-moisture root:",
        ERA5_ROOT,
    )

    print(
        "Station metadata root:",
        META_ROOT,
    )

    for layer in LAYERS:
        print(
            "\n========================================"
        )

        print(
            "Layer:",
            layer,
        )

        print(
            "========================================"
        )

        build_targets(layer)

    print("\nDone.")


if __name__ == "__main__":
    main()
