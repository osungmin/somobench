#!/usr/bin/env python3
"""
Create daily soil-moisture files and station metadata for the SoMoBench demo.

Expected directory structure
----------------------------
Place this script in ``src/`` and store the demo inputs below ``src/demo_data/``::

src/
├── 01_prepare_soilm_daily.py
└── demo_data/
    ├── raw_extracted/
    │   ├── DEMONET_ST001_sD05-sD05_SENSOR1.dat
    │   └── DEMONET_ST002_sD05-sD05_SENSOR1.dat
    ├── meta/
    │   └── ismn_point_metadata_demo.lst
    └── era5/
        └── layer1_41.875_-111.625.dat
        
Each raw sensor file must be semicolon-delimited and contain:
    - a datetime index
    - soilm: soil moisture
    - flag: ISMN quality flag
    - ori_flag: original provider flag

The station-metadata file must be comma-delimited and contain:
    - network
    - station
    - latitude
    - longitude

The script creates:
    - daily sensor files in ``demo_data/raw_daily/<layer>/``
    - daily station metadata in ``demo_data/meta/``
"""

import os

import numpy as np
import pandas as pd


############################################################
# Paths relative to this script
############################################################
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEMO_DATA_DIR = os.path.join(SCRIPT_DIR, "demo_data")

# Input: extracted sensor files
RAW_BASE = os.path.join(DEMO_DATA_DIR, "raw_extracted")
# Output: daily sensor files, grouped by layer
DAILY_BASE = os.path.join(DEMO_DATA_DIR, "raw_daily")
# Input/output: station metadata directory
META_DIR = os.path.join(DEMO_DATA_DIR, "meta")


def print_input_requirements():
    """Print a short description of the required demo input files."""
    print("\n*** Required demo input data ***")
    print("1. Raw sensor files:")
    print("   demo_data/raw_extracted/")
    print("   <network>_<station>_sDXX-sDYY_<sensor>.dat")
    print("   Semicolon-delimited with datetime, soilm, flag, and ori_flag.")
    print("2. Station metadata:")
    print("   demo_data/meta/ismn_point_metadata_demo.lst")
    print("   Required columns: network, station, latitude, longitude")
    print("********************************\n")


def gridding(latlon_dict, grid_resol=0.25):
    """Assign a station to the nearest cell center of the global regular grid."""
    lat = latlon_dict["lat"]
    lon = latlon_dict["lon"]

    # ERA5 0.25-degree grid-cell centers.
    lat0 = 89.875
    lon0 = -179.875
    lats = np.arange(lat0, -90, -grid_resol)
    lons = np.arange(lon0, 180, grid_resol)

    i_lat = int(np.argmin(np.abs(lats - lat)))
    i_lon = int(np.argmin(np.abs(lons - lon)))

    lat_d25 = lats[i_lat]
    lon_d25 = lons[i_lon]
    return lat_d25, lon_d25


def parse_sensor_filename(filename):
    """Extract network, station, sensor, and measurement interval from a raw filename."""
    stem, extension = os.path.splitext(filename)
    if extension.lower() != ".dat":
        raise ValueError(f"Expected a .dat file, received: {filename}")

    parts = stem.split("_")
    if len(parts) < 4:
        raise ValueError(f"Unexpected filename structure: {filename}")

    network = parts[0]
    station = parts[1]
    sensor = parts[-1]

    try:
        depth_info = stem.split("_sD", 1)[1].split("_", 1)[0]
        depth0 = int(depth_info.split("-sD")[0])
        depth1 = int(depth_info.split("-sD")[1])
    except (IndexError, ValueError) as error:
        raise ValueError(f"Could not parse depth information from: {filename}") from error

    if depth0 > depth1:
        raise ValueError(f"Invalid depth interval in {filename}: {depth0}-{depth1} cm")

    mid_depth = (depth0 + depth1) * 0.5
    return network, station, sensor, depth0, depth1, mid_depth


def sensor_belongs_to_layer(depth0, depth1, mid_depth, depth):
    """
    Return True when the full measurement interval is contained within
    one nominal layer.

    Point measurements at shared boundaries (10 and 30 cm) are assigned
    to the shallower layer.
    """
    if depth == "layer1":
        return depth0 >= 0 and depth1 <= 10

    if depth == "layer2":
        return depth0 >= 10 and depth1 <= 30 and mid_depth > 10

    if depth == "layer3":
        return depth0 >= 30 and depth1 <= 50 and mid_depth > 30

    raise ValueError("depth must be one of: layer1, layer2, layer3")


def load_station_metadata(region):
    """Load the station-level metadata for one region."""
    metadata_file = os.path.join(
        META_DIR,
        f"ismn_point_metadata_{region}.lst",
    )

    if not os.path.exists(metadata_file):
        raise FileNotFoundError(f"Metadata file not found: {metadata_file}")

    meta = pd.read_csv(
        metadata_file,
        header=0,
        sep=",",
        na_values=-9999.0,
    )

    required_columns = {"network", "station", "latitude", "longitude"}
    missing_columns = required_columns.difference(meta.columns)
    if missing_columns:
        raise ValueError(
            f"Missing metadata columns in {metadata_file}: {sorted(missing_columns)}"
        )

    return meta


def initialize_metadata_container():
    """Create the output metadata container for one region."""
    keys = [
        "region",
        "lat_d25",
        "lon_d25",
        "lat",
        "lon",
        "station",
        "network",
        "depth0",
        "depth1",
        "sensor",
        "fname",
    ]
    return {key: [] for key in keys}


def make_daily_series(raw_file, valid_hrs):
    """Read one sub-daily file, apply QC, and calculate daily mean soil moisture."""
    df = pd.read_csv(
        raw_file,
        names=["soilm", "flag", "ori_flag"],
        header=0,
        index_col=0,
        parse_dates=True,
        sep=";",
        na_values=-9999.0,
    )

    # Exclude physically invalid negative soil-moisture values.
    negative_values = int((df["soilm"] < 0).sum())
    if negative_values > 0:
        print(f"    Warning: excluded {negative_values} negative soil-moisture values")
        df.loc[df["soilm"] < 0, "flag"] = "ERR"

    # Retain only measurements marked as good by the ISMN quality-control flag.
    good = df[df["flag"] == "G"].copy()
    if good.empty:
        return pd.DataFrame(columns=["soilm"])

    # Retain a daily value only when enough valid observations are available.
    good["n_hour"] = good.groupby(good.index.normalize())["soilm"].transform("count")
    selected = good[good["n_hour"] >= valid_hrs].copy()
    if selected.empty:
        return pd.DataFrame(columns=["soilm"])

    return selected[["soilm"]].resample("D").mean()


def process_region(region, depth, grid_resol, valid_hrs, min_valid_days):
    """Create daily files and metadata for one region and one nominal layer."""
    datapath = RAW_BASE
    outpath = os.path.join(DAILY_BASE, depth)
    metadata_output = os.path.join(
        META_DIR,
        f"ismn_point_daily_{region}_{depth}.lst",
    )

    if not os.path.isdir(datapath):
        raise FileNotFoundError(f"Raw-data directory not found: {datapath}")

    os.makedirs(outpath, exist_ok=True)
    os.makedirs(META_DIR, exist_ok=True)

    raw_files = sorted(
        filename
        for filename in os.listdir(datapath)
        if filename.lower().endswith(".dat")
    )

    print(f"\n*** Processing region: {region}, layer: {depth} ***")
    print(f"Raw-data directory: {datapath}")
    print(f"Number of .dat files: {len(raw_files)}")
    print(f"Daily output directory: {outpath}")
    print(f"Metadata output file: {metadata_output}")

    meta = load_station_metadata(region)
    print(f"Station metadata rows: {len(meta)}")

    idxs_info = initialize_metadata_container()
    processed_count = 0
    too_small_data = 0

    for file_number, filename in enumerate(raw_files, start=1):
        print(f"[{file_number}/{len(raw_files)}] {filename}")

        network, station, sensor, depth0, depth1, mid_depth = parse_sensor_filename(
            filename
        )

        # Use the sensor only if its full measurement interval falls within the layer.
        if not sensor_belongs_to_layer(depth0, depth1, mid_depth, depth):
            continue

        idx_meta = meta[
            (meta["network"] == network) & (meta["station"] == station)
        ]
        if len(idx_meta) != 1:
            raise ValueError(
                "Metadata must contain exactly one matching row for "
                f"network={network}, station={station}; found {len(idx_meta)}"
            )

        raw_file = os.path.join(datapath, filename)
        daily = make_daily_series(raw_file, valid_hrs)
        valid_day_count = int(daily["soilm"].count())

        if valid_day_count < min_valid_days:
            too_small_data += 1
            continue

        stem = os.path.splitext(filename)[0]
        daily_filename = f"{stem}_{depth}.dat"
        daily.round(5).to_csv(
            os.path.join(outpath, daily_filename),
            na_rep=-9999,
        )
        print(f"    Saved: {daily_filename}")

        lat = idx_meta["latitude"].values[0]
        lon = idx_meta["longitude"].values[0]
        lat_d25, lon_d25 = gridding(
            {"lat": lat, "lon": lon},
            grid_resol,
        )

        idxs_info["region"].append(region)
        idxs_info["lat_d25"].append(round(lat_d25, 3))
        idxs_info["lon_d25"].append(round(lon_d25, 3))
        idxs_info["lat"].append(round(lat, 3))
        idxs_info["lon"].append(round(lon, 3))
        idxs_info["network"].append(network)
        idxs_info["station"].append(station)
        idxs_info["depth0"].append(depth0)
        idxs_info["depth1"].append(depth1)
        idxs_info["sensor"].append(sensor)
        idxs_info["fname"].append(stem)

        processed_count += 1

    df_info = pd.DataFrame(idxs_info)
    if not df_info.empty:
        df_info = df_info.sort_values(
            [
                "region",
                "lat_d25",
                "lon_d25",
                "network",
                "station",
                "depth0",
                "depth1",
            ]
        )

    df_info.to_csv(metadata_output, index=False, na_rep=-9999)

    print("\nRegion summary")
    print(f"Processed sensors: {processed_count}")
    print(f"Excluded with fewer than {min_valid_days} valid days: {too_small_data}")
    print(f"Saved metadata: {metadata_output}")


def main():
    """Run daily aggregation for the public layer-1 demo data."""
    ############################################################
    # Demo processing parameters
    ############################################################
    depth = "layer1"
    grid_resol = 0.25
    valid_hrs = 6
    min_valid_days = 90
    regions = ["demo"]

    print_input_requirements()

    if depth not in ["layer1", "layer2", "layer3"]:
        raise ValueError("depth must be one of: layer1, layer2, layer3")

    for region in regions:
        process_region(
            region=region,
            depth=depth,
            grid_resol=grid_resol,
            valid_hrs=valid_hrs,
            min_valid_days=min_valid_days,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
