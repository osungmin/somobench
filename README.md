


# SoMoBench

SoMoBench is a benchmark dataset for data-driven soil moisture modelling with targets derived from in situ observations.

## What is SoMoBench?

SoMoBench provides daily soil moisture target time series and collocated meteorological forcing data for machine-learning applications. The dataset is designed to support reproducible benchmarking and model intercomparison for soil moisture modelling.

The target data are derived from in situ soil moisture observations from the International Soil Moisture Network (ISMN) and the CEMADEN network in Brazil. The observations are harmonized to a common 0.25° grid and provided for three nominal soil layers: 0–10, 10–30, and 30–50 cm. The dataset also includes daily ERA5 meteorological forcing time series and grid-cell metadata.


## About this repository

This repository provides code associated with the SoMoBench dataset and paper.

The current release includes:

- scripts to reproduce the figures presented in the SoMoBench paper
- example scripts demonstrating the construction of the soil moisture target dataset using sample data; reproducing the full workflow requires station data downloaded directly from ISMN (https://ismn.earth/)

## Repository structure

```text
somobench/
├── figures/
│   ├── figdata/
│   ├── fig02.py
│   ├── fig03.py
│   ├── fig04.py
│   ├── fig05.py
│   ├── fig06.py
│   └── fig07.py
├── src/
│   ├── demo_data/
│   │   ├── era5/
│   │   │   └── layer1_41.875_-111.625.dat
│   │   ├── meta/
│   │   │   └── ismn_point_metadata_demo.lst
│   │   └── raw_extracted/
│   │   │   ├── DEMONET_ST001_sD05-sD05_SENSOR1.dat
│   │   │   └── DEMONET_ST002_sD05-sD05_SENSOR1.dat
│   ├── 01_prepare_soilm_daily.py
│   └── 02_build_targets.py
├── README.md
└── environment.yml
```

## Installation

Create the Conda environment from environment.yml, and then activate the environment somobench:

```bash
conda env create -f environment.yml
source activate somobench
```

## Data and figure reproduction

The SoMoBench benchmark dataset is distributed separately from this repository.

The latest release (`SoMoBench_v1.0.zip`) can be downloaded from **Zenodo**:

> **Zenodo:** *(URL to be added)*

### Reproducing the figures

1. Download `SoMoBench_v1.0.zip` from Zenodo.
2. Place the ZIP file in the same directory as the figure script (e.g., `fig02.py`).
3. Extract the archive:

```bash
unzip SoMoBench_v1.0.zip
```

4. Run the desired figure script:

```bash
python fig02.py
```

Figures 2–4 and Figure 7 use only the released SoMoBench dataset contained in `SoMoBench_v1.0`.

Figures 5 and 6 additionally require the precomputed figure data included in this repository under:

```text
figures/
└── figdata/
```

## Citations
If you use SoMoBench in your research, we would appreciate a citation to the appropriate paper(s):
 SoMoBench: a quasi-global benchmark dataset for data-driven soil moisture modelling with in situ-derived targets

