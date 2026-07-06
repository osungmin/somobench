


# SoMoBench

SoMoBench is a benchmark dataset for data-driven soil moisture modelling with targets derived from in situ observations.

## What is SoMoBench?

SoMoBench provides daily soil moisture target time series and collocated meteorological forcing data for machine-learning applications. The dataset is designed to support reproducible benchmarking and model intercomparison for soil moisture modelling.

The target data are derived from in situ soil moisture observations from the International Soil Moisture Network (ISMN) and the CEMADEN network in Brazil. The observations are harmonized to a common 0.25° grid and provided for three nominal soil layers: 0–10, 10–30, and 30–50 cm. The dataset also includes daily ERA5 meteorological forcing time series and grid-cell metadata.


## About this repository

This repository provides code associated with the SoMoBench dataset and paper.

The current release includes:

- scripts to reproduce the figures presented in the SoMoBench paper
- scripts to run the illustrative baseline modelling experiments

Additional processing scripts for constructing the target dataset from raw input data may be added in future updates.

```text
somobench/
├── paper/
│   ├── reproduce_figures.py
│   └── run_baselines.py
├── README.md
└── requirements.txt

# Citations
If you use SoMoBench in your research, we would appreciate a citation to the appropriate paper(s):

