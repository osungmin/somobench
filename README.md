


# SoMoBench
Observation-based benchmark dataset for soil moisture modeling with machine learning

# What is SoMoBench
Benchmark datasets in the area of machine learning (ML) aim to provide curated data that represent real-world problems, allowing the performance of different ML algorithms to be quantitatively evaluated and compared. Here we introduce a benchmark dataset ‘SoMoBench’ for soil moisture modeling using machine learning. In recent years, ML methods have been actively applied to simulate soil moisture dynamics, however, they are difficult to compare due to inconsistencies among the studies, including different data sources, preprocessing, and spatiotemporal scales. To overcome this issue, we aggregate and standardise existing global in-situ soil moisture data at the daily scale and 0.25° spatial resolution, and provide the data with meteorological forcing and statistic features at corresponding grid pixels. 


# About this repository

  - scripts/somobench.py: python script to directly use SomoBench data
    - one year has three .netcdf files; 1) meteorological forcing data, 2) static features, and 3) upscaled soil moisture
     
  - scripts/pre_soilm.py: python script to prepare SomoBench data at a grid pixel using ISMN raw data
    - download soil moisture data from [https://www.ismn.](https://ismn.earth) (log-in required) > DATA ACCESS
    - data can be extracted using ISMN github ()
    - see the example files in .
      
  - scripts
    - down_era5.py: to download daily ERA5 data using the CDS API service.
    - extract_era5.py: to extract meteorological forcing data from ERA5 at given grid pixels
    - extract_statics.py: to extract static features at given grid pixels
  
# Citations
If you use SoMoBench in your research, we would appreciate a citation to the appropriate paper(s):

For general use of SoMoBench you can read/cite our .. paper (bibtex).
For application of SoMoBench you can read/cite our Science Data papers (bibtex; free access).
