# Downloading and pre-processing NOAA OISSSTv2 data

This folder contains the scripts to download and pre-process the NOAA OISSSTv2 data.

Please run 

    python download_oisstv2.py <DATA_DIR>

to download the OISSTv2 data. The data will be downloaded to the folder ``<DATA_DIR>/``.
You can select which years to download by changing the ``YEARS`` variable in the script (line 10).

To create the 60x60 degree boxes and pre-process them, run

    python oisstv2_boxed_data_preprocessing.py <DATA_DIR>

where ``<DATA_DIR>`` is the folder where the OISSTv2 data is stored.
This script will 
- divide the data into 60x60 degree boxes, 
- filter out all boxes with less than 95% of ocean pixels,
- standardize the raw SSTs using daily means and standard deviations (computed on the training set only, up to 2018), 
- replace continental NaNs with zeroes (after standardization), and
- save them in the folder ``<DATA_DIR>/subregion-60x60boxes-pixelwise_stats/``.







