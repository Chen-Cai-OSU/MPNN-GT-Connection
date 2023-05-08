import logging
import os
import sys
from os.path import join
import xarray as xr

# Edit the following variables to change the download parameters
#  the data directory can be passed as an argument to the script:
#           python download_oisstv2.py /path/to/data/dir
if len(sys.argv) > 1:
    DATA_DIR = sys.argv[1]
else:
    DATA_DIR = "C:/Users/salvarc/data/oisstv2"

spacing = 60  # How many lat/lon points make a box
min_lat, max_lat = -60, 60  # The latitudes to consider (-60 to 60 means: don't use polar regions)
min_ocean_fraction = 0.95  # The minimum fraction of ocean points in a box to consider it valid
LAST_TRAIN_YEAR = 2018  # The last year to use for training (the rest will be used for testing/validation)

# We will save the data in a new sub directory
SUBREGION_DIR = join(DATA_DIR, f'subregion-{spacing}x{spacing}boxes-pixelwise_stats')
STATS_FILE = join(DATA_DIR, f"training_data_statistics_upto{LAST_TRAIN_YEAR}.nc")
os.makedirs(SUBREGION_DIR, exist_ok=True)

if __name__ == '__main__':
    # Open all years of data
    ds = xr.open_mfdataset(DATA_DIR + '/sst.day.mean.*.nc', combine='by_coords', decode_times=True, use_cftime=True)

    # First let's get the latitudes and longitudes of the grid
    lons = ds.lon.values
    lats = ds.lat.values
    # Now, let's get the limits of each box
    lon_limits = lons[::spacing]
    lat_limits = lats[::spacing]
    lat_limits = [l for l in lat_limits if min_lat <= l <= max_lat]
    ds = ds.sel(lat=slice(min_lat, max_lat))

    print(f"Saving training data to {SUBREGION_DIR}")
    if os.path.isfile(STATS_FILE):
        # Read the statistics from file
        stats = xr.open_dataset(STATS_FILE)
        sst_daily_mean = stats.sst_daily_mean
        sst_daily_std = stats.sst_daily_std
    else:
        # Compute the daily mean and std of the training data (i.e. each calendar day has its own mean and std):
        print(f'Computing daily mean and std of training data (up to year {LAST_TRAIN_YEAR})')
        daily_groupby = ds.sst.sel(time=slice(None, f'{LAST_TRAIN_YEAR}-12-31')).astype('float64').groupby(
            'time.dayofyear')
        # Also, map the dtype to float64 to avoid precision issues when computing the mean and std
        sst_daily_mean = daily_groupby.mean('time').astype('float32')
        sst_daily_std = daily_groupby.std('time').astype('float32')
        # Save the statistics to file
        xr.Dataset({'sst_daily_mean': sst_daily_mean, 'sst_daily_std': sst_daily_std}).to_netcdf(STATS_FILE)
        del daily_groupby

    # Iterate over the boxes
    for lat_index, lat_min in enumerate(lat_limits):
        for lon_index, lon_min in enumerate(lon_limits):
            grid_box_index = lat_index * len(lon_limits) + lon_index  # 0-based index of the grid box
            FILENAME = join(SUBREGION_DIR, f'sst.day.mean.box{grid_box_index}.nc')
            if os.path.exists(FILENAME):
                print(f'Grid box {grid_box_index} file already exists, skipping it.')
                continue
            print(f'Processing box {grid_box_index} (lat, lon) = ({lat_min}, {lon_min})')
            # Get the next lat and lon limits (exclusive, so subtract a small number, here 0.001)
            lat_max = lat_limits[lat_index + 1] - 0.001 if lat_index < len(lat_limits) - 1 else max_lat
            lon_max = lon_limits[lon_index + 1] - 0.001 if lon_index < len(lon_limits) - 1 else 360
            # select the grid box
            lat_lon_kwargs = dict(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))
            sst_box = ds.sst.sel(**lat_lon_kwargs)
            assert sst_box.lat.size == spacing and sst_box.lon.size == spacing
            # compute the fraction of ocean in the box
            ocean_fraction = sst_box.isel(time=0).notnull().sum(dim=('lat', 'lon')) / spacing ** 2
            if ocean_fraction < min_ocean_fraction:
                # print(f'Box {grid_box_index} has only {ocean_fraction * 100:.2f}% of ocean points')
                continue

            # Standardize the raw SST data:
            sst_box_daily_mean = sst_daily_mean.sel(**lat_lon_kwargs)
            sst_box_daily_std = sst_daily_std.sel(**lat_lon_kwargs)

            # Raise an error if any standard deviation is 0
            if (sst_box_daily_std == 0).any():
                std0_days = sst_box_daily_std[sst_box_daily_std == 0].time.values
                raise ValueError(f'Box {grid_box_index} has a std of 0 for the following days: {std0_days}')
            # Standardize the data
            sst_box_standardized = (sst_box.groupby('time.dayofyear') - sst_box_daily_mean) \
                                       .groupby('time.dayofyear') / sst_box_daily_std
            # The following DOES NOT use daily stats! sst_box - sst_box_daily_mean) / sst_box_daily_std
            # Set all Nan values to 0 (i.e. the continents)
            sst_box_standardized = sst_box_standardized.fillna(0)

            # Save the box to a new file
            sst_box_standardized.to_dataset(name='sst').to_netcdf(FILENAME)
            logging.info(f'Saved box {grid_box_index} to file in directory {SUBREGION_DIR}')



