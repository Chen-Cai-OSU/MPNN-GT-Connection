import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, Adagrad
import pickle
from torch import optim
import numpy as np
import os
import time
import argparse
import scipy

from torch_geometric.loader import DataLoader

# Dataset
import xarray
from oisstv2_dataset_pyg import read_all_raw_data, oisstv2_dataset_pyg, create_adj

# Parameters
history_window = 42
predict_window = 28

lon_filter = 2
lat_filter = 2

hetero = True

# Read all the raw datasets first
raw_datasets = read_all_raw_data('../../data/oisstv2_standard/')

num_regions = len(raw_datasets)

MIN_LON = 1e9
MAX_LON = -1e9
MIN_LAT = 1e9
MAX_LAT = -1e9

for idx in range(num_regions):
    print('Region', idx, '-----------------------------------')
    data = raw_datasets[idx]

    min_lon = min(data.lon.to_numpy())
    max_lon = max(data.lon.to_numpy())
    min_lat = min(data.lat.to_numpy())
    max_lat = max(data.lat.to_numpy())
    
    print('Min longitude:', min_lon)
    print('Max longitude:', max_lon)
    print('Min latitude:', min_lat)
    print('Max latitude:', max_lat)
    
    if min_lon < MIN_LON:
        MIN_LON = min_lon
    if max_lon > MAX_LON:
        MAX_LON = max_lon
    if min_lat < MIN_LAT:
        MIN_LAT = min_lat
    if max_lat > MAX_LAT:
        MAX_LAT = max_lat

print('Summary', '-----------------------------------')
print('Min longitude:', MIN_LON)
print('Max longitude:', MAX_LON)
print('Min latitude:', MIN_LAT)
print('Max latitude:', MAX_LAT)

print(data.lon)
print(data.lat)

print('Done')
