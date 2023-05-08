import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, Adagrad
import pickle
from torch import optim
from torch.utils.data import DataLoader
import numpy as np
import os
import time
import argparse
import scipy

# Dataset
import xarray
from oisstv2_dataset import read_all_raw_data, oisstv2_dataset, create_adj

# Parameters
history_window = 42
predict_window = 7

lon_filter = 2
lat_filter = 2

# Read all the raw datasets first
raw_datasets = read_all_raw_data('../../data/oisstv2_standard/')

# Create PyTorch dataset
train_dataset = oisstv2_dataset(raw_datasets = raw_datasets, split = 'train', history_window = history_window, predict_window = predict_window, lon_filter = lon_filter, lat_filter = lat_filter)
valid_dataset = oisstv2_dataset(raw_datasets = raw_datasets, split = 'valid', history_window = history_window, predict_window = predict_window, lon_filter = lon_filter, lat_filter = lat_filter)
test_dataset = oisstv2_dataset(raw_datasets = raw_datasets, split = 'test', history_window = history_window, predict_window = predict_window, lon_filter = lon_filter, lat_filter = lat_filter)

# Data loaders
batch_size = 20
train_dataloader = DataLoader(train_dataset, batch_size, shuffle = True)
valid_dataloader = DataLoader(valid_dataset, batch_size, shuffle = False)
test_dataloader = DataLoader(test_dataset, batch_size, shuffle = False)

for batch_idx, data in enumerate(train_dataloader):
    print(data['inputs'].size())
    print(data['targets'].size())
    num_nodes = data['inputs'].size(1)
    node_dim = data['inputs'].size(2)
    num_outputs = data['targets'].size(2)
    break
print('Number of nodes:', num_nodes)
print('Number of input node features:', node_dim)
print('Number of outputs:', num_outputs)
assert num_outputs == predict_window

print('Done')
