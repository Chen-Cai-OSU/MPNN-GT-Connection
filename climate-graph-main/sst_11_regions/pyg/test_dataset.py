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

# Create PyTorch dataset
train_dataset = oisstv2_dataset_pyg(raw_datasets = raw_datasets, split = 'train', history_window = history_window, predict_window = predict_window, lon_filter = lon_filter, lat_filter = lat_filter, hetero = hetero)
valid_dataset = oisstv2_dataset_pyg(raw_datasets = raw_datasets, split = 'valid', history_window = history_window, predict_window = predict_window, lon_filter = lon_filter, lat_filter = lat_filter, hetero = hetero)
test_dataset = oisstv2_dataset_pyg(raw_datasets = raw_datasets, split = 'test', history_window = history_window, predict_window = predict_window, lon_filter = lon_filter, lat_filter = lat_filter, hetero = hetero)

# Data loaders
batch_size = 20
train_dataloader = DataLoader(train_dataset, batch_size, shuffle = True)
valid_dataloader = DataLoader(valid_dataset, batch_size, shuffle = False)
test_dataloader = DataLoader(test_dataset, batch_size, shuffle = False)

print('Number of training examples:', len(train_dataset))
print('Number of validation examples:', len(valid_dataset))
print('Number of testing examples:', len(test_dataset))

for batch_idx, data in enumerate(train_dataloader):
    print(batch_idx)
    print(data)
    node_dim = data.x.size(1)
    edge_dim = data.edge_attr.size(1)
    num_outputs = data.y.size(1)
    print('Minimum temperature:', torch.min(data.y))
    print('Maximum temperature:', torch.max(data.y))
    break

for batch_idx, data in enumerate(valid_dataloader):
    print(batch_idx)
    print(data)
    assert node_dim == data.x.size(1)
    assert edge_dim == data.edge_attr.size(1)
    assert num_outputs == data.y.size(1)
    break

for batch_idx, data in enumerate(test_dataloader):
    print(batch_idx)
    print(data)
    assert node_dim == data.x.size(1)
    assert edge_dim == data.edge_attr.size(1)
    assert num_outputs == data.y.size(1)
    break

print('Number of node features:', node_dim)
print('Number of edge features:', edge_dim)
print('Number of outputs:', num_outputs)

print('Done')
