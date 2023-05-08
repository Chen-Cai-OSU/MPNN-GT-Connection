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

# PyTorch geometric data loader
from torch_geometric.loader import DataLoader
from torch_scatter import scatter_mean

# Dataset
import xarray
from oisstv2_dataset_pyg import oisstv2_dataset_pyg, create_adj

# Load raw data
dataset = xarray.open_dataset('../data/oisstv2_standard/sst.day.mean.box85.nc')

history_window = 7
predict_window = 7

train_slice = slice(None, '2018-12-31')
valid_slice = slice('2019-01-01', '2019-12-31')
test_slice = slice('2020-01-01', '2021-12-31')

# PyG
# train_dataset = oisstv2_dataset_pyg(dataset = dataset, slice_index = train_slice, history_window = history_window, predict_window = predict_window)
valid_dataset = oisstv2_dataset_pyg(dataset = dataset, slice_index = valid_slice, history_window = history_window, predict_window = predict_window)
test_dataset = oisstv2_dataset_pyg(dataset = dataset, slice_index = test_slice, history_window = history_window, predict_window = predict_window)

batch_size = 20
# train_dataloader = DataLoader(train_dataset, batch_size, shuffle = False)
valid_dataloader = DataLoader(valid_dataset, batch_size, shuffle = False)
test_dataloader = DataLoader(test_dataset, batch_size, shuffle = False)

print(valid_dataset[0])

for batch in valid_dataloader:
    print(batch)
    print(batch.num_graphs)

    x = scatter_mean(batch.x, batch.batch, dim = 0)
    print(x.size())

print('Done')
