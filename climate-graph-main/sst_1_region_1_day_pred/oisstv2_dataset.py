import torch
import numpy as np
import json
from torch.utils.data import Dataset
import xarray

class oisstv2_dataset(Dataset):
    def __init__(self, dataset, slice_index, window = 7, virtual_node = 0):
        self.data = dataset.sel(time = slice_index)
        self.slice_index = slice_index
        self.window = window
        self.virtual_node = virtual_node

        # Convert to numpy array
        self.array = self.data.to_array().squeeze(axis = 0)
        
        # Statistics
        self.num_days = self.array.shape[0]
        self.num_longitudes = self.array.shape[1]
        self.num_latitudes = self.array.shape[2]
        self.num_samples = self.num_days - self.window

        print('Number of days:', self.num_days)
        print('Number of samples:', self.num_samples)
        print('Number of longitudes:', self.num_longitudes)
        print('Number of latitudes:', self.num_latitudes)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        idx = idx + self.window
        start = idx - self.window

        # Input
        inputs = torch.FloatTensor(self.array[start:idx, :, :].to_numpy())
        inputs = inputs.transpose(0, 1).transpose(1, 2)
        inputs = torch.reshape(inputs, (inputs.size(0) * inputs.size(1), inputs.size(2)))
        
        # targets
        targets = torch.FloatTensor(self.array[idx : idx + 1, :, :].to_numpy())
        targets = targets.transpose(0, 1).transpose(1, 2)
        targets = torch.reshape(targets, (targets.size(0) * targets.size(1), targets.size(2)))

        if self.virtual_node > 0:
            inputs = torch.cat([inputs, torch.zeros(1, inputs.size(1))], dim = 0)

        # Sample
        sample = {
            'inputs': inputs,
            'targets': targets
        }
        return sample

def create_adj(num_longitudes, num_latitudes, virtual_node = 0, num_shortcuts = 0, distance_threshold = 40):
    # Create the adjacency matrix
    num_nodes = num_longitudes * num_latitudes
    indices = []
    values = []
    for longitude in range(num_longitudes):
        for latitude in range(num_latitudes):
            idx = longitude * num_latitudes + latitude
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    longitude_ = longitude + dx
                    latitude_ = latitude + dy
                    if longitude_ >= 0 and longitude_ < num_longitudes and latitude_ >= 0 and latitude_ < num_latitudes:
                        idx_ = longitude_ * num_latitudes + latitude_
                        indices.append(torch.LongTensor(np.array([idx, idx_])).unsqueeze(dim = 1))
                        values.append(torch.FloatTensor(np.array([1])))

    # Virtual node
    if virtual_node > 0:
        for node in range(num_nodes):
            indices.append(torch.LongTensor(np.array([num_nodes, node])).unsqueeze(dim = 1))
            values.append(torch.FloatTensor(np.array([1])))

    # Shortcuts
    if num_shortcuts > 0:
        print('Adding random shortcuts')
        count = 0
        while count < num_shortcuts:
            x1 = np.random.randint(0, num_longitudes)
            y1 = np.random.randint(0, num_latitudes)
            x2 = np.random.randint(0, num_longitudes)
            y2 = np.random.randint(0, num_latitudes)
            dist = (x1 - x2)**2 + (y1 - y2)**2
            if dist > distance_threshold**2:
                prob = 1.0 / dist
                sample = np.sum(np.random.binomial(1, prob, 1))
                if sample > 0:
                    print('Added', count + 1, 'shortcuts')
                    count += 1
                    idx1 = x1 * num_latitudes + y1
                    idx2 = x2 * num_latitudes + y2
                    indices.append(torch.LongTensor(np.array([idx1, idx2])).unsqueeze(dim = 1))
                    values.append(torch.FloatTensor(np.array([1])))
                    indices.append(torch.LongTensor(np.array([idx2, idx1])).unsqueeze(dim = 1))
                    values.append(torch.FloatTensor(np.array([1])))

    # Output
    indices = torch.cat(indices, dim = 1)
    values = torch.cat(values, dim = 0)
    # adj = torch.sparse_coo_tensor(indices, values, [num_nodes, num_nodes])
    return indices, values

