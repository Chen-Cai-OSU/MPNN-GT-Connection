import torch
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset

import numpy as np
import json
import xarray

class oisstv2_dataset_pyg(InMemoryDataset):
    def __init__(self, dataset, slice_index, history_window = 7, predict_window = 7):
        super().__init__()
        self.raw_data = dataset.sel(time = slice_index)
        self.slice_index = slice_index
        self.history_window = history_window
        self.predict_window = predict_window

        # Convert to numpy array
        self.array = self.raw_data.to_array().squeeze(axis = 0)
        
        # Statistics
        self.num_days = self.array.shape[0]
        self.num_longitudes = self.array.shape[1]
        self.num_latitudes = self.array.shape[2]
        self.num_samples = self.num_days - self.history_window - self.predict_window

        # Adjacency matrix
        self.edge_index, self.edge_attr = create_adj(self.num_longitudes, self.num_latitudes)

        print('Number of days:', self.num_days)
        print('Number of samples:', self.num_samples)
        print('Number of longitudes:', self.num_longitudes)
        print('Number of latitudes:', self.num_latitudes)

        # Create data list
        data_list = []
        for idx in range(self.num_samples):
            # Create the sample
            finish = idx + self.history_window + self.predict_window

            x = torch.FloatTensor(self.array[idx : idx + self.history_window, :, :].to_numpy())
            x = x.transpose(0, 1).transpose(1, 2)
            x = torch.reshape(x, (x.size(0) * x.size(1), x.size(2)))

            y = torch.FloatTensor(self.array[idx + self.history_window : finish, :, :].to_numpy())
            y = y.transpose(0, 1).transpose(1, 2)
            y = torch.reshape(y, (y.size(0) * y.size(1), y.size(2)))

            sample = Data()
            sample.__num_nodes__ = x.size(0)
            sample.x = x
            sample.y = y
            sample.edge_index = self.edge_index
            sample.edge_attr = self.edge_attr.unsqueeze(dim = 1)

            data_list.append(sample)

            if (idx + 1) % 1000 == 0:
                print('Done processing', idx + 1, 'samples')

        self.data, self.slices = self.collate(data_list)


def create_adj(num_longitudes, num_latitudes, virtual_node = 0, num_shortcuts = 0, distance_threshold = 40):
    # Create the adjacency matrix
    num_nodes = num_longitudes * num_latitudes
    indices = []
    values = []

    DX = [0, -1, -1, -1, 0, 1, 1, 1, 0] 
    DY = [0, -1, 0, 1, 1, 1, 0, -1, -1]
    num_directions = len(DX)

    for longitude in range(num_longitudes):
        for latitude in range(num_latitudes):
            idx = longitude * num_latitudes + latitude
            for direction in range(num_directions):
                dx = DX[direction]
                dy = DY[direction]
                longitude_ = longitude + dx
                latitude_ = latitude + dy
                if longitude_ >= 0 and longitude_ < num_longitudes and latitude_ >= 0 and latitude_ < num_latitudes:
                    idx_ = longitude_ * num_latitudes + latitude_
                    indices.append(torch.LongTensor(np.array([idx, idx_])).unsqueeze(dim = 1))
                    # values.append(torch.LongTensor(np.array([direction])))
                    values.append(torch.LongTensor(np.array([1])))

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

