import torch
import torch.nn
from torch.utils.data import Dataset
import numpy as np
import json
import xarray

train_slice = slice(None, '2018-12-31')
valid_slice = slice('2019-01-01', '2019-12-31')
test_slice = slice('2020-01-01', '2021-12-31')

data_names = [
        'sst.day.mean.box84.nc',
        'sst.day.mean.box85.nc',
        'sst.day.mean.box86.nc',
        'sst.day.mean.box87.nc',
        'sst.day.mean.box88.nc',
        'sst.day.mean.box89.nc',
        'sst.day.mean.box108.nc',
        'sst.day.mean.box109.nc',
        'sst.day.mean.box110.nc',
        'sst.day.mean.box111.nc',
        'sst.day.mean.box112.nc'
]

def read_all_raw_data(data_dir):
    raw_datasets = []
    for name in data_names:
        data_fn = data_dir + '/' + name
        print('Reading data file', data_fn)
        dataset = xarray.open_dataset(data_fn)
        raw_datasets.append(dataset)
    return raw_datasets

class oisstv2_dataset(Dataset):
    def __init__(self, raw_datasets, split, history_window = 7, predict_window = 7, virtual_node = 0, lon_filter = 1, lat_filter = 1):
        # Get the right split
        if split == 'train':
            self.slice_index = train_slice
        elif split == 'valid':
            self.slice_index = valid_slice
        elif split == 'test':
            self.slice_index = test_slice
        else:
            print('Unsupported split!')
            self.slice_index = None
        assert self.slice_index is not None

        # Take the split out
        self.datasets = []
        for dataset in raw_datasets:
            self.datasets.append(dataset.sel(time = self.slice_index))
        print('Done taking the', split, 'slice out')

        self.history_window = history_window
        self.predict_window = predict_window
        self.virtual_node = virtual_node

        # Convert to numpy array
        self.arrays = []
        self.num_days = []
        self.num_samples = []

        for dataset in self.datasets:
            array = dataset.to_array().squeeze(axis = 0)
            self.arrays.append(array)
            self.num_days.append(array.shape[0])
            self.num_samples.append(array.shape[0] - self.history_window - self.predict_window)
        print('Done converting to numpy')
        
        # Statistics
        self.lon_filter = lon_filter
        self.lat_filter = lat_filter

        assert self.arrays[0].shape[1] % self.lon_filter == 0
        assert self.arrays[0].shape[2] % self.lat_filter == 0

        self.num_longitudes = self.arrays[0].shape[1] // self.lon_filter
        self.num_latitudes = self.arrays[0].shape[2] // self.lat_filter

        self.average_pooling = torch.nn.AvgPool2d((self.lon_filter, self.lat_filter), stride = (self.lon_filter, self.lat_filter)) # For changing the resolution

        self.total_num_days = np.sum(np.array(self.num_days))
        self.total_num_samples = np.sum(np.array(self.num_samples))
        self.num_boxes = len(self.arrays)

        print('Number of boxes:', self.num_boxes)
        print('Number of days:', self.num_days)
        print('Number of samples:', self.num_samples)
        print('Total number of days:', self.total_num_days)
        print('Total number of samples:', self.total_num_samples)
        print('Longitude filter size:', self.lon_filter)
        print('Latitude filter size:', self.lat_filter)
        print('Number of original longitudes:', self.arrays[0].shape[1])
        print('Number of original latitudes:', self.arrays[0].shape[2])
        print('Number of filtered longitudes:', self.num_longitudes)
        print('Number of filtered latitudes:', self.num_latitudes)

    def __len__(self):
        return self.total_num_samples

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Search for the right box
        count = 0
        box = -1
        for b in range(self.num_boxes):
            if idx <= count + self.num_samples[b]:
                box = b
                idx -= count
                break
            else:
                count += self.num_samples[b]
        assert box != -1
        array = self.arrays[box]

        # Create the sample
        finish = idx + self.history_window + self.predict_window

        # Change the resolution
        inputs = torch.FloatTensor(array[idx : idx + self.history_window, :, :].to_numpy())
        inputs = self.average_pooling(inputs).detach()

        targets = torch.FloatTensor(array[idx + self.history_window : finish, :, :].to_numpy())
        targets = self.average_pooling(targets).detach()

        # Input
        inputs = inputs.transpose(0, 1).transpose(1, 2)
        inputs = torch.reshape(inputs, (inputs.size(0) * inputs.size(1), inputs.size(2)))
        
        # Targets
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

