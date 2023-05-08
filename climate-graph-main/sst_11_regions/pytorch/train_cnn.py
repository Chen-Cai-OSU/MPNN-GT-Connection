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

# Model
import sys
sys.path.insert(1, '../../models/pytorch/')
from cnn_model import CNN

# Fix number of threads
torch.set_num_threads(4)

def _parse_args():
    parser = argparse.ArgumentParser(description = 'Supervised learning')
    parser.add_argument('--dir', '-dir', type = str, default = '.', help = 'Directory to save results')
    parser.add_argument('--data_dir', '-data_dir', type = str, default = '.', help = 'Directory that contains the raw datasets')
    parser.add_argument('--name', '-name', type = str, default = 'NAME', help = 'Name')
    parser.add_argument('--num_epoch', '-num_epoch', type = int, default = 2048, help = 'Number of epochs')
    parser.add_argument('--batch_size', '-batch_size', type = int, default = 20, help = 'Batch size')
    parser.add_argument('--learning_rate', '-learning_rate', type = float, default = 0.001, help = 'Initial learning rate')
    parser.add_argument('--seed', '-s', type = int, default = 123456789, help = 'Random seed')
    parser.add_argument('--hidden_dim', '-hidden_dim', type = int, default = 32, help = 'Hidden dimension')
    parser.add_argument('--history_window', '-history_window', type = int, default = 7, help = 'History window')
    parser.add_argument('--predict_window', '-predict_window', type = int, default = 7, help = 'Predict window')
    parser.add_argument('--test_mode', '-test_mode', type = int, default = 0, help = 'Test mode')
    parser.add_argument('--lon_filter', '-lon_filter', type = int, default = 1, help = 'Longitude filter to reduce size')
    parser.add_argument('--lat_filter', '-lat_filter', type = int, default = 1, help = 'Latitude filter to reduce size')
    parser.add_argument('--device', '-device', type = str, default = 'cpu', help = 'cuda/cpu')
    args = parser.parse_args()
    return args

args = _parse_args()
if args.test_mode == 0:
    log_name = args.dir + "/" + args.name + ".log"
else:
    print("Test mode")
    log_name = args.dir + "/" + args.name + ".test_mode.log"
model_name = args.dir + "/" + args.name + ".model"
LOG = open(log_name, "w")

# Fix CPU torch random seed
torch.manual_seed(args.seed)

# Fix GPU torch random seed
torch.cuda.manual_seed(args.seed)

# Fix the Numpy random seed
np.random.seed(args.seed)

# Train on CPU (hide GPU) due to memory constraints
# os.environ['CUDA_VISIBLE_DEVICES'] = ""
device = args.device
print(device)

# Dataset
print(args.data_dir)
raw_datasets = read_all_raw_data(args.data_dir)

history_window = args.history_window
predict_window = args.predict_window
print('History window:', history_window)
print('Predict window:', predict_window)

# Create PyTorch dataset
lon_filter = args.lon_filter
lat_filter = args.lat_filter

train_dataset = oisstv2_dataset(raw_datasets = raw_datasets, split = 'train', history_window = history_window, predict_window = predict_window, lon_filter = lon_filter, lat_filter = lat_filter)
valid_dataset = oisstv2_dataset(raw_datasets = raw_datasets, split = 'valid', history_window = history_window, predict_window = predict_window, lon_filter = lon_filter, lat_filter = lat_filter)
test_dataset = oisstv2_dataset(raw_datasets = raw_datasets, split = 'test', history_window = history_window, predict_window = predict_window, lon_filter = lon_filter, lat_filter = lat_filter)

# Data loaders
batch_size = args.batch_size
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

# Create adjacency matrix
num_longitudes = train_dataset.num_longitudes
num_latitudes = train_dataset.num_latitudes
assert num_nodes == num_longitudes * num_latitudes

# Init model and optimizer
model = CNN(channels_in = history_window, channels_out = predict_window, channels_hidden = args.hidden_dim).to(device = device)
optimizer = Adagrad(model.parameters(), lr = args.learning_rate)

num_parameters = sum(param.numel() for param in model.parameters() if param.requires_grad)
print('Number of learnable parameters:', num_parameters)
LOG.write('Number of learnable parameters: ' + str(num_parameters) + '\n')

# Test mode
if args.test_mode == 1:
    print("Skip the training")
    num_epoch = 0
else:
    num_epoch = args.num_epoch

# Train model
best_mae = 1e9
for epoch in range(num_epoch):
    print('--------------------------------------')
    print('Epoch', epoch)
    LOG.write('--------------------------------------\n')
    LOG.write('Epoch ' + str(epoch) + '\n')

    # Training
    t = time.time()
    total_loss = 0.0
    nBatch = 0
    sum_error = 0.0
    num_samples = 0
    
    for batch_idx, data in enumerate(train_dataloader):
        node_feat = data['inputs'].float().to(device = device)
        targets = data['targets'].float().to(device = device)

        # Reshape to temporal images
        batch_size = node_feat.size(0)
        images = torch.reshape(node_feat, (batch_size, num_longitudes, num_latitudes, history_window))
        images = images.transpose(2, 3).transpose(1, 2)

        targets = torch.reshape(targets, (batch_size, num_longitudes, num_latitudes, predict_window))
        targets = targets.transpose(2, 3).transpose(1, 2)

        # Model
        predict = model(images)
        optimizer.zero_grad()
        
        # Mean squared error loss
        targets = targets.contiguous()
        predict = predict.contiguous()
        loss = F.mse_loss(predict.view(-1), targets.view(-1), reduction = 'mean')

        sum_error += torch.sum(torch.abs(predict.view(-1) - targets.view(-1))).detach().cpu().numpy()
        num_samples += node_feat.size(0)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        nBatch += 1
        if batch_idx % 100 == 0:
            print('Batch', batch_idx, '/', len(train_dataloader),': Loss =', loss.item())
            LOG.write('Batch ' + str(batch_idx) + '/' + str(len(train_dataloader)) + ': Loss = ' + str(loss.item()) + '\n')

    train_mae = sum_error / (num_samples * num_outputs * num_nodes)
    avg_loss = total_loss / nBatch
    
    print('Train average loss:', avg_loss)
    LOG.write('Train average loss: ' + str(avg_loss) + '\n')
    print('Train MAE:', train_mae)
    LOG.write('Train MAE: ' + str(train_mae) + '\n')
    print("Train time =", "{:.5f}".format(time.time() - t))
    LOG.write("Train time = " + "{:.5f}".format(time.time() - t) + "\n")

    # Validation
    t = time.time()
    model.eval()
    total_loss = 0.0
    nBatch = 0

    with torch.no_grad():
        sum_error = 0.0
        num_samples = 0
        for batch_idx, data in enumerate(valid_dataloader):
            node_feat = data['inputs'].float().to(device = device)
            targets = data['targets'].float().to(device = device)
            
            # Reshape to temporal images
            batch_size = node_feat.size(0)
            images = torch.reshape(node_feat, (batch_size, num_longitudes, num_latitudes, history_window))
            images = images.transpose(2, 3).transpose(1, 2)

            targets = torch.reshape(targets, (batch_size, num_longitudes, num_latitudes, predict_window))
            targets = targets.transpose(2, 3).transpose(1, 2)

            # Model
            predict = model(images)

            # Mean squared error loss
            targets = targets.contiguous()
            predict = predict.contiguous()
            loss = F.mse_loss(predict.view(-1), targets.view(-1), reduction = 'mean')

            total_loss += loss.item()
            nBatch += 1

            # Mean average error
            sum_error += torch.sum(torch.abs(predict.view(-1) - targets.view(-1))).detach().cpu().numpy()
            num_samples += node_feat.size(0)
             
            if batch_idx % 100 == 0:
                print('Valid Batch', batch_idx, '/', len(valid_dataloader),': Loss =', loss.item())
                LOG.write('Valid Batch ' + str(batch_idx) + '/' + str(len(valid_dataloader)) + ': Loss = ' + str(loss.item()) + '\n')

    valid_mae = sum_error / (num_samples * num_outputs * num_nodes)
    avg_loss = total_loss / nBatch

    print('Valid average loss:', avg_loss)
    LOG.write('Valid average loss: ' + str(avg_loss) + '\n')
    print('Valid MAE:', valid_mae)
    LOG.write('Valid MAE: ' + str(valid_mae) + '\n')
    print("Valid time =", "{:.5f}".format(time.time() - t))
    LOG.write("Valid time = " + "{:.5f}".format(time.time() - t) + "\n")
    
    if valid_mae < best_mae:
        best_mae = valid_mae
        print('Current best MAE updated:', best_mae)
        LOG.write('Current best MAE updated: ' + str(best_mae) + '\n')
        
        torch.save(model.state_dict(), model_name)
        print("Save the best model to " + model_name)
        LOG.write("Save the best model to " + model_name + "\n")
    else:
        # Early stopping
        # break
        pass

if args.test_mode == 0:
    print('--------------------------------------')
    LOG.write('--------------------------------------\n')
    print('Best valid MAE:', best_mae)
    LOG.write('Best valid MAE: ' + str(best_mae) + '\n')

# Load the model with the best validation
print("Load the trained model at", model_name)
model.load_state_dict(torch.load(model_name))

# Testing
t = time.time()
model.eval()
total_loss = 0.0
nBatch = 0

# For visualization
'''
all_predicts = []
all_targets = []
'''

with torch.no_grad():
    sum_error = 0.0
    num_samples = 0
    for batch_idx, data in enumerate(test_dataloader):
        node_feat = data['inputs'].float().to(device = device)
        targets = data['targets'].float().to(device = device)

        # Reshape to temporal images
        batch_size = node_feat.size(0)
        images = torch.reshape(node_feat, (batch_size, num_longitudes, num_latitudes, history_window))
        images = images.transpose(2, 3).transpose(1, 2)

        targets = torch.reshape(targets, (batch_size, num_longitudes, num_latitudes, predict_window))
        targets = targets.transpose(2, 3).transpose(1, 2)

        # Model
        predict = model(images)
        
        # Keep track for visualization
        '''
        all_predicts.append(predict.detach().cpu())
        all_targets.append(targets.detach().cpu())
        '''

        # Mean squared error loss
        targets = targets.contiguous()
        predict = predict.contiguous()
        loss = F.mse_loss(predict.view(-1), targets.view(-1), reduction = 'mean')

        total_loss += loss.item()
        nBatch += 1

        sum_error += torch.sum(torch.abs(predict.view(-1) - targets.view(-1))).detach().cpu().numpy()
        num_samples += node_feat.size(0)

        if batch_idx % 100 == 0:
            print('Test Batch', batch_idx, '/', len(test_dataloader),': Loss =', loss.item())
            LOG.write('Test Batch ' + str(batch_idx) + '/' + str(len(test_dataloader)) + ': Loss = ' + str(loss.item()) + '\n')

test_mae = sum_error / (num_samples * num_outputs * num_nodes)
avg_loss = total_loss / nBatch

# For visualization
'''
all_predicts = torch.cat(all_predicts, dim = 0).squeeze(dim = 2)
all_targets = torch.cat(all_targets, dim = 0).squeeze(dim = 2)

all_predicts = torch.reshape(all_predicts, (all_predicts.size(0), num_longitudes, num_latitudes))
all_targets = torch.reshape(all_targets, (all_targets.size(0), num_longitudes, num_latitudes))

all_predicts = all_predicts.detach().numpy()
all_targets = all_targets.detach().numpy()

np.save(args.dir + "/" + args.name + '.test_predicts', all_predicts)
np.save(args.dir + "/" + args.name + '.test_targets', all_targets)
'''

# Visualization
'''
import matplotlib.pyplot as plt
n_plots = 10
fig, axs = plt.subplots(2, n_plots, figsize = (19, 6), sharey = True, sharex = True)
for i in range(n_plots):
    predict = all_predicts[i, :, :]
    target = all_targets[i, :, :]
    axs[0, i].imshow(predict)
    axs[1, i].imshow(target)
fig.savefig(args.dir + "/" + args.name + '.png')
'''

print('--------------------------------------')
LOG.write('--------------------------------------\n')
print('Test average loss:', avg_loss)
LOG.write('Test average loss: ' + str(avg_loss) + '\n')
print('Test MAE:', test_mae)
LOG.write('Test MAE: ' + str(test_mae) + '\n')
print("Test time =", "{:.5f}".format(time.time() - t))
LOG.write("Test time = " + "{:.5f}".format(time.time() - t) + "\n")

LOG.close()
