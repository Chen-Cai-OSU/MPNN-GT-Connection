#!/bin/bash

program=train_gnn
dir=./$program/
mkdir $dir

data_dir=../../data/oisstv2_standard/

num_epoch=100
batch_size=20
learning_rate=0.001
seed=123456789
n_layers=3
hidden_dim=256

# Device
device=cuda
device_idx=2

# Position encoding
pe=none
pos_dim=5

# Test mode
test_mode=1

# Longitude filter
lon_filter=2

# Latitude filter
lat_filter=2

for gnn_type in gcn
do
for history_window in 42
do
for predict_window in 14
do
for virtual_node in 0
do
name=${program}.num_epoch.${num_epoch}.batch_size.${batch_size}.learning_rate.${learning_rate}.seed.${seed}.n_layers.${n_layers}.hidden_dim.${hidden_dim}.pe.${pe}.pos_dim.${pos_dim}.virtual_node.${virtual_node}.gnn_type.${gnn_type}.lon_filter.${lon_filter}.lat_filter.${lat_filter}.history_window.${history_window}.predict_window.${predict_window}
CUDA_VISIBLE_DEVICES=$device_idx python3 $program.py --dir=$dir --name=$name --num_epoch=$num_epoch --batch_size=$batch_size --learning_rate=$learning_rate --seed=$seed --n_layers=$n_layers --hidden_dim=$hidden_dim --pe=$pe --pos_dim=$pos_dim --virtual_node=$virtual_node --gnn_type=$gnn_type --test_mode=$test_mode --device=$device --data_dir=$data_dir --history_window=$history_window --predict_window=$predict_window --lon_filter=$lon_filter --lat_filter=$lat_filter
done
done
done
done

