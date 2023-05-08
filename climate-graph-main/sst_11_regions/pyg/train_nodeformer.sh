#!/bin/bash

program=train_nodeformer
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
device_idx=4

# Position encoding
pe=none
pos_dim=5

# Test mode
test_mode=1

# Longitude filter
lon_filter=2

# Latitude filter
lat_filter=2

# NodeFormer
rb_order=0

for history_window in 42
do
for predict_window in 7
do
name=${program}.rb_order.${rb_order}.num_epoch.${num_epoch}.batch_size.${batch_size}.learning_rate.${learning_rate}.seed.${seed}.n_layers.${n_layers}.hidden_dim.${hidden_dim}.pe.${pe}.pos_dim.${pos_dim}.lon_filter.${lon_filter}.lat_filter.${lat_filter}.history_window.${history_window}.predict_window.${predict_window}
CUDA_VISIBLE_DEVICES=$device_idx python3 $program.py --rb_order=$rb_order --dir=$dir --name=$name --num_epoch=$num_epoch --batch_size=$batch_size --learning_rate=$learning_rate --seed=$seed --n_layers=$n_layers --hidden_dim=$hidden_dim --pe=$pe --pos_dim=$pos_dim --test_mode=$test_mode --device=$device --data_dir=$data_dir --history_window=$history_window --predict_window=$predict_window --lon_filter=$lon_filter --lat_filter=$lat_filter
done
done

