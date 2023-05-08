#!/bin/bash

program=train_tfnet
dir=./$program/
mkdir $dir

data_dir=../../data/oisstv2_standard/

num_epoch=100
batch_size=20
learning_rate=0.001
seed=123456789

# Device
device=cuda
device_idx=5

# Change resolution
lon_filter=2
lat_filter=2

# Test mode
test_mode=1

# History window
history_window=42

for predict_window in 7
do
name=${program}.num_epoch.${num_epoch}.batch_size.${batch_size}.learning_rate.${learning_rate}.seed.${seed}.lon_filter.${lon_filter}.lat_filter.${lat_filter}.history_window.${history_window}.predict_window.${predict_window}
CUDA_VISIBLE_DEVICES=$device_idx python3 $program.py --dir=$dir --data_dir=${data_dir} --name=$name --num_epoch=$num_epoch --batch_size=$batch_size --learning_rate=$learning_rate --seed=$seed --test_mode=$test_mode --lon_filter=$lon_filter --lat_filter=$lat_filter --history_window=$history_window --predict_window=$predict_window --device=$device
done
