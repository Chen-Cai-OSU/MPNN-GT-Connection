#!/bin/bash

program=train_cnn
dir=./$program/
mkdir $dir

data_dir=../../data/oisstv2_standard/

num_epoch=100
batch_size=20
learning_rate=0.001
seed=123456789
hidden_dim=256
device=cuda

# Change resolution
lon_filter=2
lat_filter=2

# Test mode
test_mode=0

for history_window in 42
do
for predict_window in 42 28 7 1
do
name=${program}.num_epoch.${num_epoch}.batch_size.${batch_size}.learning_rate.${learning_rate}.seed.${seed}.hidden_dim.${hidden_dim}.lon_filter.${lon_filter}.lat_filter.${lat_filter}.history_window.${history_window}.predict_window.${predict_window}
python3 $program.py --dir=$dir --data_dir=${data_dir} --name=$name --num_epoch=$num_epoch --batch_size=$batch_size --learning_rate=$learning_rate --seed=$seed --hidden_dim=$hidden_dim --test_mode=$test_mode --lon_filter=$lon_filter --lat_filter=$lat_filter --history_window=$history_window --predict_window=$predict_window --device=$device
done
done
