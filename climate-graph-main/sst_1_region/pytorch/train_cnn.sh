#!/bin/bash

program=train_cnn
dir=./$program/
mkdir $dir

data_fn=../data/oisstv2_standard/sst.day.mean.box85.nc

num_epoch=100
batch_size=20
learning_rate=0.001
seed=123456789
hidden_dim=64
device=cuda

# Test mode
test_mode=0

for history_window in 28
do
for predict_window in 1 7 28
do
name=${program}.num_epoch.${num_epoch}.batch_size.${batch_size}.learning_rate.${learning_rate}.seed.${seed}.hidden_dim.${hidden_dim}.history_window.${history_window}.predict_window.${predict_window}
python3 $program.py --dir=$dir --data_fn=${data_fn} --name=$name --num_epoch=$num_epoch --batch_size=$batch_size --learning_rate=$learning_rate --seed=$seed --hidden_dim=$hidden_dim --test_mode=$test_mode --history_window=$history_window --predict_window=$predict_window --device=$device
done
done
