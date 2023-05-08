#!/bin/bash

program=train_mpnn_shortcuts
dir=./$program/
mkdir $dir

data_fn=../data/oisstv2_standard/sst.day.mean.box85.nc

num_epoch=10
batch_size=20
learning_rate=0.001
seed=123456789
n_layers=6
hidden_dim=256
z_dim=256
num_shortcuts=100
device=cuda

# Test mode
test_mode=0

for window in 1 7 28
do
name=${program}.num_epoch.${num_epoch}.batch_size.${batch_size}.learning_rate.${learning_rate}.seed.${seed}.n_layers.${n_layers}.hidden_dim.${hidden_dim}.z_dim.${z_dim}.window.${window}.num_shortcuts.${num_shortcuts}
python3 $program.py --dir=$dir --data_fn=${data_fn} --name=$name --num_epoch=$num_epoch --batch_size=$batch_size --learning_rate=$learning_rate --seed=$seed --n_layers=$n_layers --hidden_dim=$hidden_dim --z_dim=$z_dim --test_mode=$test_mode --window=$window --num_shortcuts=$num_shortcuts --device=$device
done
