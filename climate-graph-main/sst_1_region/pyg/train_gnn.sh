#!/bin/bash

program=train_gnn
dir=./$program/
mkdir $dir

data_fn=../../data/oisstv2_standard/sst.day.mean.box85.nc

num_epoch=100
batch_size=20
learning_rate=0.001
seed=123456789
n_layers=3
hidden_dim=256
device=cuda

# Position encoding
pe=none
pos_dim=5

# Test mode
test_mode=0

# Virtual node
virtual_node=0

for gnn_type in gcn gin pna
do
for predict_window in 7 28
do
for history_window in 1 7 28
do
name=${program}.num_epoch.${num_epoch}.batch_size.${batch_size}.learning_rate.${learning_rate}.seed.${seed}.n_layers.${n_layers}.hidden_dim.${hidden_dim}.pe.${pe}.pos_dim.${pos_dim}.virtual_node.${virtual_node}.gnn_type.${gnn_type}.history_window.${history_window}.predict_window.${predict_window}
python3 $program.py --dir=$dir --name=$name --num_epoch=$num_epoch --batch_size=$batch_size --learning_rate=$learning_rate --seed=$seed --n_layers=$n_layers --hidden_dim=$hidden_dim --pe=$pe --pos_dim=$pos_dim --virtual_node=$virtual_node --gnn_type=$gnn_type --test_mode=$test_mode --device=$device --data_fn=$data_fn --history_window=$history_window --predict_window=$predict_window
done
done
done
