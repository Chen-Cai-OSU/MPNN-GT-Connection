#!/bin/bash

program=train_transformer
dir=./$program/
mkdir $dir

data_fn=../data/oisstv2_standard/sst.day.mean.box85.nc

num_epoch=100
batch_size=20
learning_rate=0.001
seed=123456789
hidden_dim=256
heads=4
local_heads=1
depth=1
device=cuda

# Test mode
test_mode=0

# Position encoding dimension if used
pe_dim=5

for predict_window in 7 28
do
for history_window in 1 7 28
do
for pe_type in lap none
do
name=${program}.num_epoch.${num_epoch}.batch_size.${batch_size}.learning_rate.${learning_rate}.seed.${seed}.hidden_dim.${hidden_dim}.heads.${heads}.local_heads.${local_heads}.depth.${depth}.pe_type.${pe_type}.pe_dim.${pe_dim}.history_window.${history_window}.predict_window.${predict_window}
python3 $program.py --dir=$dir --data_fn=${data_fn} --name=$name --num_epoch=$num_epoch --batch_size=$batch_size --learning_rate=$learning_rate --seed=$seed --hidden_dim=$hidden_dim --heads=$heads --local_heads=$local_heads --depth=$depth --test_mode=$test_mode --pe_type=$pe_type --pe_dim=$pe_dim --history_window=$history_window --predict_window=$predict_window --device=$device
done
done
done
