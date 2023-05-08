#!/bin/bash

program=train_clstm
dir=./$program/
mkdir $dir

data_fn=../data/oisstv2_standard/sst.day.mean.box85.nc

num_epoch=100
batch_size=20
learning_rate=0.001
seed=123456789
device=cuda

# Test mode
test_mode=0

for window in 1 7 28
do
name=${program}.num_epoch.${num_epoch}.batch_size.${batch_size}.learning_rate.${learning_rate}.seed.${seed}.window.${window}
python3 $program.py --dir=$dir --data_fn=${data_fn} --name=$name --num_epoch=$num_epoch --batch_size=$batch_size --learning_rate=$learning_rate --seed=$seed --test_mode=$test_mode --window=$window --device=$device
done
