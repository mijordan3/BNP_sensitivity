#!/bin/bash

seed=8901
alpha=4.0

out_folder='../fits/'
out_filename='iris_fit'

# python fit_iris.py \
#   --seed ${seed} \
#   --alpha ${alpha} \
#   --out_folder ${out_folder} \
#   --out_filename ${out_filename}_alpha${alpha} \

python get_lr_derivative.py \
  --out_folder ${out_folder} \
  --fit_file ${out_filename}_alpha${alpha}.npz \
