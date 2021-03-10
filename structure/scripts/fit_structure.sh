#!/bin/bash

source activate bnp_sensitivity_jax

seed=45319801

data_file=$1
alpha=$2
out_folder=$3
out_filename=$4

python fit_structure.py \
  --seed ${seed} \
  --alpha ${alpha} \
  --data_file ${data_file} \
  --out_folder ${out_folder} \
  --out_filename ${out_filename}_alpha${alpha} \
  --k_approx 20 \
  