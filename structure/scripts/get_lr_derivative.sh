#!/bin/bash

source activate bnp_sensitivity_jax

data_file=$1
alpha=$2
out_folder=$3
out_filename=$4

python get_lr_derivative.py \
  --data_file ${data_file} \
  --out_folder ${out_folder} \
  --fit_file ${out_filename}_alpha${alpha}.npz \
  --cg_tol 1e-8
