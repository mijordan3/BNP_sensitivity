#!/bin/bash

source activate bnp_sensitivity_jax

alpha=6.0 

seed=453

data_dir=/accounts/grad/runjing_liu/BNP/fastStructure/test/
out_folder=../fits/fits_20201106/

python get_structure_fit.py \
  --seed ${seed} \
  --alpha ${alpha} \
  --data_file ${data_dir}testdata.npz \
  --out_folder ${out_folder} \
  --out_filename testdata_fit_alpha${alpha} \
  --k_approx 20
