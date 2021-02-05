#!/bin/bash

source activate bnp_sensitivity_jax

alpha_vec=(3.0 6.0 9.0)

alpha=${alpha_vec[$job_indx]}
echo $alpha

# data_file=../data/phased_HGDP+India+Africa_2810SNPs-regions1to36.npz 
# out_filename=huang2011_fit
# out_folder=../fits/hgdp_fits/

data_file=../data/huang2011_sub_nobs25_nloci75.npz
out_filename=huang2011_fit_sub
out_folder=../fits/tmp/

python get_lr_derivative.py \
  --data_file ${data_file} \
  --out_folder ${out_folder} \
  --fit_file ${out_filename}_alpha${alpha}.npz \
  --cg_tol 1e-2