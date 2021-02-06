#!/bin/bash

source activate bnp_sensitivity_jax

# data_file=../data/phased_HGDP+India+Africa_2810SNPs-regions1to36.npz 
# out_filename=huang2011_fit
# out_folder=../fits/hgdp_fits/

data_file=../data/huang2011_sub_nobs25_nloci75.npz
out_filename=huang2011_fit_sub
out_folder=../fits/tmp/

python get_posterior_quantities.py \
  --data_file ${data_file} \
  --fit_file ${out_folder}${out_filename}_${perturbation}_delta${delta}_eps${SLURM_ARRAY_TASK_ID}.npz \
  --lr_file ${out_folder}${out_filename}_alpha6.0_lrderivatives.npz 