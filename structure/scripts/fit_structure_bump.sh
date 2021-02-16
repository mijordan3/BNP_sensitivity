#!/bin/bash

source activate bnp_sensitivity_jax

epsilon_indx=$SLURM_ARRAY_TASK_ID

# data_file=../data/phased_HGDP+India+Africa_2810SNPs-regions1to36.npz 
# out_filename=huang2011_fit
# out_folder=../fits/hgdp_fits/

data_file=../data/huang2011_sub_nobs25_nloci75.npz
out_filename=huang2011_fit_sub
out_folder=../fits/tmp/

python fit_structure_gauss_bump.py \
  --epsilon_indx ${epsilon_indx} \
  --mu_indx ${mu_indx} \
  --data_file ${data_file} \
  --out_folder ${out_folder} \
  --out_filename ${out_filename}_gaussbump_mu${mu_indx}_eps${epsilon_indx} \
  --init_fit ${out_folder}${out_filename}_alpha6.0.npz 