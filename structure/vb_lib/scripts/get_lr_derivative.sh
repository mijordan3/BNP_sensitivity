#!/bin/bash

source activate bnp_sensitivity_jax

alpha_vec=(3.0 6.0 9.0)

alpha=${alpha_vec[$job_indx]}
echo $alpha

data_file=../simulated_data/simulated_structure_data_nobs20_nloci50_npop4.npz
out_filename=simulated_fit

# fs_dir=/accounts/grad/runjing_liu/BNP/fastStructure/hgdp_data/huang2011_plink_files/
# data_file=${fs_dir}phased_HGDP+India+Africa_2810SNPs-regions1to36.npz
# out_filename=huang2011_fit

out_folder=../fits/tmp/

python get_lr_derivative.py \
  --data_file ${data_file} \
  --out_folder ${out_folder} \
  --fit_file ${out_filename}_alpha${alpha}.npz
