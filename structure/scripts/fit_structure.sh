#!/bin/bash

source activate bnp_sensitivity_jax

alpha=6.0 

seed=453

# data_file=/accounts/grad/runjing_liu/BNP/fastStructure/test/testdata.npz
# out_filename=testdata_fit

# data_dir=/accounts/grad/runjing_liu/BNP/fastStructure/hgdp_data/huang2011_plink_files
# data_file=${data_dir}/phased_HGDP+India+Africa_2810SNPs-regions1to36.npz 
# out_filename=huang2011_fit

# data_file=../data/huang2011_subsampled.npz
# out_filename=huang2011_sub_fit

data_file=../simulated_data/simulated_structure_data_nobs20_nloci50_npop4.npz
out_filename=simulated_fit

out_folder=../fits/tmp/

python fit_structure.py \
  --seed ${seed} \
  --alpha ${alpha} \
  --data_file ${data_file} \
  --out_folder ${out_folder} \
  --out_filename ${out_filename}_alpha${alpha} \
  --k_approx 20