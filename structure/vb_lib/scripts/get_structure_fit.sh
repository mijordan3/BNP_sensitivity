#!/bin/bash

source activate bnp_sensitivity_jax

alpha=6.0 

seed=453

# data_file=/accounts/grad/runjing_liu/BNP/fastStructure/test/testdata.npz
# out_filename=testdata_fits_alpha${alpha}

data_file=/accounts/grad/runjing_liu/BNP/fastStructure/hgdp_data/huang2011_plink_files/phased_HGDP+India+Africa_2810SNPs-regions1to36.npz 
out_filename=huang2011_fits_alpha

out_folder=../fits/fits_20201108/

python get_structure_fit.py \
  --seed ${seed} \
  --alpha ${alpha} \
  --data_file ${data_file} \
  --out_folder ${out_folder} \
  --out_filename ${out_filename}${alpha} \
  --k_approx 30
