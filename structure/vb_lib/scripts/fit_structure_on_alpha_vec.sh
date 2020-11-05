#!/bin/bash

source activate bnp_sensitivity_jax

alpha_vec=(1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0 10.0)
alpha=${alpha_vec[$SLURM_ARRAY_TASK_ID]}
echo $alpha 

seed=453

data_dir=/accounts/grad/runjing_liu/BNP/fastStructure/hgdp_data/huang2011_plink_files/
out_folder=../fits/fits_20201105/

python get_structure_fit.py \
  --seed ${seed} \
  --alpha ${alpha} \
  --data_file ${data_dir}phased_HGDP+India+Africa_2810SNPs-regions1to36.npz \
  --out_folder ${out_folder} \
  --out_filename huang2011_fit_alpha${alpha} \
  --k_approx 30
