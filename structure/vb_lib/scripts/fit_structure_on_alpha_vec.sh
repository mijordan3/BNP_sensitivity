#!/bin/bash

source activate bnp_sensitivity_jax

# alpha_vec=(1.0 1.5 2.0 2.5 3.0 3.5 4.0 4.5 5.0 5.5 \
#             6.5 7.0 7.5 8.0 8.5 9.0 9.5 10.0 10.5 11.0)

# alpha=${alpha_vec[$SLURM_ARRAY_TASK_ID]}
alpha = 2.0
echo $alpha 

seed=453

data_file=../simulated_data/simulated_structure_data_nobs20_nloci50_npop4.npz
out_filename=simulated_fit

# data_file=/accounts/grad/runjing_liu/BNP/fastStructure/hgdp_data/huang2011_plink_files/phased_HGDP+India+Africa_2810SNPs-regions1to36.npz
# out_filename=huang2011_fit


# data_file=../data/huang2011_subsampled.npz
# out_filename=huang2011_sub_fit

out_folder=../fits/fits_20201122/

python get_structure_fit.py \
  --seed ${seed} \
  --alpha ${alpha} \
  --data_file ${data_file} \
  --out_folder ${out_folder} \
  --out_filename ${out_filename}_alpha${alpha} \
  --warm_start True \
  --init_fit ${out_folder}${out_filename}_alpha6.0.npz \
  --k_approx 20
