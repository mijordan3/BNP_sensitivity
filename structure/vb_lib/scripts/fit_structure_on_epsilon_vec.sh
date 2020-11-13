#!/bin/bash

source activate bnp_sensitivity_jax

epsilon_indx=$SLURM_ARRAY_TASK_ID

# data_file=/accounts/grad/runjing_liu/BNP/fastStructure/hgdp_data/huang2011_plink_files/phased_HGDP+India+Africa_2810SNPs-regions1to36.npz
# out_filename=huang2011_fit


data_file=../data/huang2011_subsampled.npz
out_filename=huang2011_sub_fit

out_folder=../fits/fits_20201112/

python get_functional_perturbation_fit.py \
  --epsilon_indx ${epsilon_indx} \
  --data_file ${data_file} \
  --out_folder ${out_folder} \
  --out_filename ${out_filename}_ws_indx${epsilon_indx} \
  --init_fit ${out_folder}${out_filename}_alpha3.5.npz 
