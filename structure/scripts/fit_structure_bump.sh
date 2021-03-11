#!/bin/bash

source activate bnp_sensitivity_jax

epsilon_indx=$SLURM_ARRAY_TASK_ID

python fit_structure_bump.py \
  --epsilon_indx ${epsilon_indx} \
  --mu_indx ${mu_indx} \
  --data_file ${data_file} \
  --out_folder ${out_folder} \
  --out_filename ${out_filename}_stepbump_mu${mu_indx}_eps${epsilon_indx} \
  --init_fit ${out_folder}${out_filename}_alpha$alpha.npz 