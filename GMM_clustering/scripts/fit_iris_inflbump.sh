#!/bin/bash

source activate bnp_sensitivity_jax

epsilon_indx=$SLURM_ARRAY_TASK_ID

out_folder='../fits/'
out_filename='iris_fit'

python fit_iris_inflbump.py \
  --epsilon_indx ${epsilon_indx} \
  --mu_indx ${mu_indx} \
  --out_folder ${out_folder} \
  --out_filename ${out_filename}_inflbump_mu${mu_indx}_eps${epsilon_indx} \
  --init_fit ${out_folder}${out_filename}_alpha6.0.npz 