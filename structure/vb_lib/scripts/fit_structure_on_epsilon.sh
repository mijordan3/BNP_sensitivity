#!/bin/bash

source activate bnp_sensitivity_jax

epsilon_indx=$SLURM_ARRAY_TASK_ID

data_file=../simulated_data/simulated_structure_data_nobs20_nloci50_npop4.npz
out_filename=simulated_fit

out_folder=../fits/tmp/

python fit_structure_perturbed.py \
  --epsilon_indx ${epsilon_indx} \
  --data_file ${data_file} \
  --out_folder ${out_folder} \
  --out_filename ${out_filename}_pertwc${epsilon_indx} \
  --init_fit ${out_folder}${out_filename}_alpha6.0.npz \
  --influence_file ${out_folder}${out_filename}_alpha6.0_lrderivatives.npz