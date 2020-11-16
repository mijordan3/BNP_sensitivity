#!/bin/bash

source activate bnp_sensitivity_jax

epsilon_indx=$SLURM_ARRAY_TASK_ID
# epsilon_indx=10

data_file=../data/huang2011_subsampled.npz
out_filename=huang2011_sub_fit

out_folder=../fits/f_sens_20201115/

python get_functional_perturbation_fit.py \
  --epsilon_indx ${epsilon_indx} \
  --data_file ${data_file} \
  --out_folder ${out_folder} \
  --out_filename ${out_filename}_logphi_indx${epsilon_indx} \
  --init_fit ${out_folder}${out_filename}_alpha6.0.npz \
  --use_worst_case False
