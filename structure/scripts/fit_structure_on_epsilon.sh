#!/bin/bash

source activate bnp_sensitivity_jax

epsilon_indx=$SLURM_ARRAY_TASK_ID

python fit_structure_perturbed.py \
  --perturbation ${perturbation} \
  --epsilon_indx ${epsilon_indx} \
  --data_file ${data_file} \
  --out_folder ${out_folder} \
  --out_filename ${out_filename}_${perturbation}_delta${delta}_eps${epsilon_indx} \
  --init_fit ${out_folder}${out_filename}_alpha${alpha}.npz \
  --delta ${delta}
