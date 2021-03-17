#!/bin/bash

source activate bnp_sensitivity_jax

epsilon_indx=$SLURM_ARRAY_TASK_ID

python fit_structure_worst_case.py \
  --g_name ${g_name} \
  --data_file ${data_file} \
  --epsilon_indx ${epsilon_indx} \
  --out_folder ${out_folder} \
  --out_filename ${out_filename}_${g_name}_wc_eps${epsilon_indx} \
  --init_fit ${out_folder}${out_filename}_alpha${alpha}.npz \
  --influence_file ${out_folder}${out_filename}_alpha${alpha}_infl_funcs.npz 
