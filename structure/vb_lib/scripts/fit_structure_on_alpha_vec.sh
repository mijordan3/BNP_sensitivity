#!/bin/bash

source activate bnp_sensitivity_jax

alpha_vec=(1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0 10.0)
alpha=${alpha_vec[$SLURM_ARRAY_TASK_ID]}
echo $alpha 

seed=453

nobs=20
nloci=50
npop=4

out_folder=../fits/fits_20201105/

python get_structure_fit.py \
  --seed ${seed} \
  --alpha ${alpha} \
  --data_file ../simulated_data/simulated_structure_data_nobs${nobs}_nloci${nloci}_npop${npop}.npz \
  --out_folder ${out_folder} \
  --out_filename structure_fit_nobs${nobs}_nloci${nloci}_npop${npop}_alpha${alpha}
