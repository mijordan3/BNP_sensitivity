#!/bin/bash

source activate bnp_sensitivity_jax

alpha_vec=(1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0 10.0)

seed=453

nobs=200
nloci=500
npop=4

alpha=${alpha_vec[$SLURM_ARRAY_TASK_ID]}

echo $alpha

scratch_folder=/scratch/users/genomic_times_series_bnp/structure/
out_folder=${scratch_folder}fits/fits_20201008/

python get_structure_fit.py \
  --seed ${seed} \
  --alpha ${alpha} \
  --data_file ${scratch_folder}/simulated_data/simulated_structure_data_nobs${nobs}_nloci${nloci}_npop${npop}.npz \
  --out_folder ${out_folder} \
  --out_filename structure_fit_nobs${nobs}_nloci${nloci}_npop${npop}_alpha${alpha} \
  --warm_start True \
  --init_fit ${out_folder}structure_fit_nobs${nobs}_nloci${nloci}_npop${npop}_alpha3.5.npz