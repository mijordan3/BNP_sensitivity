#!/bin/bash

source activate bnp_sensitivity_jax

seed=345345

nobs=200
nloci=500
npop=4

alpha=3.5

scratch_folder=/scratch/users/genomic_times_series_bnp/structure/
data_file=${scratch_folder}simulated_data/simulated_structure_data_nobs${nobs}_nloci${nloci}_npop${npop}.npz

out_folder=${scratch_folder}fits/fits_20201008/
out_filename=structure_fit_nobs${nobs}_nloci${nloci}_npop${npop}_alpha${alpha}

# get fit 
python get_structure_fit.py \
  --seed ${seed} \
  --data_file ${data_file} \
  --alpha ${alpha} \
  --out_folder ${out_folder} \
  --out_filename ${out_filename}

# compute alpha sensitivity derivatives
python get_alpha_derivative.py \
    --data_file ${data_file} \
    --fit_file ${out_folder}${out_filename}.npz \
    --out_folder ${out_folder} \
    --out_file alpha_sens_nobs${nobs}_nloci${nloci}_npop${npop}_alpha${alpha}