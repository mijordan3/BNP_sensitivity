#!/bin/bash

source activate bnp_sensitivity_jax

seed=345345

nobs=1000
nloci=700000
npop=7

k_approx=20

alpha=3.5

scratch_folder='/scratch/users/genomic_times_series_bnp/structure'

/usr/bin/env python3 ./get_structure_fit.py \
  --seed ${seed} \
  --load_data False \
  --data_file ${scratch_folder}/data/simulated_structure_data_nobs${nobs}_nloci${nloci}_npop${npop}.npz \
  --alpha ${alpha} \
  --n_obs ${nobs} \
  --n_loci ${nloci} \
  --n_pop ${npop} \
  --k_approx ${k_approx} \
  --outfolder ${scratch_folder}/fits/fits_20201007/ \
  --out_filename structure_fit_nobs${nobs}_nloci${nloci}_npop${npop}_alpha${alpha}
