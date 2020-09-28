#!/bin/bash

source activate genomic_time_series_py35

alpha_vec=(1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0 10.0)

seed=453

nobs=40
nloci=100
npop=4

alpha=${alpha_vec[$SLURM_ARRAY_TASK_ID]}

echo $alpha

scratch_folder='/scratch/users/genomic_times_series_bnp/structure'

/usr/bin/env python3 ./get_structure_fit.py \
  --seed ${seed} \
  --alpha ${alpha} \
  --load_data True \
  --data_file ${scratch_folder}/data/simulated_structure_data_nobs${nobs}_nloci${nloci}_npop${npop}.npz \
  --n_obs ${nobs} \
  --n_loci ${nloci} \
  --n_pop ${npop} \
  --outfolder ${scratch_folder}/fits/fits_20200928/ \
  --out_filename structure_fit_nobs${nobs}_nloci${nloci}_npop${npop}_alpha${alpha} \
  --warm_start True \
  --init_fit ${scratch_folder}/fits/structure_fit_nobs${nobs}_nloci${nloci}_npop${npop}_alpha3.5.npz
