#!/bin/bash

source activate genomic_time_series_py35

alpha_vec=(1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0 10.0)

seed=453

nobs=40
nloci=50
npop=4

alpha=${alpha_vec[$SLURM_ARRAY_TASK_ID]}

echo $alpha

/usr/bin/env python3 ./get_structure_fit.py \
  --seed ${seed} \
  --alpha ${alpha} \
  --load_data True \
  --data_file /scratch/users/genomic_times_series_bnp/structure/data/simulated_structure_data_nobs${nobs}_nloci${nloci}_npop${npop}.npz \
  --n_obs ${nobs} \
  --n_loci ${nloci} \
  --n_pop ${npop} \
  --outfolder /scratch/users/genomic_times_series_bnp/structure/fits \
  --out_filename structure_fit_nobs${nobs}_nloci${nloci}_npop${npop}_alpha${alpha} \
  --warm_start True \
  --init_fit /scratch/users/genomic_times_series_bnp/structure/fits/structure_fit_nobs${nobs}_nloci${nloci}_npop${npop}_alpha4.0.npz 
