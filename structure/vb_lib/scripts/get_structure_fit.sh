#!/bin/bash

source activate genomic_time_series_py35

seed=345345

nobs=100
nloci=2000
npop=4

alpha=4.0

/usr/bin/env python3 ./get_structure_fit.py \
  --seed ${seed} \
  --load_data True \
  --data_file /scratch/users/genomic_times_series_bnp/structure/data/simulated_structure_data_nobs${nobs}_nloci${nloci}_npop${npop}.npz \
  --alpha ${alpha} \
  --n_obs ${nobs} \
  --n_loci ${nloci} \
  --n_pop ${npop} \
  --outfolder /scratch/users/genomic_times_series_bnp/structure/fits \
  --out_filename structure_fit_nobs${nobs}_nloci${nloci}_npop${npop}_alpha${alpha}
