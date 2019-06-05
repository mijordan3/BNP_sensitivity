#!/bin/bash

source activate genomic_time_series_py35

seed=453453
n_clusters=3

/usr/bin/env python3 ./estimate_iris_gmm.py \
  --outfolder /scratch/users/genomic_times_series_bnp/iris_fits/full_data_fits/\
  --out_filename iris_bnp_full_data_fit.json\
  --use_bnp_prior True \
  --bootstrap_sample False \
  --seed ${seed} \
  --k_approx ${n_clusters}
