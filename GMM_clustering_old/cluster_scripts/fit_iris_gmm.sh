#!/bin/bash

source activate genomic_time_series_py35

seed=345345

/usr/bin/env python3 ./estimate_iris_gmm.py \
  --outfolder /scratch/users/genomic_times_series_bnp/iris_fits/full_data_fits/\
  --out_filename iris_nclusters${n_clusters}_full_data_fit.json\
  --use_bnp_prior False \
  --bootstrap_sample False \
  --seed ${seed} \
  --k_approx ${n_clusters}
