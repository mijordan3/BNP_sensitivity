#!/bin/bash

source activate genomic_time_series_py35

seed=$(( $SLURM_ARRAY_TASK_ID + 78793 * ${n_clusters}))

/usr/bin/env python3 ./estimate_iris_gmm.py \
  --outfolder /scratch/users/genomic_times_series_bnp/iris_fits/bootstrap_bnp_results/\
  --out_filename iris_bnp_boostr_trial$SLURM_ARRAY_TASK_ID.json\
  --use_bnp_prior True \
  --bootstrap_sample True \
  --warm_start True \
  --init_fit /scratch/users/genomic_times_series_bnp/iris_fits/full_data_fits/iris_bnp_full_data_fit.json\
  --seed ${seed} \
  --k_approx ${n_clusters}
