#!/bin/bash

for n_clusters in 12; do
  echo $n_clusters 
  sbatch --array 1-200 \
  --export=n_clusters=${n_clusters} \
  fit_bnp_gmm_bootstrap.sh
done;
