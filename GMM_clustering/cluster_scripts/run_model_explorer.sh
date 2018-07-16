#!/bin/bash

for n_clusters in 2 3 4 5 6 7 8 9 10; do
  echo $n_clusters 
  sbatch --array 1-200 \
  --export=n_clusters=${n_clusters} \
  fit_iris_gmm_bootstrap.sh
done;
