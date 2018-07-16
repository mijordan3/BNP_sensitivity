#!/bin/bash

for n_clusters in 2; do
  echo $n_clusters 
  sbatch --export=n_clusters=${n_clusters} \
  fit_iris_gmm.sh
done;
