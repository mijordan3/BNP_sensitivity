#!/bin/bash

for mu_indx in {0..11}
do
   sbatch --array 0-8 --export=mu_indx=$mu_indx fit_iris_inflbump.sh
done
