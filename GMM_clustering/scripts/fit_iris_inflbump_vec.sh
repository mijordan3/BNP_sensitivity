#!/bin/bash

for i in {0..8}
do
   sbatch --array 0-8 --export=mu_indx=$i fit_iris_inflbump.sh
done
