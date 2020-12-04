#!/bin/bash

# using a for loop instead of sbatch --array
# so each job has own ID 
# (makes it easier to track nodes and resources)

for i in {0..2}
do
   sbatch --export=job_indx=$i get_alpha_derivative.sh 
done
