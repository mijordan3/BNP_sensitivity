#!/bin/bash
for i in {0..2}
do
   sbatch --export=job_indx=$i get_alpha_derivative.sh 
done
