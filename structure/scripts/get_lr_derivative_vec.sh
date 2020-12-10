#!/bin/bash

# using a for loop instead of sbatch --array
# so each job has own ID 
# (makes it easier to track nodes and resources)


# sbatch --export=job_indx=0 get_lr_derivative.sh
sbatch --export=job_indx=1 get_lr_derivative.sh
# sbatch --export=job_indx=2 get_lr_derivative.sh

# for i in {0..2}
# do
#    sbatch --export=job_indx=$i get_lr_derivative.sh 
# done
