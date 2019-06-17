#!/bin/bash
1;4205;0c
# Run a selection of refit scripts with slurm.

FIT_DIR="/accounts/grad/rgiordano/Documents/git_repos/BNP_sensitivity/RegressionClustering/fits"

#for FIT_SCRIPT in $(ls ./*genes700_inflate?.0*.sh); do
#for FIT_SCRIPT in $(ls ./*genes7000_inflate?.0*.sh); do
for FIT_SCRIPT in $(ls ./*functionalpert_*genes700*_inflate?.0*.sh); do
    echo; echo; echo ------------------------------
    echo Running $FIT_SCRIPT
    sbatch $FIT_SCRIPT
done
