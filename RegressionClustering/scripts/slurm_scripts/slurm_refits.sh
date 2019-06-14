#!/bin/bash

# Run a selection of refit scripts with slurm.

FIT_DIR="/accounts/grad/rgiordano/Documents/git_repos/BNP_sensitivity/RegressionClustering/"

for FIT_SCRIPT in $(ls ./*genes700_inflate1.0*.sh); do
    echo; echo; echo ------------------------------
    echo Running $FIT_SCRIPT
    sbatch $FIT_SCRIPT
done
