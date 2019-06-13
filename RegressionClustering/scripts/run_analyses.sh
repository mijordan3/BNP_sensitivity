#!/bin/bash

# Run a number of initial fits on which to base our results.

FIT_DIR="/home/rgiordan/Documents/git_repos/BNP_sensitivity/RegressionClustering/fits"

for REFIT_FILENAME in $(ls $FIT_DIR/*700_*refit.npz); do
    echo; echo; echo ------------------------------
    echo Running for $REFIT_FILENAME
    ./analyze_refit.py \
        --refit_filename $REFIT_FILENAME
done
