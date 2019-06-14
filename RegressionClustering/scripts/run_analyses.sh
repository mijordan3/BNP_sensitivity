#!/bin/bash

# Run a number of initial fits on which to base our results.

FIT_DIR="/home/rgiordan/Documents/git_repos/BNP_sensitivity/RegressionClustering/fits"

#GENES=7000
GENES=700

#INFLATE=0.0
INFLATE=1.0

for REFIT_FILENAME in $(ls $FIT_DIR/*genes${GENES}*_inflate${INFLATE}_*refit.npz); do
    echo; echo; echo ------------------------------
    echo Running for $REFIT_FILENAME
    ./analyze_refit.py \
        --refit_filename $REFIT_FILENAME
done