#!/bin/bash

# Run a selection of refit scripts.

FIT_DIR="/home/rgiordan/Documents/git_repos/BNP_sensitivity/RegressionClustering/fits"

for FIT_SCRIPT in $(ls ./*genes700_inflate1.0*.sh); do
    echo; echo; echo ------------------------------
    echo Running $FIT_SCRIPT
    source $FIT_SCRIPT &
done
