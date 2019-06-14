#!/bin/bash

# Run a number of initial fits on which to base our results.

#FIT_DIR="/home/rgiordan/Documents/git_repos/BNP_sensitivity/RegressionClustering/fits"
FIT_DIR="/accounts/grad/rgiordano/Documents/git_repos/BNP_sensitivity/RegressionClustering/fits/"

#GENES=7000
GENES=700

#INFLATE=0.0
INFLATE=1.0

for REFIT_FILENAME in $(ls $FIT_DIR/*genes${GENES}*_inflate${INFLATE}_*refit.npz); do
    echo; echo; echo ------------------------------
    echo Running for $REFIT_FILENAME
    SCRIPT_FILENAME=$(mktemp)".sh"
    echo $SCRIPT_FILENAME
    echo ./analyze_refit.py --fit_directory $FIT_DIR --refit_filename $REFIT_FILENAME > $SCRIPT_FILENAME
    sbatch $SCRIPT_FILENAME
done
