#!/bin/bash

# Run a number of initial fits on which to base our results.

FIT_DIR="/home/rgiordan/Documents/git_repos/BNP_sensitivity/RegressionClustering/fits"

./initial_fit.py \
    --data_filename $FIT_DIR/\
shrunken_transformed_gene_regression_df4_degree3_genes10000_inflate0.0.npz \
    --alpha 2.0 \
    --num_components 60 &

./initial_fit.py \
    --data_filename $FIT_DIR/\
shrunken_transformed_gene_regression_df4_degree3_genes10000_inflate1.0.npz \
    --alpha 2.0 \
    --num_components 60 &

./initial_fit.py \
    --data_filename $FIT_DIR/\
shrunken_transformed_gene_regression_df4_degree3_genes1000_inflate0.0.npz \
    --alpha 2.0 \
    --num_components 40 &

./initial_fit.py \
    --data_filename $FIT_DIR/\
shrunken_transformed_gene_regression_df4_degree3_genes1000_inflate1.0.npz \
    --alpha 2.0 \
    --num_components 40 &
