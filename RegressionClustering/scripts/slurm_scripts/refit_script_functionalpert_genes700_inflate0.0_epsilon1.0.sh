#!/bin/bash
source ../../venv/bin/activate
../refit.py --fit_directory /home/rgiordan/Documents/git_repos/BNP_sensitivity/RegressionClustering/fits --input_filename /home/rgiordan/Documents/git_repos/BNP_sensitivity/RegressionClustering/fits/transformed_gene_regression_df4_degree3_genes700_num_components40_inflate0.0_shrunkTrue_alpha2.0_fit.npz --alpha_scale 1.0 --functional --log_phi_desc expit
