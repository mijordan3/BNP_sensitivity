#!/bin/bash
source ../../venv/bin/activate
../refit.py --fit_directory /home/rgiordan/Documents/git_repos/BNP_sensitivity/RegressionClustering/fits --input_filename /home/rgiordan/Documents/git_repos/BNP_sensitivity/RegressionClustering/fits/transformed_gene_regression_df4_degree3_genes7000_num_components60_inflate0.0_shrunkTrue_alpha2.0_fit.npz --alpha_scale 0.889 --functional --log_phi_desc expit
