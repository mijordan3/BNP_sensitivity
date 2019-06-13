#!/usr/bin/env python3

"""Fit a clustering model at a particular alpha.

This uses the output of the jupyter notebook PreprocessData.

Example usage:

./initial_fit.py \
    --data_filename /home/rgiordan/Documents/git_repos/BNP_sensitivity/RegressionClustering/jupyter/fits/shrunken_transformed_gene_regression_df4_degree3_genes1000_inflate0.0.npz \
    --alpha 2.0 \
    --num_components 40
"""

import argparse

import numpy as np
import inspect
import os
import sys
import time

import paragami
import vittles

from copy import deepcopy

import bnpregcluster_runjingdev.regression_mixture_lib as gmm_lib

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=42)

# results folders
parser.add_argument('--out_filename', default=None, type=str)

# Specify either the initinal fit file or the df, degree, and num_components.
parser.add_argument('--data_filename', required=True, type=str)
parser.add_argument('--alpha', required=True, type=float)
parser.add_argument('--num_components', required=True, type=int)

args = parser.parse_args()

np.random.seed(args.seed)

if not os.path.isfile(args.data_filename):
    raise ValueError('Data file {} does not exist'.format(
        args.input_filename))

if args.alpha < 0:
    raise ValueError('``alpha`` must be greater than zero.')

if args.num_components < 2:
    raise ValueError('``num_components`` must be at least 2.')

# Load the data.

reg_params = dict()
with np.load(args.data_filename) as infile:
    reg_params['y_info'] = infile['y_info']
    reg_params['beta_mean'] = infile['transformed_beta_mean']
    reg_params['beta_info'] = infile['transformed_beta_info']
    df = infile['df']
    degree = infile['degree']
    inflate_cov = infile.get('inflate_cov', 0)
    eb_shrunk = infile.get('eb_shrunk', False)

num_genes = reg_params['beta_mean'].shape[0]
obs_dim = reg_params['beta_mean'].shape[1]

if args.out_filename is None:
    analysis_name = \
        ('transformed_gene_regression_' +
         'df{}_degree{}_genes{}_num_components{}_' +
         'inflate{}_shrunk{}_alpha{}_fit').format(
        df, degree, num_genes, args.num_components, inflate_cov, eb_shrunk,
        args.alpha)
    outdir, _ = os.path.split(args.data_filename)
    outfile = os.path.join(outdir, '{}.npz'.format(analysis_name))
else:
    outfile = args.out_filename

outdir, _ = os.path.split(outfile)
if not os.path.exists(outdir):
    raise ValueError('Destination directory {} does not exist'.format(outdir))


# Fit
prior_params = gmm_lib.get_base_prior_params(obs_dim, args.num_components)
prior_params['probs_alpha'][:] = args.alpha

prior_params_pattern = gmm_lib.get_prior_params_pattern(
    obs_dim, args.num_components)
prior_params_pattern.validate_folded(prior_params)
gmm = gmm_lib.GMM(args.num_components, prior_params, reg_params)


print('Running k-means init.')
kmeans_params = \
    gmm_lib.kmeans_init(gmm.reg_params,
                        gmm.num_components, 50)
print('Done.')
init_gmm_params = dict()
init_gmm_params['centroids'] = kmeans_params['centroids']
init_gmm_params['stick_propn_mean'] = np.zeros(gmm.num_components - 1)
init_gmm_params['stick_propn_info'] = np.ones(gmm.num_components - 1)

init_x = gmm.gmm_params_pattern.flatten(init_gmm_params, free=True)


print('Getting initial optimum.')
gmm.conditioned_obj.reset() # Reset the logging and iteration count.
gmm.conditioned_obj.set_print_every(1)

opt_time = time.time()
gmm_opt, init_x2 = gmm.optimize(init_x, gtol=1e-2, maxiter=50)
opt_time = time.time() - opt_time


print('Getting Hessian for preconditioning.')
tic = time.time()
# Note that h_cond is the Hessian.
h_cond = gmm.update_preconditioner(init_x2)
opt_time += time.time() - tic


print('Completely optimizing with preconditioning.')
gmm.conditioned_obj.reset()
tic = time.time()
gmm_opt, gmm_opt_x = gmm.optimize_fully(
    init_x2, verbose=True, kl_hess=h_cond)
opt_time += time.time() - tic

opt_gmm_params = gmm.gmm_params_pattern.fold(gmm_opt_x, free=True)

print('Done!')
print('Optimization time: {} seconds'.format(opt_time))

print('Saving to {}'.format(outfile))

save_dict = deepcopy(gmm_opt)
save_dict['df'] = df
save_dict['degree'] = degree
save_dict['datafile'] = args.data_filename
save_dict['num_components'] = args.num_components
save_dict['gmm_params_pattern_json'] = \
    gmm.gmm_params_pattern.to_json()
save_dict['opt_gmm_params_flat'] = \
    gmm.gmm_params_pattern.flatten(opt_gmm_params, free=False)
save_dict['prior_params_pattern_json'] = \
    prior_params_pattern.to_json()
save_dict['prior_params_flat'] = \
    prior_params_pattern.flatten(prior_params, free=False)

save_dict['opt_time'] = opt_time

np.savez_compressed(file=outfile, **save_dict)
