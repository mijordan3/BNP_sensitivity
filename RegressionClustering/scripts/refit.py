#!/usr/bin/env python3

"""Load a previous fit and refit, possibly with a new alpha.

This uses the output of the jupyter notebook InitialFit.

Example usage:

./refit.py \
    --input_filename /home/rgiordan/Documents/git_repos/BNP_sensitivity/RegressionClustering/jupyter/fits/transformed_gene_regression_df4_degree3_genes700_num_components30_inflate1.0_shrunkTrue_fit.npz \
    --alpha_scale 0.01
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
parser.add_argument('--input_filename', required=True, type=str)
parser.add_argument('--alpha_scale', default=None, type=float)

args = parser.parse_args()

np.random.seed(args.seed)

if not os.path.isfile(args.input_filename):
    raise ValueError('Initial fit file {} does not exist'.format(
        args.input_filename))

if args.alpha_scale < 0:
    raise ValueError('alpha_scale must be greater than zero.')

# Load the data.

with np.load(args.input_filename) as infile:
    gmm_params_pattern = paragami.get_pattern_from_json(
        str(infile['gmm_params_pattern_json']))
    opt_gmm_params = gmm_params_pattern.fold(
        infile['opt_gmm_params_flat'], free=False)
    prior_params_pattern = paragami.get_pattern_from_json(
        str(infile['prior_params_pattern_json']))
    prior_params = prior_params_pattern.fold(
        infile['prior_params_flat'], free=False)
    kl_hess = infile['kl_hess']
    df = infile['df']
    degree = infile['degree']
    datafile = str(infile['datafile'])
    num_components = int(infile['num_components'])

if not os.path.isfile(datafile):
    raise ValueError('Datafile {} does not exist'.format(datafile))

reg_params = dict()
with np.load(datafile) as infile:
    reg_params['beta_mean'] = infile['transformed_beta_mean']
    reg_params['beta_info'] = infile['transformed_beta_info']
    inflate_cov = infile.get('inflate_cov', 0)
    eb_shrunk = infile.get('eb_shrunk', False)

num_genes = reg_params['beta_mean'].shape[0]

if args.out_filename is None:
    analysis_name = \
        ('transformed_gene_regression_df{}_degree{}_genes{}_' +
         'num_components{}_inflate{}_shrunk{}_refit').format(
        df, degree, num_genes, num_components, inflate_cov, eb_shrunk)
    outdir, _ = os.path.split(args.input_filename)
    outfile = os.path.join(outdir, '{}.npz'.format(analysis_name))
else:
    outfile = args.out_filename

outdir, _ = os.path.split(outfile)
if not os.path.exists(outdir):
    raise ValueError('Destination directory {} does not exist'.format(outdir))


# Re-optimize.

new_alpha = prior_params['probs_alpha'] * args.alpha_scale
new_prior_params = deepcopy(prior_params)
new_prior_params['probs_alpha'][:] = new_alpha

gmm = gmm_lib.GMM(num_components, new_prior_params, reg_params)

print('Setting preconditioner...')
gmm.get_kl_conditioned.set_preconditioner_with_hessian(
    hessian=kl_hess, ev_min=1e-6)
print('Done.')

print('Optimizing...')
init_x = gmm.gmm_params_pattern.flatten(opt_gmm_params, free=True)
gmm.conditioned_obj.reset()
tic = time.time()
gmm_opt, gmm_opt_x = gmm.optimize_fully(
    init_x, verbose=True, kl_hess=kl_hess)
opt_time = time.time() - tic
print('Done.')
print('Re-optimization time: {} seconds'.format(opt_time))

reopt_gmm_params = gmm.gmm_params_pattern.fold(gmm_opt_x, free=True)

# Save the refit
print('Saving to {}'.format(outfile))

save_dict = dict()

save_dict['input_filename'] = args.input_filename
save_dict['alpha_scale'] = args.alpha_scale
save_dict['new_alpha'] = new_alpha
save_dict['reopt_gmm_params_flat'] = \
    gmm.gmm_params_pattern.flatten(reopt_gmm_params, free=False)
save_dict['reopt_time'] = opt_time
save_dict['gmm_params_pattern_json'] = gmm.gmm_params_pattern.to_json()
save_dict['reopt_kl_hess'] = gmm_opt['kl_hess']
save_dict['reopt_prior_params_flat'] = \
    prior_params_pattern.flatten(gmm.prior_params, free=False)
save_dict['reopt_prior_params_pattern_json'] = \
    prior_params_pattern.to_json()

np.savez_compressed(file=outfile, **save_dict)
