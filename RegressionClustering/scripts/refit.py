#!/usr/bin/env python3

"""Load a previous fit and refit, possibly with a new alpha.

This uses the output of the jupyter notebook InitialFit.

Example usage:

./refit.py \
    --fit_directory /home/rgiordan/Documents/git_repos/BNP_sensitivity/RegressionClustering/fits \
    --input_filename transformed_gene_regression_df4_degree3_genes700_num_components40_inflate0.0_shrunkTrue_alpha2.0_fit.npz \
    --alpha_scale 0.001
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

parser.add_argument('--out_filename', default=None, type=str)
# If the fit_directory argument is set, use that as the location for
# all datafiles.
parser.add_argument('--fit_directory', default=None, type=str)
parser.add_argument('--input_filename', required=True, type=str)
parser.add_argument('--alpha_scale', required=True, type=float)

# If ``functional`` is true, use a functional perturbation.
parser.add_argument('--functional', dest='functional', action='store_true')
parser.add_argument('--no-functional', dest='functional', action='store_false')
parser.set_defaults(functional=True)

args = parser.parse_args()

np.random.seed(args.seed)

if args.fit_directory is not None:
    if not os.path.isdir(args.fit_directory):
        raise ValueError('Fit directory {} does not exist'.format(
            args.fit_directory))

def set_directory(filename):
    # If the fit_directory argument is set, replace a datafile's directory
    # with the specified fit_directory and return the new location.
    if args.fit_directory is None:
        return filename
    else:
        _, file_only_name = os.path.split(filename)
        return os.path.join(args.fit_directory, file_only_name)

args.input_filename = set_directory(args.input_filename)
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
    datafile = set_directory(str(infile['datafile']))
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
         'num_components{}_inflate{}_shrunk{}_alphascale{}_refit').format(
        df, degree, num_genes, num_components, inflate_cov, eb_shrunk,
        args.alpha_scale)
    # Save in the same place as the input file.
    outdir, _ = os.path.split(args.input_filename)
    outfile = os.path.join(outdir, '{}.npz'.format(analysis_name))
else:
    # If out_filename is set, it overrides fit_directory.
    outfile = args.out_filename

outdir, _ = os.path.split(outfile)
if not os.path.exists(outdir):
    raise ValueError('Destination directory {} does not exist'.format(outdir))



# Define a perturbation.
if args.functional:
    # TODO: specify this from the command line somehow?
    log_phi_desc = 'expit'
    def log_phi(logit_v):
        return(sp.special.expit(logit_v))

    prior_pert = gmm_lib.PriorPerturbation(log_phi, gmm.gh_loc, gmm.gh_weights)
    gmm.set_perturbation_fun(prior_pert.get_e_log_perturbation)
    prior_pert.set_epsilon(args.alpha_scale)
else:
    new_alpha = prior_params['probs_alpha'] * args.alpha_scale
    new_prior_params = deepcopy(prior_params)
    new_prior_params['probs_alpha'][:] = new_alpha

gmm = gmm_lib.GMM(num_components, new_prior_params, reg_params)

# Re-optimize.

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
save_dict['functional'] = args.functional
if args.functional:
    save_dict['log_phi_desc'] = log_phi_desc
else:
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
