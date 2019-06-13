#!/usr/bin/env python3

"""Load a previous fit and refit, possibly with a new alpha.

This uses the output of the jupyter notebook InitialFit.
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
parser.add_argument('--alpha_scale', default=None, type=int)

args = parser.parse_args()

np.random.seed(args.seed)

outdir, _ = os.path.split(args.out_filename)
if not os.path.exists(outdir):
    raise ValueError('Destination directory {} does not exist'.format(outdir))

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


# TODO: finish saving the data
save_dict = deepcopy(gmm_opt)
save_dict['input_filename'] = args.input_filename
# save_dict['datafile'] = datafile
# save_dict['num_components'] = num_components
# save_dict['gmm_params_pattern_json'] = \
#     gmm.gmm_params_pattern.to_json()
# save_dict['opt_gmm_params_flat'] = \
#     gmm.gmm_params_pattern.flatten(opt_gmm_params, free=False)
# save_dict['prior_params_pattern_json'] = \
#     prior_params_pattern.to_json()
# save_dict['prior_params_flat'] = \
#     prior_params_pattern.flatten(prior_params, free=False)
#
# save_dict['opt_time'] = opt_time
#
# outfile = './fits/{}.npz'.format(analysis_name)
# print('Saving to {}'.format(outfile))
#
# np.savez_compressed(file=outfile, **save_dict)
