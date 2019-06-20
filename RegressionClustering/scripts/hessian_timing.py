#!/usr/bin/env python3

"""Load a previous fit and calculate how much time it takes to calculate the
Hessian.  Unfortunately, this wasn't saved separately in the initial fit.

This uses the output of initial_fit.py.

Example usage:

./hessian_timing.py \
    --fit_directory /home/rgiordan/Documents/git_repos/BNP_sensitivity/RegressionClustering/fits \
    --input_filename transformed_gene_regression_df4_degree3_genes7000_num_components60_inflate0.0_shrunkTrue_alpha2.0_fit.npz
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

# If the fit_directory argument is set, use that as the location for
# all datafiles.
parser.add_argument('--fit_directory', default=None, type=str)
parser.add_argument('--input_filename', required=True, type=str)

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

gmm = gmm_lib.GMM(num_components, prior_params, reg_params)

print('Caculating Hessian...')
hess_time = time.time()
init_x = gmm.gmm_params_pattern.flatten(opt_gmm_params, free=True)
kl_hess = gmm.kl_obj.hessian(init_x)
hess_time = time.time() - hess_time
print('Done.')

print('Hessian time: {}s'.format(hess_time))
