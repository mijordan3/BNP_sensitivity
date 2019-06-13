#!/usr/bin/env python3

"""Load a refit and compare to the Taylor series approximation for a range of
posterior quantites.

Example usage:

./analyze_refit.py \
    --refit_filename /home/rgiordan/Documents/git_repos/BNP_sensitivity/RegressionClustering/fits/transformed_gene_regression_df4_degree3_genes700_num_components40_inflate0.0_shrunkTrue_alphascale100.0_refit.npz
"""

import argparse

import numpy as np
import inspect
import json_tricks
import os
import sys
import time

import paragami
import vittles

from copy import deepcopy

import bnpregcluster_runjingdev.regression_mixture_lib as gmm_lib
import bnpregcluster_runjingdev.posterior_quantities_lib as post_lib

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=42)

# refit results
parser.add_argument('--refit_filename', required=True, type=str)
parser.add_argument('--out_filename', default=None, type=str)
parser.add_argument('--taylor_order', default=1, type=int)

args = parser.parse_args()

np.random.seed(args.seed)

if not os.path.isfile(args.refit_filename):
    raise ValueError('Refit file {} does not exist'.format(
        args.refit_filename))

if args.taylor_order < 1:
    raise ValueError('``taylor_order`` must be greater than zero.')

with np.load(args.refit_filename) as infile:
    initial_fitfile = str(infile['input_filename'])
    gmm_params_pattern = paragami.get_pattern_from_json(
        str(infile['gmm_params_pattern_json']))
    reopt_gmm_params = gmm_params_pattern.fold(
        infile['reopt_gmm_params_flat'], free=False)
    prior_params_pattern = paragami.get_pattern_from_json(
        str(infile['reopt_prior_params_pattern_json']))
    reopt_prior_params = prior_params_pattern.fold(
        infile['reopt_prior_params_flat'], free=False)
    reopt_time = infile['reopt_time']
    alpha_scale = infile['alpha_scale']

if not os.path.isfile(initial_fitfile):
    raise ValueError('Initial fit {} not found'.format(initial_fitfile))

with np.load(initial_fitfile) as infile:
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
    raise ValueError('Datafile {} not found'.format(datafile))

reg_params = dict()
with np.load(datafile) as infile:
    reg_params['beta_mean'] = infile['transformed_beta_mean']
    reg_params['beta_info'] = infile['transformed_beta_info']
    inflate_cov = infile.get('inflate_cov', 0)
    eb_shrunk = infile.get('eb_shrunk', False)

orig_alpha =  np.unique(prior_params['probs_alpha'])
new_alpha =  np.unique(reopt_prior_params['probs_alpha'])
print('New alpha: ', new_alpha)
print('Old alpha: ', orig_alpha)

num_genes = reg_params['beta_mean'].shape[0]

gmm = gmm_lib.GMM(num_components, reopt_prior_params, reg_params)


if args.out_filename is None:
    analysis_name = \
        ('transformed_gene_regression_df{}_degree{}_genes{}_' +
         'num_components{}_inflate{}_shrunk{}_alphascale{}_analysis').format(
        df, degree, num_genes, num_components, inflate_cov, eb_shrunk,
        alpha_scale)
    outdir, _ = os.path.split(args.refit_filename)
    # Note!  Not npz this time.
    outfile = os.path.join(outdir, '{}.json'.format(analysis_name))
else:
    outfile = args.out_filename

outdir, _ = os.path.split(outfile)
if not os.path.exists(outdir):
    raise ValueError('Destination directory {} does not exist'.format(outdir))

# Set up the approximation
prior_free = False
taylor_order = args.taylor_order

get_kl_from_vb_free_prior_free = \
    paragami.FlattenFunctionInput(original_fun=
        gmm.get_params_prior_kl,
        patterns = [gmm.gmm_params_pattern, prior_params_pattern],
        free = [True, prior_free],
        argnums = [0, 1])


vb_sens = \
    vittles.ParametricSensitivityTaylorExpansion(
        objective_function =    get_kl_from_vb_free_prior_free,
        input_val0 =            gmm.gmm_params_pattern.flatten(opt_gmm_params, free=True),
        hyper_val0 =            prior_params_pattern.flatten(prior_params, free=prior_free),
        order =                 taylor_order,
        hess0 =                 kl_hess)

predict_gmm_params = \
    paragami.FoldFunctionInputAndOutput(
        original_fun=vb_sens.evaluate_taylor_series,
        input_patterns=prior_params_pattern,
        input_free=prior_free,
        input_argnums=[0],
        output_patterns=gmm.gmm_params_pattern,
        output_free=True,
        output_retnums=[0])

# Get a range of posterior quantities

n_samples = 10000

results = []

for threshold in np.arange(0, 10):
    for predictive in [True, False]:

        threshold = int(threshold)
        get_posterior_quantity = post_lib.get_posterior_quantity_function(
            predictive, gmm, n_samples, threshold)

        lr_time = time.time()
        pred_gmm_params = predict_gmm_params(reopt_prior_params)
        lr_time = lr_time - time.time()

        e_num0 = get_posterior_quantity(opt_gmm_params)
        e_num1 = get_posterior_quantity(reopt_gmm_params)
        e_num_pred = get_posterior_quantity(pred_gmm_params)

        print(('\n----------------------\n' +
               'Predictive: {}\tthreshold: {}\n').format(
            predictive, threshold))
        print(('Orig e: \t{}\nRefit e:\t{}\nPred e:\t\t{}\n' +
              'Actual diff:\t{:0.5}\nPred diff:\t{:0.5}').format(
                e_num0, e_num1, e_num_pred,
                e_num1 - e_num0,
                e_num_pred - e_num0))

        # Save in a redundant format that will be easily converted to a
        # tidy dataframe.
        results.append(
            { 'n_samples': n_samples,
              'threshold': threshold,
              'predictive': predictive,
              'taylor_order': taylor_order,
              'prior_free': prior_free,
              'alpha1': new_alpha,
              'alpha0': orig_alpha,
              'e_num0': e_num0,
              'e_num1': e_num1,
              'e_num_pred': e_num_pred,
              'lr_time': lr_time,
              'refit_time': reopt_time,
              'refit_filename': args.refit_filename })

with open(outfile, 'w') as outfile:
    outfile.write(json_tricks.dumps(results))
