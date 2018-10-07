import numpy as np
import scipy as sp

import autograd

import sys
sys.path.insert(0, './../../../LinearResponseVariationalBayes.py')
sys.path.insert(0, './../../src/vb_modeling/')

import os

import json

import LinearResponseVariationalBayes as vb
import LinearResponseVariationalBayes.ExponentialFamilies as ef

from numpy.polynomial.hermite import hermgauss

import gmm_clustering_lib as gmm_utils
import common_modeling_lib

import time

from copy import deepcopy
import json_tricks
from gmm_clustering_lib import np_string

from sklearn import datasets
from sklearn.cluster import KMeans

import argparse
import distutils.util

parser = argparse.ArgumentParser()
parser.add_argument('--outfolder', '-o', default='./')
parser.add_argument('--out_filename', default='iris_data_fit.json', type=str)
parser.add_argument('--seed', '-s', type=int, default=42)

parser.add_argument('--use_bnp_prior', type=distutils.util.strtobool, default='True')

parser.add_argument('--warm_start', type=distutils.util.strtobool, default='False')
parser.add_argument('--init_fit', type=str)

parser.add_argument('--alpha', type=float, default = 4.0)
parser.add_argument('--k_approx', type = int, default = 12)
parser.add_argument('--bootstrap_sample', type=distutils.util.strtobool, default='False')

args = parser.parse_args()

def validate_args():
    assert os.path.exists(args.outfolder)

    if args.warm_start:
        assert os.path.isfile(args.init_fit)

validate_args()

np.random.seed(args.seed)

iris = datasets.load_iris(return_X_y= True)

iris_features = iris[0]

demean = True
if demean:
    iris_features -= np.mean(iris_features, axis = 0)[None, :]

iris_species = iris[1]

dim = iris_features.shape[1]

prior_params = gmm_utils.get_default_prior_params(dim)
prior_params['alpha'].set(args.alpha)
prior_params['prior_gamma_df'].set(8)
prior_params['prior_gamma_inv_scale'].set(np.eye(dim) * 0.62)

print(prior_params)

gh_deg = 8
model = gmm_utils.DPGaussianMixture(iris_features, \
            args.k_approx, prior_params, gh_deg,
            use_logitnormal_sticks=True,
            use_bnp_prior = args.use_bnp_prior)

if args.bootstrap_sample:
    model.use_weights = True
    samples_indx = np.random.choice(model.n_obs,
                            size = model.n_obs,
                            replace = True); print('drawing boostrap samples')
    samples_indx, sample_indx_counts = np.unique(samples_indx,
                                                return_counts = True)

    model.weights = np.zeros((model.n_obs, 1))
    model.weights[samples_indx] = sample_indx_counts[:, None]

    # print(model.weights)
    # print(np.sum(model.weights))

if args.warm_start:
    with open(args.init_fit, 'r') as fp:
        fit_dict_init = json.load(fp)

    init_global_free_param = json_tricks.loads(
        fit_dict_init['vb_global_free_par' + np_string]); print('loading init fit from ', args.init_fit)

    # check the prior parameters
    reloaded_prior_params_vec = json_tricks.loads(
        fit_dict_init['prior_params_vec' + np_string])

    assert len(reloaded_prior_params_vec) == \
                len(model.prior_params.get_vector())
    assert np.max(np.abs(reloaded_prior_params_vec - \
                    model.prior_params.get_vector())) < 1e-8

    model.global_vb_params.set_free(init_global_free_param)
    model.set_optimal_z()

else:
    print('running k-means init')
    n_kmeans_init = 50
    init_global_free_param = \
            model.cluster_and_set_inits(n_kmeans_init = n_kmeans_init)

print('running Newton steps: ')
t0 = time.time()
best_param, kl_hessian, kl_hessian_corrected, \
    init_opt_time, newton_time, x_conv, f_conv, vb_opt = \
        model.optimize_full(init_global_free_param,
            init_max_iter=100,
            final_max_iter=500)

t_newton = time.time() - t0
print('done optimizing. Optim time: {}'.format(t_newton))

fit_dict = model.get_checkpoint_dictionary(seed=args.seed)

json_output_file = os.path.join(args.outfolder, args.out_filename)
print('saving fit dict to ', json_output_file)

with open(json_output_file, 'w') as outfile:
    json.dump(fit_dict, outfile)
