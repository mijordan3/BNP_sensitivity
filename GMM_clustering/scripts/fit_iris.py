import jax
from jax import numpy as np
from jax import scipy as sp
from jax import random

from numpy.polynomial.hermite import hermgauss

import paragami

# BNP gmm libraries
import bnpgmm_runjingdev.gmm_clustering_lib as gmm_lib
import bnpgmm_runjingdev.gmm_cavi_lib as cavi_lib
import bnpgmm_runjingdev.utils_lib as utils_lib

import time

import os
import argparse
import distutils.util

parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, default=42)

# where to save the gmm fit
parser.add_argument('--out_folder', default='../fits/')
parser.add_argument('--out_filename', default='iris_fit', type=str)

# model parameters
parser.add_argument('--alpha', type=float, default = 4.0)
parser.add_argument('--k_approx', type = int, default = 15)

args = parser.parse_args()

assert os.path.exists(args.out_folder), args.out_folder

########################
# load iris data
########################
dataset_name = 'iris'
iris_obs, iris_species = utils_lib.load_iris_data()
dim = iris_obs.shape[1]
n_obs = len(iris_species)

iris_obs = np.array(iris_obs)

########################
# Get priors
########################
prior_params_dict, prior_params_paragami = gmm_lib.get_default_prior_params(dim)

# set initial alpha
prior_params_dict['alpha'] = args.alpha
print(prior_params_dict)

########################
# Variational parameters
########################

# Gauss-Hermite points for integrating logitnormal stick-breaking prior
gh_deg = 8
gh_loc, gh_weights = hermgauss(gh_deg)

# convert to jax arrays
gh_loc, gh_weights = np.array(gh_loc), np.array(gh_weights)

# get vb parameters
_, vb_params_paragami = gmm_lib.get_vb_params_paragami_object(dim, args.k_approx)

# run a kmeans init
print('initializing with k-means ...')
n_kmeans_init = 50
_, init_vb_params_dict, init_ez = \
    utils_lib.cluster_and_get_k_means_inits(iris_obs,
                                            vb_params_paragami, 
                                            n_kmeans_init = n_kmeans_init, 
                                            seed = args.seed)

########################
# Optimize
########################
x_tol = 1e-3
vb_opt_dict, e_z_opt, cavi_time = \
    cavi_lib.run_cavi(iris_obs,
                     init_vb_params_dict,
                     vb_params_paragami,
                     prior_params_dict,
                     gh_loc, gh_weights,
                     debug = False, 
                     x_tol = x_tol)

final_kl = gmm_lib.get_kl(iris_obs,
                          vb_opt_dict,
                          prior_params_dict,
                          gh_loc,
                          gh_weights,
                          e_z = e_z_opt)

#####################
# Save results
#####################
outfile = os.path.join(args.out_folder, args.out_filename)
print('saving iris fit to ', outfile)

paragami.save_folded(outfile, 
                     vb_opt_dict,
                     vb_params_paragami, 
                     final_kl = final_kl, 
                     optim_time = cavi_time, 
                     gh_deg = gh_deg, 
                     alpha = args.alpha)

                     