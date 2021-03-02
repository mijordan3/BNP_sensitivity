import jax
from jax import numpy as np
from jax import scipy as sp
from jax import random

from numpy.polynomial.hermite import hermgauss

import paragami

# BNP regression mixture libraries
from bnpreg_runjingdev import genomics_data_utils
from bnpreg_runjingdev import regression_mixture_lib
from bnpreg_runjingdev.regression_optimization_lib import optimize_regression_mixture, \
    set_params_w_kmeans

import time

import os
import argparse
import distutils.util

parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, default=42)

# Set bnp_data_repo to be the location of a clone of the repo
# https://github.com/NelleV/genomic_time_series_bnp
parser.add_argument('--bnp_data_repo', type=str, 
                    default = '../../../genomic_time_series_bnp')

# where to save the gmm fit
parser.add_argument('--out_folder', default='../fits/')
parser.add_argument('--out_filename', default='mice_genomics_fit', type=str)

# model parameters
parser.add_argument('--alpha', type=float, default = 6.0)
parser.add_argument('--k_approx', type = int, default = 40)

args = parser.parse_args()

assert os.path.exists(args.out_folder), args.out_folder

########################
# load mice regression data
########################
genome_data, timepoints, regressors, beta, beta_infos, y_infos = \
    genomics_data_utils.load_data_and_run_regressions(args.bnp_data_repo)

n_genes = genome_data.shape[0]
reg_dim = regressors.shape[1]

n_timepoints = len(np.unique(timepoints))

########################
# Get priors
########################
prior_params_dict, prior_params_paragami = regression_mixture_lib.get_default_prior_params()

# set initial alpha
prior_params_dict['dp_prior_alpha'] = args.alpha
print(prior_params_dict)

########################
# Variational parameters
########################

# get vb parameters
vb_params_dict, vb_params_paragami = \
    regression_mixture_lib.get_vb_params_paragami_object(reg_dim, args.k_approx)

# Gauss-Hermite points for integrating logitnormal stick-breaking prior
gh_deg = 8
gh_loc, gh_weights = hermgauss(gh_deg)

# convert to jax arrays
gh_loc, gh_weights = np.array(gh_loc), np.array(gh_weights)

########################
# Optimize
########################
vb_params_dict = set_params_w_kmeans(genome_data,
                                     regressors,
                                     vb_params_dict, 
                                     vb_params_paragami, 
                                     prior_params_dict,
                                     gh_loc, gh_weights, 
                                     seed = args.seed)                     

vb_opt_dict, vb_opt, ez_opt, out, optim_time = \
    optimize_regression_mixture(genome_data, regressors, 
                                vb_params_dict,
                                vb_params_paragami,
                                prior_params_dict, 
                                gh_loc, 
                                gh_weights)

final_kl = out.fun

# #####################
# # Save results
# #####################
outfile = os.path.join(args.out_folder, args.out_filename)
print('saving gmm regression fit to ', outfile)

paragami.save_folded(outfile, 
                     vb_opt_dict,
                     vb_params_paragami, 
                     final_kl = final_kl, 
                     optim_time = optim_time, 
                     gh_deg = gh_deg, 
                     dp_prior_alpha = args.alpha)


# paragami.save_folded(outfile, 
#                      vb_opt_dict,
#                      vb_params_paragami, 
#                      genome_data = genome_data, 
#                      regressors = regressors, 
#                      gh_loc = gh_loc, 
#                      gh_weights = gh_weights,
#                      prior_free = prior_params_paragami.flatten(prior_params_dict, free = True), 
#                      init_free = vb_params_paragami.flatten(vb_params_init, free = True))
