import jax
from jax import numpy as np
from jax import scipy as sp
from jax import random

from numpy.polynomial.hermite import hermgauss

import time

import os
import argparse
import distutils.util

import paragami

# functional sensitivity library
from bnpmodeling_runjingdev import log_phi_lib

# BNP gmm libraries
import bnpgmm_runjingdev.gmm_clustering_lib as gmm_lib
from bnpgmm_runjingdev.gmm_optimization_lib import optimize_gmm
import bnpgmm_runjingdev.utils_lib as utils_lib


parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, default=42)

# where to save the gmm fit
parser.add_argument('--out_folder', default='../fits/')
parser.add_argument('--out_filename', type=str)

# the initial fit
parser.add_argument('--init_fit', type=str)

# which epsilon (index of bump size)
parser.add_argument('--epsilon_indx', type=int, default = 0)

# which mu (index of bump location)
parser.add_argument('--mu_indx', type=int, default = 0)

args = parser.parse_args()

assert os.path.exists(args.out_folder), args.out_folder
assert os.path.isfile(args.init_fit), args.init_fit

########################
# load iris data
########################
dataset_name = 'iris'
iris_obs, iris_species = utils_lib.load_iris_data()
dim = iris_obs.shape[1]
n_obs = len(iris_species)

iris_obs = np.array(iris_obs)

########################
# Variational parameters
########################
vb_init_dict, vb_params_paragami, meta_data = \
        paragami.load_folded(args.init_fit)

# gauss-hermite parameters
gh_deg = int(meta_data['gh_deg'])
gh_loc, gh_weights = hermgauss(gh_deg)

gh_loc = np.array(gh_loc)
gh_weights = np.array(gh_weights)

########################
# load prior parameters
########################
prior_params_dict, prior_params_paragami = \
    gmm_lib.get_default_prior_params(dim)

# set alpha
dp_prior_alpha = meta_data['dp_prior_alpha']
prior_params_dict['dp_prior_alpha'] = dp_prior_alpha

########################
# Functional perturbation 
########################
# set epsilon
epsilon_vec = np.linspace(0, 1, 8)[1:]**2 
epsilon = epsilon_vec[args.epsilon_indx]
print('epsilon = ', epsilon)
print('epsilon_indx = ', args.epsilon_indx)

# set mu 
mu_vec = np.arange(-6, 7)
mu = mu_vec[args.mu_indx]

print('mu = ', mu)
print('mu_indx = ', args.mu_indx)

assert args.epsilon_indx < len(epsilon_vec)
assert args.mu_indx < (len(mu_vec) - 1)

def e_step_bump(means, infos, mu_indx): 
    cdf1 = sp.stats.norm.cdf(mu_vec[mu_indx+1], loc = means, scale = 1 / np.sqrt(infos))
    cdf2 = sp.stats.norm.cdf(mu_vec[mu_indx], loc = means, scale = 1 / np.sqrt(infos))
    
    return (cdf1 - cdf2).sum()

e_log_phi = lambda means, infos : e_step_bump(means, infos, args.mu_indx) * epsilon

########################
# Optimize
########################
vb_opt_dict, _, e_z_opt, _, optim_time = \
    optimize_gmm(iris_obs,
                 vb_init_dict,
                 vb_params_paragami,
                 prior_params_dict,
                 gh_loc, gh_weights,
                 e_log_phi = e_log_phi, 
                 # just directly run Newton steps
                 run_lbfgs = False)

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
                     optim_time = optim_time, 
                     gh_deg = gh_deg, 
                     dp_prior_alpha = dp_prior_alpha, 
                     epsilon = epsilon, 
                     mu = mu)