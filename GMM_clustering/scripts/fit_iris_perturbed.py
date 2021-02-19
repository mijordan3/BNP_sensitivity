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
import bnpgmm_runjingdev.gmm_cavi_lib as cavi_lib
import bnpgmm_runjingdev.utils_lib as utils_lib


parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, default=42)

# where to save the gmm fit
parser.add_argument('--out_folder', default='../fits/')
parser.add_argument('--out_filename', type=str)

# the initial fit
parser.add_argument('--init_fit', type=str)

# which epsilon 
parser.add_argument('--epsilon_indx', type=int, default = 0)

# delta 
parser.add_argument('--delta', type=float, default = 1.0)

# which perturbation
parser.add_argument('--perturbation', type=str, default = 'sigmoidal')

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

# set initial alpha
alpha0 = meta_data['alpha']
prior_params_dict['alpha'] = alpha0

########################
# Functional perturbation 
########################
# set epsilon
epsilon_vec = np.linspace(0, 1, 20)[1:]**2 
epsilon = epsilon_vec[args.epsilon_indx]
print('epsilon = ', epsilon)
print('epsilon_indx = ', args.epsilon_indx)

# define perturbation
f_obj_all = log_phi_lib.LogPhiPerturbations(vb_params_paragami, 
                                            alpha0,
                                            gh_loc, 
                                            gh_weights,
                                            logit_v_grid = None, 
                                            influence_grid = None,
                                            delta = args.delta, 
                                            stick_key = 'stick_params')

f_obj = getattr(f_obj_all, 'f_obj_' + args.perturbation)
e_log_phi = lambda means, infos : f_obj.e_log_phi_epsilon(means, infos, epsilon)

########################
# Optimize
########################
x_tol = 1e-3
vb_opt_dict, e_z_opt, cavi_time = \
    cavi_lib.run_cavi(iris_obs,
                     vb_init_dict,
                     vb_params_paragami,
                     prior_params_dict,
                     gh_loc, gh_weights,
                     e_log_phi = e_log_phi,
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
                     alpha = alpha0, 
                     epsilon = epsilon, 
                     delta = args.delta, 
                     perturbation = args.perturbation)

                     