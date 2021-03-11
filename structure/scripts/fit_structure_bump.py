import jax

import jax.numpy as np
import jax.scipy as sp

import numpy as onp
from numpy.polynomial.hermite import hermgauss

import structure_vb_lib.structure_model_lib as structure_model_lib
import structure_vb_lib.structure_optimization_lib as s_optim_lib
from structure_vb_lib import data_utils

from bnpmodeling_runjingdev import log_phi_lib

import paragami

import argparse
import distutils.util

import os

import time

parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, default=42)

# data file
parser.add_argument('--data_file', type=str)

# where to save the structure fit
parser.add_argument('--out_folder', default='../fits/')
parser.add_argument('--out_filename', default='structure_fit', type=str)

# the initial fit
parser.add_argument('--init_fit', type=str)

# which epsilon 
parser.add_argument('--epsilon_indx', type=int, default = 0)

# which mu
parser.add_argument('--mu_indx', type=int, default = 0)

args = parser.parse_args()

def validate_args():
    assert os.path.exists(args.out_folder), args.out_folder
    assert os.path.isfile(args.init_fit), args.init_fit
    assert os.path.isfile(args.data_file), args.data_file

validate_args()

onp.random.seed(args.seed)

##################
# Load data
##################
g_obs = data_utils.load_thrush_data(args.data_file)[0]

print(g_obs.shape)

n_obs = g_obs.shape[0]
n_loci = g_obs.shape[1]
n_allele = g_obs.shape[-1]

##################
# Load initial fit
##################
print('loading fit from ', args.init_fit)
vb_params_dict, vb_params_paragami, \
    prior_params_dict, prior_params_paragami, \
        gh_loc, gh_weights, fit_meta_data = \
            structure_model_lib.load_structure_fit(args.init_fit)

print(prior_params_dict)

##################
# Define perturbation
##################
epsilon_vec = np.linspace(0, 1, 10)**2
assert args.epsilon_indx < len(epsilon_vec)
epsilon = epsilon_vec[args.epsilon_indx]

print('epsilon = ', epsilon)

mu_vec = np.linspace(-5, 5, 11)
assert args.mu_indx < (len(mu_vec) - 1)
mu = mu_vec[args.mu_indx]
print('mu = ', mu)

def e_step_bump(means, infos, mu_indx): 
    cdf1 = sp.stats.norm.cdf(mu_vec[mu_indx+1], loc = means, scale = 1 / np.sqrt(infos))
    cdf2 = sp.stats.norm.cdf(mu_vec[mu_indx], loc = means, scale = 1 / np.sqrt(infos))
    
    return (cdf1 - cdf2).sum()

e_log_phi = lambda means, infos : e_step_bump(means, infos, args.mu_indx) * epsilon


######################
# OPTIMIZE
######################
t0 = time.time() 

vb_opt_dict, vb_opt, ez_opt, out, optim_time = \
    s_optim_lib.optimize_structure(g_obs, 
                                   vb_params_dict, 
                                   vb_params_paragami,
                                   prior_params_dict,
                                   gh_loc, gh_weights, 
                                   e_log_phi = e_log_phi)

######################
# save optimization results
######################
outfile = os.path.join(args.out_folder, args.out_filename)

print('saving structure model to ', outfile)

print('Optim time (ignoring compilation time) {:.3f}secs'.format(optim_time))

# save final KL
final_kl = out.fun

# save paragami object
structure_model_lib.save_structure_fit(outfile, 
                                       vb_opt_dict,
                                       vb_params_paragami, 
                                       prior_params_dict,
                                       fit_meta_data['gh_deg'], 
                                       epsilon = epsilon,
                                       mu = mu,
                                       data_file = args.data_file, 
                                       final_kl = final_kl, 
                                       optim_time = optim_time)

print('Total optim time: {:.3f} secs'.format(time.time() - t0))


print('done. ')
