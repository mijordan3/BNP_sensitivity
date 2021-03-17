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
from bnpmodeling_runjingdev import influence_lib

# BNP gmm libraries
# BNP regression mixture libraries
import structure_vb_lib.structure_model_lib as structure_model_lib
import structure_vb_lib.structure_optimization_lib as s_optim_lib
from structure_vb_lib import data_utils


parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, default=42)

parser.add_argument('--data_file', type=str)

# where to save the structure fit
parser.add_argument('--out_folder', default='../fits/')
parser.add_argument('--out_filename', default='structure_fit', type=str)

# the initial fit
parser.add_argument('--init_fit', type=str)

# file with influence function
parser.add_argument('--influence_file', type=str)

# which epsilon 
parser.add_argument('--epsilon_indx', type=int, default = 0)

# delta 
parser.add_argument('--delta', type=float, default = 1.0)

# which perturbation
# get worst case for which perturbation
parser.add_argument('--g_name', type=str, default = 'number of clsuters')

args = parser.parse_args()

assert os.path.exists(args.out_folder), args.out_folder
assert os.path.isfile(args.init_fit), args.init_fit
assert os.path.isfile(args.influence_file), args.influence_file

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

########################
# load influence function  
########################
print("loading influence file from: ")
print(args.influence_file)
influence_results = np.load(args.influence_file)

# check that things in the influence file match things in my fit file
assert np.all(influence_results['vb_opt'] == vb_params_paragami.flatten(vb_params_dict, free = True))
assert np.abs(influence_results['kl'] - fit_meta_data['final_kl']) < 1e-10

# the influence function for the posterior statistic "g_name"
influence_grid = influence_results[args.g_name + '_infl']
logit_v_grid = influence_results['logit_v_grid']

########################
# define worst-case perturbation
########################
worst_case = influence_lib.WorstCasePerturbation(influence_fun = None, 
                                                 logit_v_grid = logit_v_grid, 
                                                 delta = args.delta,
                                                 cached_influence_grid = influence_grid)


########################
# Functional perturbation 
########################
# set epsilon
epsilon_vec = np.linspace(0, 1, 10)[1:]**2 
epsilon = epsilon_vec[args.epsilon_indx]
print('epsilon = ', epsilon)
print('epsilon_indx = ', args.epsilon_indx)

e_log_phi = lambda means, infos : worst_case.get_e_log_linf_perturbation(means, infos) * epsilon * args.delta

########################
# Optimize
########################
t0 = time.time() 

vb_opt_dict, vb_opt, ez_opt, out, optim_time = \
    s_optim_lib.optimize_structure(g_obs, 
                                   vb_params_dict, 
                                   vb_params_paragami,
                                   prior_params_dict,
                                   gh_loc, gh_weights, 
                                   e_log_phi = e_log_phi)

#####################
# Save results
#####################
outfile = os.path.join(args.out_folder, args.out_filename)

print('saving structure model to ', outfile)

print('Optim time (ignoring compilation time) {:.3f}secs'.format(optim_time))

# save paragami object
structure_model_lib.save_structure_fit(outfile, 
                                       vb_opt_dict,
                                       vb_params_paragami, 
                                       prior_params_dict,
                                       fit_meta_data['gh_deg'], 
                                       epsilon = epsilon,
                                       delta = args.delta,
                                       data_file = args.data_file, 
                                       final_kl = out.fun, 
                                       optim_time = optim_time, 
                                       g_name = args.g_name)

print('Total optim time: {:.3f} secs'.format(time.time() - t0))


print('done. ')
