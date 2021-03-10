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

# delta 
parser.add_argument('--delta', type=float, default = 1.0)

# which perturbation
parser.add_argument('--perturbation', type=str, default = 'sigmoidal')

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
# set epsilon 
epsilon_vec = np.linspace(0, 1, 10)[1:]**2 
assert args.epsilon_indx < len(epsilon_vec)

epsilon = epsilon_vec[args.epsilon_indx]
print('epsilon = ', epsilon)
print('epsilon_indx = ', args.epsilon_indx)

f_obj_all = log_phi_lib.LogPhiPerturbations(vb_params_paragami, 
                                                 prior_params_dict['dp_prior_alpha'],
                                                 gh_loc, 
                                                 gh_weights,
                                                 delta = args.delta, 
                                                 stick_key = 'ind_admix_params')

f_obj = getattr(f_obj_all, 'f_obj_' + args.perturbation)
e_log_phi = lambda means, infos : f_obj.e_log_phi_epsilon(means, infos, epsilon)

# # warm start w linear response 
# vb_opt = vb_params_paragami.flatten(vb_params_dict, 
#                                     free = True)
# vb_opt_pert = vb_opt + lr_data['dinput_dfun_' + args.perturbation] * epsilon
# vb_params_dict = vb_params_paragami.fold(vb_opt_pert, 
#                                          free = True)

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
                                       perturbation = args.perturbation)

print('Total optim time: {:.3f} secs'.format(time.time() - t0))


print('done. ')
