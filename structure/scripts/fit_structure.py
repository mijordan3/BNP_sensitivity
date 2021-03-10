import jax

import jax.numpy as np
import jax.scipy as sp

import numpy as onp
from numpy.polynomial.hermite import hermgauss

import structure_vb_lib.structure_model_lib as structure_model_lib
import structure_vb_lib.structure_optimization_lib as s_optim_lib
from structure_vb_lib import data_utils

import paragami

import argparse
import distutils.util

import os

import time

parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, default=42)

# the genome dataset
parser.add_argument('--data_file', type=str)

# where to save the structure fit
parser.add_argument('--out_folder', default='../fits/')
parser.add_argument('--out_filename', default='structure_fit', type=str)

# whether to use a warm start
parser.add_argument('--warm_start', type=distutils.util.strtobool, default='False')
parser.add_argument('--init_fit', type=str)

# model parameters
parser.add_argument('--alpha', type=float, default = 3.0)
parser.add_argument('--k_approx', type = int, default = 20)

args = parser.parse_args()

def validate_args():
    assert os.path.exists(args.out_folder), args.out_folder

    if args.warm_start:
        assert os.path.isfile(args.init_fit), args.init_fit
    
    assert os.path.isfile(args.data_file)
validate_args()

onp.random.seed(args.seed)

######################
# Load Data
######################
g_obs = data_utils.load_thrush_data(args.data_file)[0]

print(g_obs.shape)

n_obs = g_obs.shape[0]
n_loci = g_obs.shape[1]
n_allele = g_obs.shape[-1]

######################
# GET PRIOR
######################
prior_params_dict, prior_params_paragami = \
    structure_model_lib.get_default_prior_params(n_allele)

prior_params_dict['dp_prior_alpha'] = args.alpha

print('prior params: ')
print(prior_params_dict)

######################
# GET VB PARAMS 
######################
k_approx = args.k_approx
gh_deg = 8
gh_loc, gh_weights = hermgauss(gh_deg)

vb_params_dict, vb_params_paragami = \
    structure_model_lib.get_vb_params_paragami_object(n_obs = n_obs,
                                                      n_loci = n_loci,
                                                      n_allele = n_allele, 
                                                      k_approx = k_approx)
    


print(vb_params_paragami)

######################
# OPTIMIZE
######################
init_optim_time = time.time()

# optimize with preconditioner 
vb_opt_dict, vb_opt, ez_opt, out, optim_time = \
    s_optim_lib.optimize_structure(g_obs, 
                                   vb_params_dict, 
                                   vb_params_paragami,
                                   prior_params_dict,
                                   gh_loc, gh_weights)

######################
# save optimizaiton results
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
                                       gh_deg, 
                                       data_file = args.data_file, 
                                       final_kl = final_kl, 
                                       optim_time = optim_time)

print('Total optim time: {:.3f} secs'.format(time.time() - init_optim_time))


print('done. ')
