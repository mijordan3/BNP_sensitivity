import jax

import jax.numpy as np
import jax.scipy as sp

import numpy as onp
from numpy.polynomial.hermite import hermgauss

from vb_lib import structure_model_lib, data_utils, cavi_lib
from vb_lib.structure_optimization_lib import define_structure_objective
from bnpmodeling_runjingdev.optimization_lib import run_lbfgs

import paragami

from copy import deepcopy

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

# whether to use a warm start
parser.add_argument('--warm_start', type=distutils.util.strtobool, default='False')
parser.add_argument('--init_fit', type=str)

# model parameters
parser.add_argument('--alpha', type=float, default = 4.0)
parser.add_argument('--k_approx', type = int, default = 15)
parser.add_argument('--use_logitnormal_sticks', type=distutils.util.strtobool,
                        default='True')


args = parser.parse_args()

def validate_args():
    assert os.path.exists(args.out_folder), args.out_folder

    if args.warm_start:
        assert os.path.isfile(args.init_fit), args.init_fit

    assert os.path.isfile(args.data_file), args.data_file

validate_args()

onp.random.seed(args.seed)

######################
# Load Data
######################
print('loading data from ', args.data_file)
data = np.load(args.data_file)
g_obs = np.array(data['g_obs'])

n_obs = g_obs.shape[0]
n_loci = g_obs.shape[1]

print('g_obs.shape', g_obs.shape)

######################
# GET PRIOR
######################
prior_params_dict, prior_params_paragami = \
    structure_model_lib.get_default_prior_params()

prior_params_dict['dp_prior_alpha'] = np.array([args.alpha])

print('prior params: ')
print(prior_params_dict)

######################
# GET VB PARAMS
######################
k_approx = args.k_approx
gh_deg = 8
gh_loc, gh_weights = hermgauss(gh_deg)

vb_params_dict, vb_params_paragami = \
    structure_model_lib.get_vb_params_paragami_object(n_obs, n_loci, k_approx,
                                    args.use_logitnormal_sticks)

print('vb params: ')
print(vb_params_paragami)

######################
# get init
######################
init_optim_time = time.time()
if args.warm_start:
    print('warm start from ', args.init_fit)
    vb_params_dict, _, _ = \
        paragami.load_folded(args.init_fit)
# else: 
#     vb_params_dict = \
#         structure_model_lib.set_init_vb_params(g_obs, k_approx, vb_params_dict,
#                                                 seed = args.seed)


######################
# OPTIMIZE
######################
# get optimization objective 
optim_objective, init_vb_free = \
    define_structure_objective(g_obs, vb_params_dict,
                        vb_params_paragami,
                        prior_params_dict,
                        gh_loc = gh_loc,
                        gh_weights = gh_weights)

out = run_lbfgs(optim_objective, init_vb_free)

vb_opt = out.x
vb_opt_dict = vb_params_paragami.fold(vb_opt, free = True)

######################
# save optimizaiton results
######################
outfile = os.path.join(args.out_folder, args.out_filename)
print('saving structure model to ', outfile)

optim_time = time.time() - init_optim_time


# save final KL
final_kl = structure_model_lib.get_kl(g_obs, vb_opt_dict,
                            prior_params_dict,
                            gh_loc = gh_loc,
                            gh_weights = gh_weights)

# save paragami object
paragami.save_folded(outfile,
                     vb_opt_dict,
                     vb_params_paragami,
                     data_file = args.data_file,
                     dp_prior_alpha = prior_params_dict['dp_prior_alpha'],
                     allele_prior_alpha = prior_params_dict['allele_prior_alpha'],
                     allele_prior_beta = prior_params_dict['allele_prior_beta'],
                     gh_deg = gh_deg,
                     use_logitnormal_sticks = args.use_logitnormal_sticks,
                     final_kl = final_kl,
                     optim_time = optim_time)

print('Total optimization time: {:03f} secs'.format(optim_time))


print('done. ')
