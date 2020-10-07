import jax

import jax.numpy as np
import jax.scipy as sp

import numpy as onp
from numpy.polynomial.hermite import hermgauss

from vb_lib import structure_model_lib, data_utils, cavi_lib
from vb_lib.structure_optimization_lib import optimize_structure

import paragami

from copy import deepcopy

import argparse
import distutils.util

import os

import time

parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, default=42)

# whether to load data, and if so, from which file
parser.add_argument('--load_data', type=distutils.util.strtobool, default = False)
parser.add_argument('--data_file', type=str)

# if we do not load data, generate data and save
parser.add_argument('--n_obs', type=int, default=100)
parser.add_argument('--n_loci', type=int, default=2000)
parser.add_argument('--n_pop', type=int, default=4)

# where to save the structure fit
parser.add_argument('--outfolder', default='../fits/')
parser.add_argument('--out_filename', default='structure_fit', type=str)

# whether to use a warm start
parser.add_argument('--warm_start', type=distutils.util.strtobool, default='False')
parser.add_argument('--init_fit', type=str)

# model parameters
parser.add_argument('--alpha', type=float, default = 4.0)
parser.add_argument('--k_approx', type = int, default = 10)
parser.add_argument('--use_logitnormal_sticks', type=distutils.util.strtobool,
                        default='True')


args = parser.parse_args()

def validate_args():
    assert os.path.exists(args.outfolder), args.outfolder

    if args.warm_start:
        assert os.path.isfile(args.init_fit), args.init_fit

    if args.load_data:
        assert os.path.isfile(args.data_file), args.data_file

validate_args()

onp.random.seed(args.seed)

######################
# DRAW DATA
######################
if args.load_data:
    print('loading data from ', args.data_file)
    data = np.load(args.data_file)

    g_obs = np.array(data['g_obs'])

else:
    print('simulating data')
    g_obs, true_pop_allele_freq, true_ind_admix_propn = \
        data_utils.draw_data(args.n_obs, args.n_loci, args.n_pop)

    print('saving simulated data into ', args.data_file)
    np.savez(args.data_file,
            g_obs = g_obs,
            true_pop_allele_freq = true_pop_allele_freq,
            true_ind_admix_propn = true_ind_admix_propn)

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
if not args.warm_start:
    vb_params_dict = \
        structure_model_lib.set_init_vb_params(g_obs, k_approx, vb_params_dict,
                                                seed = args.seed)
else:
    print('warm start from ', args.init_fit)
    vb_params_dict, _, _ = \
        paragami.load_folded(args.init_fit)

######################
# OPTIMIZE
######################
# vb_opt_dict, vb_opt, _, _ = \
#     cavi_lib.run_cavi(g_obs, vb_params_dict,
#                         vb_params_paragami,
#                         prior_params_dict,
#                         gh_loc = gh_loc,
#                         gh_weights = gh_weights,
#                         max_iter = 2000,
#                         x_tol = 1e-4,
#                         print_every = 20)

vb_opt_dict, vb_out, _ = \
    optimize_structure(g_obs, vb_params_dict,
                        vb_params_paragami,
                        prior_params_dict,
                        gh_loc = gh_loc,
                        gh_weights = gh_weights)
                        
######################
# save results
######################
outfile = os.path.join(args.outfolder, args.out_filename + '.npz')
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
