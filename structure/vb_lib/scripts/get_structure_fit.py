import autograd

import autograd.numpy as np
import autograd.scipy as sp
from numpy.polynomial.hermite import hermgauss

import sys
sys.path.insert(0, '../')

import structure_model_lib
import structure_optimization_lib as str_opt_lib

import paragami
import vittles

from copy import deepcopy

import argparse
import distutils.util

import os

import time

parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--data_file', type=str)


parser.add_argument('--outfolder', default='../fits/')
parser.add_argument('--out_filename', default='structure_fit', type=str)

parser.add_argument('--warm_start', type=distutils.util.strtobool, default='False')
parser.add_argument('--init_fit', type=str)

parser.add_argument('--alpha', type=float, default = 4.0)
parser.add_argument('--k_approx', type = int, default = 12)
parser.add_argument('--use_logitnormal_sticks', type=distutils.util.strtobool,
                        default='False')


args = parser.parse_args()

def validate_args():
    assert os.path.exists(args.outfolder)

    if args.warm_start:
        assert os.path.isfile(args.init_fit)

validate_args()

np.random.seed(args.seed)

######################
# DRAW DATA
######################
# observations
data = np.load('../data/simulated_structure_data_nobs50_nloci40_npop4.npz')

g_obs = data['g_obs']

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
                                                args.use_logitnormal_sticks)
else:
    print('warm start from ', args.init_file)
    vb_params_dict, _, _ = \
        paragami.load_folded(args.init_file)

######################
# OPTIMIZE
######################
vb_opt_free_params = \
    str_opt_lib.optimize_structure(g_obs, vb_params_dict, vb_params_paragami,
                                prior_params_dict,
                                gh_loc, gh_weights,
                                use_logitnormal_sticks = args.use_logitnormal_sticks,
                                run_cavi = True,
                                cavi_max_iter = 2000,
                                cavi_tol = 1e-2,
                                netwon_max_iter = 20,
                                max_precondition_iter = 25,
                                gtol=1e-8, ftol=1e-8, xtol=1e-8,
                                approximate_hessian = True)

vb_opt_dict = vb_params_paragami.fold(vb_opt_free_params, free=True)

structure_model_lib.assert_optimizer(g_obs, vb_opt_dict, vb_params_paragami,
                        prior_params_dict, gh_loc, gh_weights,
                        args.use_logitnormal_sticks)

######################
# save results
######################
outfile = os.path.join(args.outfolder, args.out_filename)
print('saving structure model to ', outfile)
paragami.save_folded(outfile,
                     vb_opt_dict,
                     vb_params_paragami,
                     alpha = prior_params_dict['dp_prior_alpha'],
                     gh_deg = gh_deg,
                     use_logitnormal_sticks = args.use_logitnormal_sticks)

print('Total optimization time: {:03f} secs'.format(time.time() - init_optim_time))

#######################
# Get Hessian and save
#######################
print('getting sensitivity object: ')
t0 = time.time()
get_kl_from_vb_free_prior_free = \
    paragami.FlattenFunctionInput(original_fun=structure_model_lib.get_kl,
                                    patterns = [vb_params_paragami, prior_params_paragami],
                                    free = True,
                                    argnums = [1, 2])
objective_fun = lambda x, y: \
    get_kl_from_vb_free_prior_free(g_obs, x, y, args.use_logitnormal_sticks,
                                    gh_loc, gh_weights)
vb_sens = \
    vittles.HyperparameterSensitivityLinearApproximation(
        objective_fun = objective_fun,
        opt_par_value = vb_opt_free_params,
        hyper_par_value = prior_params_paragami.flatten(prior_params_dict, free=True),
        validate_optimum=False,
        hessian_at_opt=None,
        cross_hess_at_opt=None,
        factorize_hessian=True,
        hyper_par_objective_fun=None,
        grad_tol=1e-8)

print('Hessian time: {:03f}'.format(time.time() - t0))

np.savez(outfile + '_sens_obj',
        hessian = vb_sens._hess0,
        cross_hess = vb_sens._cross_hess)

# print('checking sensitivity derivative ... ')
# which_prior = np.array([1., 0., 0.])
# hessian_dir = str_opt_lib.check_hessian(vb_sens, which_prior)
# print('L inf norm of sensitivity derivative: {}'.format(
#             np.max(np.abs(hessian_dir))))

print('done. ')
