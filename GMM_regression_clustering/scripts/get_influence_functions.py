import jax

import jax.numpy as np
import jax.scipy as sp

from numpy.polynomial.hermite import hermgauss

# BNP regression mixture libraries
from bnpreg_runjingdev import genomics_data_utils
from bnpreg_runjingdev import regression_mixture_lib
from bnpreg_runjingdev import regression_posterior_quantities as reg_posterior_quantities

# bnp libraries
from bnpmodeling_runjingdev import influence_lib

from bnpmodeling_runjingdev.sensitivity_lib import \
        HyperparameterSensitivityLinearApproximation

import paragami

from copy import deepcopy

import time

import re
import os
import argparse
parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, default = 2344523)

# Set bnp_data_repo to be the location of a clone of the repo
# https://github.com/NelleV/genomic_time_series_bnp
parser.add_argument('--bnp_data_repo', type=str, 
                    default = '../../../genomic_time_series_bnp')

# folder where the fit was saved
parser.add_argument('--out_folder', type=str)

# name of the initial fit 
parser.add_argument('--fit_file', type=str)

# tolerance of CG solver
parser.add_argument('--cg_tol', type=float, default=1e-3)

args = parser.parse_args()

fit_file = os.path.join(args.out_folder, args.fit_file)

def validate_args():
    assert os.path.exists(args.out_folder), args.out_folder
    
    assert args.fit_file.endswith('.npz')
    assert os.path.isfile(fit_file), fit_file


validate_args()

outfile = re.sub('.npz', '_influence', fit_file)
print('derivative outfile: ', outfile)

########################
# load mice regression data
########################
genome_data, timepoints, regressors, beta, beta_infos, y_infos = \
    genomics_data_utils.load_data_and_run_regressions(args.bnp_data_repo)

n_genes = genome_data.shape[0]
reg_dim = regressors.shape[1]

n_timepoints = len(np.unique(timepoints))


##################
# Load initial fit
##################
print('loading fit from ', fit_file)
vb_opt_dict, vb_params_paragami, meta_data = \
        paragami.load_folded(fit_file)
vb_opt = vb_params_paragami.flatten(vb_opt_dict, free = True)    

# gauss-hermite parameters
gh_deg = int(meta_data['gh_deg'])
gh_loc, gh_weights = hermgauss(gh_deg)

gh_loc = np.array(gh_loc)
gh_weights = np.array(gh_weights)
    
# load prior parameters
prior_params_dict, prior_params_paragami = \
    regression_mixture_lib.get_default_prior_params()

# set initial alpha
dp_prior_alpha = meta_data['dp_prior_alpha']
prior_params_dict['dp_prior_alpha'] = dp_prior_alpha
print('alpha: ', prior_params_dict['dp_prior_alpha'])

###############
# Define objective and check KL
###############
# this also contains the hvp

def objective_fun(vb_free, epsilon): 
    # NOTE! epsilon doesn't actually enter 
    # into this function. 
    
    # since the initial fit is at epsilon = 0, 
    # we just return the actual KL
    
    # we will set the hyper-param objective function 
    # appropriately, later. 
    
    vb_params_dict = vb_params_paragami.fold(vb_free, free = True)
    
    return regression_mixture_lib.get_kl(genome_data, regressors,
                                         vb_params_dict,
                                         prior_params_dict,
                                         gh_loc,
                                         gh_weights).squeeze()


# check KL's match
kl = objective_fun(vb_opt, None)
diff = np.abs(kl - meta_data['final_kl'])
assert diff < 1e-8, diff

###############
# Define the linear sensitivity class
###############
vb_sens = HyperparameterSensitivityLinearApproximation(
                    objective_fun = objective_fun, 
                    opt_par_value = vb_opt, 
                    hyper_par_value0 = np.array([0.]), 
                    # will set appropriately later
                    hyper_par_objective_fun = lambda x, y : 0., 
                    cg_tol = args.cg_tol)

# class VBSens(object): 
#     def __init__(self): 
#         foo = 1 
        
#     def hessian_solver(self, x): 
#         return x

# vb_sens = VBSens()

###############
# Define the posterior quantities
###############
prng_key = jax.random.PRNGKey(args.seed)

def get_n_clusters_insample(vb_free):
    
    vb_params_dict = vb_params_paragami.fold(vb_free, free = True)
    
    return reg_posterior_quantities.get_e_num_clusters_from_vb_dict(genome_data,
                                                                    regressors,
                                                                    vb_params_dict,
                                                                    prior_params_dict,
                                                                    gh_loc, gh_weights,
                                                                    threshold = 0,
                                                                    prng_key = prng_key)

def get_n_clusters_pred(vb_free):
    
    vb_params_dict = vb_params_paragami.fold(vb_free, free = True)
    
    return reg_posterior_quantities.get_e_num_pred_clusters_from_vb_dict(vb_params_dict,
                                                               n_obs = genome_data.shape[0],
                                                               threshold = 0,
                                                               prng_key = prng_key)

###############
# Define influence operator
###############
influence_operator = influence_lib.InfluenceOperator(vb_opt, 
                                                     vb_params_paragami, 
                                                     vb_sens.hessian_solver,
                                                     prior_params_dict['dp_prior_alpha'],
                                                     stick_key = 'stick_params')


# function to get influence function 
logit_v_grid = np.linspace(-12, 12, 1000)

def get_influence(g): 
    print('computing gradient ...')
    t0 = time.time()
    get_grad_g = jax.jacobian(g, argnums = 0)
    grad_g = get_grad_g(vb_opt).block_until_ready()
    grad_g_time = time.time() - t0  
    print('Elapsed: {:.03f}sec'.format(grad_g_time))

    # get influence function
    print('inverting Hessian (twice) ...')
    t0 = time.time()

    # get influence function as defined
    influence_grid, grad_g_hess_inv = \
        influence_operator.get_influence(logit_v_grid, grad_g)


    # this is influence times the prior
    influence_grid_x_prior, _ = \
        influence_operator.get_influence(logit_v_grid, 
                                         grad_g, 
                                         normalize_by_prior = False)

    hess_inv_time = time.time() - t0
    print('Elapsed: {:.03f}sec'.format(hess_inv_time))

    return influence_grid, influence_grid_x_prior, grad_g_hess_inv

###############
# compute influence function and save
###############

vars_to_save = dict()

def get_influence_and_save(g, g_name): 
    
    influence_grid, influence_grid_x_prior, grad_g_hess_inv = \
        get_influence(g)
    
    vars_to_save[g_name + '_infl'] = influence_grid
    vars_to_save[g_name + '_infl_x_prior'] = influence_grid_x_prior
    vars_to_save[g_name + '_ghess'] = grad_g_hess_inv
    
    print('saving into: ', outfile)
    np.savez(outfile,
             vb_opt = vb_opt,
             dp_prior_alpha = dp_prior_alpha,
             seed = args.seed,
             logit_v_grid = logit_v_grid,
             kl= kl,
             **vars_to_save)

get_influence_and_save(get_n_clusters_insample, 
                       'num_clust')

get_influence_and_save(get_n_clusters_pred, 
                       'num_clust_pred')


print('done. ')
