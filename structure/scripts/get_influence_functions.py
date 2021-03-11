import jax

import jax.numpy as np
import jax.scipy as sp

import scipy as osp

import paragami

from bnpmodeling_runjingdev import influence_lib, cluster_quantities_lib
from bnpmodeling_runjingdev.sensitivity_lib import \
        HyperparameterSensitivityLinearApproximation, get_cross_hess

import bnpmodeling_runjingdev.functional_sensitivity_lib as func_sens_lib

from structure_vb_lib import structure_model_lib, posterior_quantities_lib, data_utils

import time

import os 
import re

import argparse
parser = argparse.ArgumentParser()

# seed
parser.add_argument('--seed', type=int, default = 2344523)

# the genome dataset
parser.add_argument('--data_file', type=str)

# folder where the structure fit was saved
parser.add_argument('--out_folder', type=str)

# name of the structure fit 
parser.add_argument('--fit_file', type=str)

# tolerance of CG solver
parser.add_argument('--cg_tol', type=float, default=1e-2)

args = parser.parse_args()

fit_file = os.path.join(args.out_folder, args.fit_file)

def validate_args():
    assert os.path.exists(args.out_folder), args.out_folder
    assert os.path.isfile(args.data_file), args.data_file
    
    assert args.fit_file.endswith('.npz')
    assert os.path.isfile(fit_file), fit_file


validate_args()

outfile = re.sub('.npz', '_infl_funcs', fit_file)
print('derivative outfile: ', outfile)

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
print('loading fit from ', fit_file)
vb_opt_dict, vb_params_paragami, \
    prior_params_dict, prior_params_paragami, \
        gh_loc, gh_weights, meta_data = \
            structure_model_lib.load_structure_fit(fit_file)

vb_opt = vb_params_paragami.flatten(vb_opt_dict, free = True)

###############
# Define objective and check KL
###############
def objective_fun(vb_params_free, epsilon): 
    
    # this actually does not depend on epsilon! 
    # we will set the perturbation later
    
    vb_params_dict = vb_params_paragami.fold(vb_params_free, free = True)
    
    return structure_model_lib.get_kl(g_obs, 
                                      vb_params_dict,
                                      prior_params_dict,
                                      gh_loc, 
                                      gh_weights).squeeze()

# check KL's match
kl = objective_fun(vb_opt, 0.)
diff = np.abs(kl - meta_data['final_kl'])
assert diff < 1e-8, diff


###############
# set up linear response object
###############
# we just need to compiled hessian solver method  in this class
vb_sens = HyperparameterSensitivityLinearApproximation(
                    objective_fun = objective_fun, 
                    opt_par_value = vb_opt, 
                    hyper_par_value0 = np.array([0.]),
                    # don't really need
                    hyper_par_objective_fun = lambda x, y: 0., 
                    cg_tol = args.cg_tol,
                    cg_maxiter = None)

print('cg tol: ')
print(vb_sens.cg_tol)
print(vb_sens.cg_maxiter)

# class VBSens(object):
#     def __init__(self): 
#         self.cg_tol = 1e-2
#         self.cg_maxiter = 100
#         self.hessian_solver = lambda x : np.zeros(len(x))

# vb_sens = VBSens()

###############
# Define the posterior quantities
###############
# get influence on cluster weights 
# the below function instantiates the whole matrix of individual admixtures
# before summing and subsetting
# throws a memory error on hgdp data ... 
# def get_e_num_ind(vb_free, k): 
#     vb_params_dict = vb_params_paragami.fold(vb_free, free = True)
#     return posterior_quantities_lib.get_e_num_ind_per_cluster(vb_params_dict, 
#                                                               gh_loc, gh_weights)[k]


def get_e_num_ind(vb_free, k = 0): 
    vb_params_dict = vb_params_paragami.fold(vb_free, free = True)
    
    stick_means = vb_params_dict['ind_admix_params']['stick_means']
    stick_infos = vb_params_dict['ind_admix_params']['stick_infos']
    
    e_ind_admix = cluster_quantities_lib.get_e_cluster_probabilities(stick_means, 
                                                                     stick_infos, 
                                                                     gh_loc, 
                                                                     gh_weights)
    
    # select a 'k' before summing ... 
    # this is probably more efficient
    return e_ind_admix[:, k].sum()


prng_key = jax.random.PRNGKey(args.seed)

def get_e_num_clusters(vb_free): 
    vb_params_dict = vb_params_paragami.fold(vb_free, free = True)
    
    return posterior_quantities_lib.get_e_num_clusters(g_obs,
                                                       vb_params_dict,
                                                       gh_loc, gh_weights,
                                                       threshold = 0,
                                                       n_samples = 1000, 
                                                       prng_key = prng_key)

def get_e_num_clusters_pred(vb_free): 
    vb_params_dict = vb_params_paragami.fold(vb_free, free = True)
    
    return posterior_quantities_lib.\
            get_e_num_pred_clusters(vb_params_dict,
                                    gh_loc,
                                    gh_weights, 
                                    n_samples = 1000,
                                    threshold = 0, 
                                    prng_key = prng_key,
                                    return_samples = False)


###############
# compute grad(g)H^{-1} for various posterior statistics g
###############
vars_to_save = dict()

# class to get influence functions
influence_operator = influence_lib.InfluenceOperator(vb_opt, 
                           vb_params_paragami, 
                           vb_sens.hessian_solver,
                           prior_params_dict['dp_prior_alpha'], 
                           stick_key = 'ind_admix_params')


logit_v_grid = np.linspace(-6, 6, 100)

def get_influence(g): 
    print('computing gradient ...')
    t0 = time.time()
    get_grad_g = jax.jacobian(g, argnums = 0)
    grad_g = get_grad_g(vb_opt).block_until_ready()
    grad_g_time = time.time() - t0  
    print('Elapsed: {:.03f}sec'.format(grad_g_time))
        
    # get influence function
    print('inverting Hessian ...')
    t0 = time.time()
    
    influence_grid, grad_g_hess_inv = influence_operator.get_influence(logit_v_grid, 
                                                                       grad_g)
    
    influence_grid_x_prior, _ = influence_operator.get_influence(logit_v_grid, 
                                                                 grad_g, 
                                                                 normalize_by_prior = False)
    
    hess_inv_time = time.time() - t0
    print('Elapsed: {:.03f}sec'.format(hess_inv_time))
          
    return influence_grid, influence_grid_x_prior, grad_g_hess_inv

def get_worst_cross_hess(influence_grid): 
    worst_case = influence_lib.WorstCasePerturbation(influence_fun = None, 
                                                     logit_v_grid = logit_v_grid, 
                                                     delta = 1.0,
                                                     cached_influence_grid = influence_grid)
    
    f_obj = func_sens_lib.FunctionalPerturbationObjective(worst_case.log_phi, 
                                                          vb_params_paragami, 
                                                          e_log_phi = lambda x,y : worst_case.get_e_log_linf_perturbation(x,y), 
                                                          gh_loc = gh_loc, 
                                                          gh_weights = gh_weights, 
                                                          stick_key = 'ind_admix_params')
    
    cross_hess = get_cross_hess(f_obj.hyper_par_objective_fun)(vb_opt, 0.)
    
    return cross_hess

def get_influence_and_save(g, post_stat_name): 
    
    influence_grid, influence_grid_x_prior, grad_g_hess_inv = \
        get_influence(g)
    
    vars_to_save[post_stat_name + '_infl'] = influence_grid
    vars_to_save[post_stat_name + '_infl_x_prior'] = influence_grid_x_prior
    vars_to_save[post_stat_name + '_ghess'] = grad_g_hess_inv
    
    print('computing cross-hessian ...')
    vars_to_save[post_stat_name + '_wc_cross_hess'] = get_worst_cross_hess(influence_grid)

    print('saving into: ', outfile)
    
    np.savez(outfile,
             vb_opt = vb_opt,
             dp_prior_alpha = prior_params_dict['dp_prior_alpha'],
             kl = kl,
             seed = args.seed,
             cg_tol = vb_sens.cg_tol,
             cg_maxiter = vb_sens.cg_maxiter,
             logit_v_grid = logit_v_grid,
             **vars_to_save)




for k in range(4): 
    print('###############')
    print('Computing influence for cluster weight k = {}'.format(k))
    print('###############')
    get_influence_and_save(lambda x : get_e_num_ind(x, k), 
                           'e_num_ind{}'.format(k))

print('###############')
print('Computing influence for expected number of insample clusters')
print('###############')
          
get_influence_and_save(get_e_num_clusters,
                       'num_clust')

print('###############')
print('Computing influence for expected number of predicted clusters')
print('###############')
          
get_influence_and_save(get_e_num_clusters_pred,
                       'num_clust_pred')
          
print('done. ')
          
          
          
          
          
          
