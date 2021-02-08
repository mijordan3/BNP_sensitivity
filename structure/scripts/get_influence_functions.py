import jax

import jax.numpy as np
import jax.scipy as sp

import scipy as osp

import paragami

from bnpmodeling_runjingdev import influence_lib
from bnpmodeling_runjingdev.sensitivity_lib import \
        HyperparameterSensitivityLinearApproximation

from structure_vb_lib import structure_model_lib, posterior_quantities_lib
from structure_vb_lib.preconditioner_lib import get_mfvb_cov_matmul
import structure_vb_lib.structure_optimization_lib as s_optim_lib

import time

import os 
import re

import argparse
parser = argparse.ArgumentParser()

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
print('loading data from ', args.data_file)
data = np.load(args.data_file)
g_obs = np.array(data['g_obs'])

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
# this also contains the hvp
stru_objective = s_optim_lib.StructureObjective(g_obs, 
                                                vb_params_paragami,
                                                prior_params_dict, 
                                                gh_loc, gh_weights, 
                                                jit_functions = False)

# check KL's match
kl = stru_objective.f(vb_opt)
diff = np.abs(kl - meta_data['final_kl'])
assert diff < 1e-8, diff

###############
# Define preconditioner
###############
cg_precond = lambda v : get_mfvb_cov_matmul(v, vb_opt_dict,
                                            vb_params_paragami,
                                            return_sqrt = False, 
                                            return_info = True)

###############
# set up linear response object
###############
# we just need to compiled hessian solver method 
vb_sens = HyperparameterSensitivityLinearApproximation(
                    objective_fun = stru_objective.f, 
                    opt_par_value = vb_opt, 
                    hyper_par_value0 = np.array([0.]), 
                    obj_fun_hvp = stru_objective.hvp, 
                    # just a null perturbation
                    # don't actually need this here
                    hyper_par_objective_fun = lambda x, y : 0., 
                    cg_precond = cg_precond, 
                    cg_tol = args.cg_tol,
                    cg_maxiter = None)

print('cg tol: ')
print(vb_sens.cg_tol)
print(vb_sens.cg_maxiter)
hessian_solver = vb_sens.hessian_solver

###############
# compute grad(g)H^{-1} for various posterior statistics g
###############
vars_to_save = dict()

logit_v_grid = np.linspace(-10, 10, 200)

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
    influence_operator = influence_lib.InfluenceOperator(vb_opt, 
                               vb_params_paragami, 
                               hessian_solver,
                               prior_params_dict['dp_prior_alpha'], 
                               stick_key = 'ind_admix_params')
    
    influence_grid, grad_g_hess_inv = influence_operator.get_influence(logit_v_grid, 
                                                                       grad_g)
    # to not mess up timing results
    _ = grad_g_hess_inv.block_until_ready()
    hess_inv_time = time.time() - t0
    print('Elapsed: {:.03f}sec'.format(hess_inv_time))
          
    return influence_grid, grad_g_hess_inv, grad_g_time, hess_inv_time

def get_influence_and_save(g, post_stat_name): 
    
    influence_grid, grad_g_hess_inv, grad_g_time, hess_inv_time = \
        get_influence(g)
    
    vars_to_save[post_stat_name + '_inf_grid'] = influence_grid
    vars_to_save[post_stat_name + '_ghess'] = grad_g_hess_inv
    vars_to_save[post_stat_name + '_grad_g_time'] = grad_g_time
    vars_to_save[post_stat_name + '_hess_inv_time'] = hess_inv_time

    print('saving into: ', outfile)
    np.savez(outfile,
             vb_opt = vb_opt,
             alpha0 = prior_params_dict['dp_prior_alpha'],
             kl= kl,
             cg_tol = vb_sens.cg_tol,
             cg_maxiter = vb_sens.cg_maxiter,
             **vars_to_save)



# get influence on cluster weights 
def get_e_num_ind(vb_free, k): 
    vb_params_dict = vb_params_paragami.fold(vb_free, free = True)
    return posterior_quantities_lib.get_e_num_ind_per_cluster(vb_params_dict, 
                                                              gh_loc, gh_weights).sum(0)

for k in range(8): 
    print('###############')
    print('Computing influence for cluster weight k = {}'.format(k))
    print('###############')
    get_influence_and_save(lambda x : get_e_num_ind(x, k), 
                           'e_num_ind{}'.format(k))

# influence on expected number of predicted clusters
print('###############')
print('Computing influence for expected number of predicted clusters')
print('###############')
          
def get_e_num_pred_clusters_from_vb_free(vb_free): 
    vb_params_dict = vb_params_paragami.fold(vb_free, free = True)
    
    return posterior_quantities_lib.\
            get_e_num_pred_clusters(vb_params_dict,
                                    gh_loc,
                                    gh_weights, 
                                    n_samples = 1000,
                                    threshold = 0, 
                                    prng_key = jax.random.PRNGKey(3453), 
                                    return_samples = False)
          
          
          
          
          
get_influence_and_save(get_e_num_pred_clusters_from_vb_free,
                       'e_num_pred_clusters')
          
print('done. ')
          
          
          
          
          
          