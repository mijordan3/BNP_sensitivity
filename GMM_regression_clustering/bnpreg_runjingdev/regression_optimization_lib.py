import jax 
import jax.numpy as np

import numpy as onp

import time

from bnpmodeling_runjingdev.sensitivity_lib import get_jac_hvp_fun
from bnpmodeling_runjingdev.bnp_optimization_lib import optimize_kl, \
    update_stick_beta_params, convert_beta_sticks_to_logitnormal

from bnpgmm_runjingdev.gmm_optimization_lib import init_centroids_w_kmeans

from bnpreg_runjingdev import regression_mixture_lib 
from bnpreg_runjingdev.regression_posterior_quantities import get_optimal_local_params_from_vb_dict
from bnpreg_runjingdev.genomics_utils import regression_lib

from sklearn.cluster import KMeans

from copy import deepcopy

#####################
# functions to initialize 
#####################
def set_params_w_kmeans(y, regressors,
                        vb_params_dict, 
                        vb_params_paragami, 
                        prior_params_dict,
                        gh_loc, gh_weights, 
                        seed = 4353): 
    
    onp.random.seed(seed)
    
    # run initial regressions
    print('running initial regressions ...')
    beta, _, y_infos = \
        regression_lib.run_regressions(y - np.mean(y, axis = 1, keepdims = True),
                                       regressors)

    
    k_approx = vb_params_dict['centroids'].shape[0]
    
    print('running k-means ... ')
    vb_params_dict['centroids'], _ = \
        init_centroids_w_kmeans(beta, k_approx, 
                                n_kmeans_init = 10, 
                                seed = seed)        
    return vb_params_dict


################
# Function to optimize
################

def optimize_regression_mixture(y, regressors, 
                                vb_params_dict, 
                                vb_params_paragami,
                                prior_params_dict, 
                                gh_loc, gh_weights, 
                                e_log_phi = None, 
                                run_lbfgs = True,
                                run_newton = True): 
        
    ###################
    # Define loss
    ###################
    def get_kl_loss(vb_params_free): 
        
        vb_params_dict = vb_params_paragami.fold(vb_params_free, free = True)
    
        return regression_mixture_lib.get_kl(y, regressors,
                                             vb_params_dict,
                                             prior_params_dict,
                                             gh_loc,
                                             gh_weights, 
                                             e_log_phi = e_log_phi)
    
    ###################
    # optimize
    ###################
    vb_opt_dict, vb_opt, out, optim_time = optimize_kl(get_kl_loss,
                                                       vb_params_dict, 
                                                       vb_params_paragami, 
                                                       run_lbfgs = run_lbfgs,
                                                       run_newton = run_newton)
                
    # compute optimal ez
    ez_opt = get_optimal_local_params_from_vb_dict(y, regressors,
                                                   vb_opt_dict,
                                                   prior_params_dict, 
                                                   gh_loc, gh_weights)[0]
        
    return vb_opt_dict, vb_opt, ez_opt, out, optim_time
