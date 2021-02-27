import jax 
import jax.numpy as np

import numpy as onp

import time

from bnpmodeling_runjingdev.sensitivity_lib import get_jac_hvp_fun
from bnpmodeling_runjingdev.bnp_optimization_lib import \
    update_stick_beta_params, convert_beta_sticks_to_logitnormal, optimize_kl

from bnpreg_runjingdev import regression_mixture_lib 
from bnpreg_runjingdev.regression_posterior_quantities import get_optimal_z_from_vb_dict

from copy import deepcopy
from sklearn.cluster import KMeans

################
# Functions to initialize
################
def init_centroids_w_kmeans(gamma, k_approx, n_kmeans_init = 10): 
    
    n_obs = np.shape(gamma)[0]
    dim = np.shape(gamma)[1]

    # K means init.
    for i in range(n_kmeans_init):
        km = KMeans(n_clusters = k_approx).fit(gamma)
        enertia = km.inertia_
        if (i == 0):
            enertia_best = enertia
            km_best = deepcopy(km)
        elif (enertia < enertia_best):
            enertia_best = enertia
            km_best = deepcopy(km)
    
    init_centroids = np.array(km_best.cluster_centers_)
    
    return init_centroids

def set_params_w_kmeans(gamma, gamma_info, vb_params_dict, vb_params_paragami, 
                        dp_prior_alpha, gh_loc, gh_weights): 
    
    k_approx = vb_params_dict['centroids'].shape[0]
    
    init_centroids = init_centroids_w_kmeans(gamma, k_approx, n_kmeans_init = 10)
    vb_params_dict['centroids'] = init_centroids
    
    
    stick_shape = vb_params_dict['stick_params']['stick_means'].shape
    vb_params_dict['stick_params']['stick_means'] = np.ones(stick_shape)
    vb_params_dict['stick_params']['stick_infos'] = np.ones(stick_shape)
    
    return vb_params_dict


################
# Functions to optimize
################

def optimize_regression_mixture(gamma, gamma_info, 
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
    
        return regression_mixture_lib.get_kl(gamma, 
                                             gamma_info,
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
    ez_opt = get_optimal_z_from_vb_dict(gamma, gamma_info,
                                        vb_opt_dict,
                                        gh_loc, gh_weights)
        
    return vb_opt_dict, vb_opt, ez_opt, out, optim_time
