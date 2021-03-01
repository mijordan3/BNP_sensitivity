import jax 
import jax.numpy as np

import numpy as onp

import time

from bnpmodeling_runjingdev.sensitivity_lib import get_jac_hvp_fun
from bnpmodeling_runjingdev.bnp_optimization_lib import optimize_kl, \
    update_stick_beta_params, convert_beta_sticks_to_logitnormal

from bnpreg_runjingdev import regression_mixture_lib 
from bnpreg_runjingdev.regression_posterior_quantities import get_optimal_local_params_from_vb_dict
from bnpreg_runjingdev.genomics_utils import regression_lib

from sklearn.cluster import KMeans

from copy import deepcopy

#####################
# functions to initialize 
#####################

# TODO make this function shared with 
# The GMM init
def init_centroids_w_kmeans(beta, k_approx, n_kmeans_init = 10): 
    
    
    n_obs = np.shape(beta)[0]
    dim = np.shape(beta)[1]

    # K means init.
    for i in range(n_kmeans_init):
        km = KMeans(n_clusters = k_approx).fit(beta)
        enertia = km.inertia_
        if (i == 0):
            enertia_best = enertia
            km_best = deepcopy(km)
        elif (enertia < enertia_best):
            enertia_best = enertia
            km_best = deepcopy(km)
    
    init_centroids = np.array(km_best.cluster_centers_)
    
    return init_centroids

def set_params_w_kmeans(y, regressors,
                        vb_params_dict, vb_params_paragami, 
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
    vb_params_dict['centroids'] = \
        init_centroids_w_kmeans(beta, k_approx, n_kmeans_init = 10)

    # intialize shifts
    _data_info = 100. # set to be large to weight the data more than the prior in the init
    e_b_init, e_b2_init = \
        regression_mixture_lib.get_optimal_shifts(y, regressors,
                                                  vb_params_dict['centroids'],
                                                  _data_info,
                                                  prior_params_dict)
    
    # intialize ez -- disregard prior
    loglik_nk = regression_mixture_lib.get_loglik_obs_by_nk(y, 
                                                            regressors,
                                                            vb_params_dict['centroids'], 
                                                            _data_info,
                                                            e_b_init, e_b2_init)
    ez_init = jax.nn.softmax(loglik_nk)
    
    # sort z's from largest to smallest
    perm = np.argsort(-ez_init.sum(0))
    ez_init = ez_init[:, perm]
    vb_params_dict['centroids'] = vb_params_dict['centroids'][perm]    

    # initialize sticks
    print('initializing sticks ...')
    stick_beta1, stick_beta2 = update_stick_beta_params(ez_init, prior_params_dict['dp_prior_alpha'])
    beta_params = np.stack((stick_beta1, stick_beta2), axis = -1)
    
    vb_params_dict['stick_params'] = convert_beta_sticks_to_logitnormal(beta_params, 
                                                                        vb_params_dict['stick_params'],
                                                                        vb_params_paragami['stick_params'], 
                                                                        gh_loc, gh_weights)[0]

#     stick_shape = vb_params_dict['stick_params']['stick_means'].shape
#     vb_params_dict['stick_params']['stick_means'] = np.ones(stick_shape)
#     vb_params_dict['stick_params']['stick_infos'] = np.ones(stick_shape)
    
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
