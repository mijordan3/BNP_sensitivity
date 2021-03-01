# some function to compute 
# posterior quantities of interest

import jax 
import jax.numpy as np

import bnpmodeling_runjingdev.cluster_quantities_lib as cluster_lib

from bnpgmm_runjingdev.gmm_posterior_quantities_lib import \
    get_e_mixture_weights_from_vb_dict, get_e_num_pred_clusters_from_vb_dict

from bnpreg_runjingdev import regression_mixture_lib

def get_optimal_local_params_from_vb_dict(y, x, vb_params_dict, prior_params_dict, 
                                          gh_loc, gh_weights): 
    
    # get vb parameters
    stick_means = vb_params_dict['stick_params']['stick_means']
    stick_infos = vb_params_dict['stick_params']['stick_infos']
    centroids = vb_params_dict['centroids']
    data_info = vb_params_dict['data_info']
    
    # optimal shifts
    e_b, e_b2 = regression_mixture_lib.get_optimal_shifts(y, x, centroids, data_info, prior_params_dict)

    # optimal z's
    ez, ez_free = \
        regression_mixture_lib.get_optimal_z(y, x, 
                                             stick_means, stick_infos,
                                             data_info, centroids,
                                             e_b, e_b2, 
                                             gh_loc, gh_weights, 
                                             prior_params_dict)
    
    return ez, ez_free, e_b, e_b2



# Get the expected posterior number of distinct clusters.
def get_e_num_clusters_from_vb_dict(y, x,
                                    vb_params_dict,
                                    prior_params_dict,
                                    gh_loc, gh_weights,
                                    threshold = 0,
                                    n_samples = 10000,
                                    prng_key = jax.random.PRNGKey(0)):

    e_z  = get_optimal_local_params_from_vb_dict(y, x,
                                                vb_params_dict,
                                                prior_params_dict,
                                                gh_loc,
                                                gh_weights)[0]

    return cluster_lib.get_e_num_clusters_from_ez(e_z,
                                                  threshold = threshold,
                                                  n_samples = n_samples,
                                                  prng_key = prng_key)
