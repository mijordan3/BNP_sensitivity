# some function to compute 
# posterior quantities of interest

import jax 
import jax.numpy as np

from bnpreg_runjingdev import regression_mixture_lib

from bnpgmm_runjingdev.gmm_posterior_quantities_lib import get_e_mixture_weights_from_vb_dict

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

# def get_e_mixture_weights_from_vb_dict(vb_params_dict, gh_loc, gh_weights): 
#     stick_means = vb_params_dict['stick_params']['stick_means']
#     stick_infos = vb_params_dict['stick_params']['stick_infos']
    
#     weights = cluster_lib.get_e_cluster_probabilities(stick_means, 
#                                                       stick_infos,
#                                                       gh_loc,
#                                                       gh_weights)
    
#     return weights



# def get_e_num_pred_clusters_from_vb_dict(vb_params_dict,
#                                          n_obs,
#                                          threshold = 0,
#                                          n_samples = 10000,
#                                          prng_key = jax.random.PRNGKey(0)):
    
#     # get posterior predicted number of clusters

#     stick_means = vb_params_dict['stick_params']['stick_means']
#     stick_infos = vb_params_dict['stick_params']['stick_infos']

#     return cluster_lib.get_e_num_pred_clusters_from_logit_sticks(stick_means,
#                                                                  stick_infos,
#                                                                  n_obs,
#                                                                  threshold = threshold,
#                                                                  n_samples = n_samples,
#                                                                  prng_key = prng_key)


# # Get the expected posterior number of distinct clusters.
# def get_e_num_clusters_from_vb_dict(y, 
#                                     vb_params_dict,
#                                     gh_loc, gh_weights,
#                                     threshold = 0,
#                                     n_samples = 10000,
#                                     prng_key = jax.random.PRNGKey(0)):

#     e_z  = get_optimal_z_from_vb_dict(y, 
#                                       vb_params_dict,
#                                       gh_loc,
#                                       gh_weights,
#                                       use_bnp_prior = True)

#     return cluster_lib.get_e_num_clusters_from_ez(e_z,
#                                                   threshold = threshold,
#                                                   n_samples = n_samples,
#                                                   prng_key = prng_key)
