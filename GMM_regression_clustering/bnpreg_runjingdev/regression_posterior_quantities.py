# some function to compute 
# posterior quantities of interest

import jax 
import jax.numpy as np

import bnpmodeling_runjingdev.cluster_quantities_lib as cluster_lib

from bnpgmm_runjingdev.gmm_posterior_quantities_lib import \
    get_e_mixture_weights_from_vb_dict, get_e_num_pred_clusters_from_vb_dict

from bnpreg_runjingdev import regression_mixture_lib



# Get the expected posterior number of distinct clusters.
def get_e_num_clusters_from_vb_dict(y, x,
                                    vb_params_dict,
                                    prior_params_dict,
                                    gh_loc, gh_weights,
                                    threshold = 0,
                                    n_samples = 10000,
                                    prng_key = jax.random.PRNGKey(0)):

    e_z  = regression_mixture_lib.get_optimal_z(y, x,
                                       vb_params_dict,
                                       gh_loc,
                                       gh_weights, 
                                       prior_params_dict)[0]
    
    if threshold == 0: 
        # if threshold is zero, we can return the analytic expectation
        return cluster_lib.get_e_num_clusters_from_ez_analytic(e_z)
    else: 
        return cluster_lib.get_e_num_clusters_from_ez(e_z,
                                                      threshold = threshold,
                                                      n_samples = n_samples,
                                                      prng_key = prng_key)
