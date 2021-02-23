import jax
import jax.numpy as np

from bnpgmm_runjingdev.gmm_clustering_lib import get_optimal_z

import bnpmodeling_runjingdev.cluster_quantities_lib as cluster_lib
import bnpmodeling_runjingdev.modeling_lib as modeling_lib

########################
# Posterior quantities of interest
#######################
def get_optimal_z_from_vb_dict(y, vb_params_dict, gh_loc, gh_weights,
                               use_bnp_prior = True):

    """
    Returns the optimal cluster belonging probabilities, given the
    variational parameters.

    Parameters
    ----------
    y : ndarray
        The array of datapoints, one observation per row.
    vb_params_dict : dictionary
        Dictionary of variational parameters.
    gh_loc : vector
        Locations for gauss-hermite quadrature. We need this compute the
        expected prior terms.
    gh_weights : vector
        Weights for gauss-hermite quadrature. We need this compute the
        expected prior terms.
    use_bnp_prior : boolean
        Whether or not to use a prior on the cluster mixture weights.
        If True, a DP prior is used.

    Returns
    -------
    e_z : ndarray
        The optimal cluster belongings as a function of the variational
        parameters, stored in an array whose (n, k)th entry is the probability
        of the nth datapoint belonging to cluster k

    """

    # get global vb parameters
    stick_means = vb_params_dict['stick_params']['stick_means']
    stick_infos = vb_params_dict['stick_params']['stick_infos']
    centroids = vb_params_dict['cluster_params']['centroids']
    cluster_info = vb_params_dict['cluster_params']['cluster_info']

    # compute optimal e_z from vb global parameters
    e_z, _ = get_optimal_z(y, stick_means, stick_infos, centroids, cluster_info,
                        gh_loc, gh_weights,
                        use_bnp_prior = use_bnp_prior)

    return e_z

def get_e_mixture_weights_from_vb_dict(vb_params_dict, gh_loc, gh_weights): 
    stick_means = vb_params_dict['stick_params']['stick_means']
    stick_infos = vb_params_dict['stick_params']['stick_infos']
    
    weights = cluster_lib.get_e_cluster_probabilities(stick_means, 
                                                      stick_infos,
                                                      gh_loc,
                                                      gh_weights)
    
    return weights



def get_e_num_pred_clusters_from_vb_dict(vb_params_dict,
                                         n_obs,
                                         threshold = 0,
                                         n_samples = 10000,
                                         prng_key = jax.random.PRNGKey(0)):
    
    # get posterior predicted number of clusters

    stick_means = vb_params_dict['stick_params']['stick_means']
    stick_infos = vb_params_dict['stick_params']['stick_infos']

    return cluster_lib.get_e_num_pred_clusters_from_logit_sticks(stick_means,
                                                                 stick_infos,
                                                                 n_obs,
                                                                 threshold = threshold,
                                                                 n_samples = n_samples,
                                                                 prng_key = prng_key)


# Get the expected posterior number of distinct clusters.
def get_e_num_clusters_from_vb_dict(y, 
                                    vb_params_dict,
                                    gh_loc, gh_weights,
                                    threshold = 0,
                                    n_samples = 10000,
                                    prng_key = jax.random.PRNGKey(0)):

    e_z  = get_optimal_z_from_vb_dict(y, 
                                      vb_params_dict,
                                      gh_loc,
                                      gh_weights,
                                      use_bnp_prior = True)

    return cluster_lib.get_e_num_clusters_from_ez(e_z,
                                                  threshold = threshold,
                                                  n_samples = n_samples,
                                                  prng_key = prng_key)
