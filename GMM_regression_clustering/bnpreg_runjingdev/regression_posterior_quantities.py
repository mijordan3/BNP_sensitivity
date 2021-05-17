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
    
    """
    Returns the local parameters: the cluster assignments and the shifts, 
    given the vb parameters. 

    Parameters
    ----------
    y : ndarray
        The array of observations shape (num_obs x n_timepoints) 
    x : ndarray 
        The b-spline regression matrix, shape = (n_timepoints, dim).
    vb_params_dict : dictionary
        Dictionary of variational parameters.
    gh_loc : vector
        Locations for gauss-hermite quadrature. 
    gh_weights : vector
        Weights for gauss-hermite quadrature. 

    Returns
    -------
    e_z : ndarray
        The optimal cluster belongings as a function of the variational
        parameters, stored in an array whose (n, k)th entry is the probability
        of the nth datapoint belonging to cluster k
    e_z_free : ndarray 
        The unconstrained parameterization of e_z
    e_b : ndarray
        The expectation of the shifts under its optimal 
        Gaussian distribution. 
    e_b2 : ndarray 
        The expectation of the squared shifts under 
        its optimal Gaussian distribution. 

    """

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
    """
    Returns a monte-carlo estimate of the expected number of posterior 
    in-sample number of clusters

    Parameters
    ----------
    y : ndarray
        The array of observations shape (num_obs x n_timepoints) 
    x : ndarray 
        The b-spline regression matrix, shape = (n_timepoints, dim).
    vb_params_dict : dictionary
        Dictionary of variational parameters.
    prior_params_dict : dictionary
        Dictionary of prior parameters.
    gh_loc : vector
        Locations for gauss-hermite quadrature. 
    gh_weights : vector
        Weights for gauss-hermite quadrature. 
    threshold : integer
        Minimum number of observations required to count as a ``cluster". 
        If zero, we actually compute the expectation analytically
    n_samples : integer 
        Number of samples in a monte-carlo estimate 
    prng_key : 
        A jax.random.PRNGKey 
    
    Returns
    -------
    float 
        the expected number of posterior in-sample clusters

    """



    e_z  = get_optimal_local_params_from_vb_dict(y, x,
                                                vb_params_dict,
                                                prior_params_dict,
                                                gh_loc,
                                                gh_weights)[0]
    
    if threshold == 0: 
        # if threshold is zero, we can return the analytic expectation
        return cluster_lib.get_e_num_clusters_from_ez_analytic(e_z)
    else: 
        return cluster_lib.get_e_num_clusters_from_ez(e_z,
                                                      threshold = threshold,
                                                      n_samples = n_samples,
                                                      prng_key = prng_key)
