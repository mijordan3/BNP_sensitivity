import jax
import jax.numpy as np

import numpy as onp

from bnpmodeling_runjingdev.bnp_optimization_lib import optimize_kl, init_centroids_w_kmeans

from bnpgmm_runjingdev.gmm_clustering_lib import get_kl
from bnpgmm_runjingdev.gmm_posterior_quantities_lib import get_optimal_z_from_vb_dict

def cluster_and_get_k_means_inits(y, 
                                  vb_params_paragami,
                                  n_kmeans_init = 1,
                                  seed = 1):
    """
    Runs k-means to initialize the variational parameters.

    Parameters
    ----------
    y : array
        The array of datapoints, one observation per row.
    vb_params_paragami : paragami Patterned Dictionary
        A paragami patterned dictionary that contains the variational parameters.
    n_kmeans_init : int
        The number of re-initializations for K-means.

    Returns
    -------
    vb_params_dict : dictionary
        Dictionary of initialized variational parameters. 
    init_free_par : vector
        Vector of the initialized variational parameters, unconstrained. 
    e_z_init : ndarray
        Array encoding cluster assignments as found by kmeans
    """

    # get dictionary of vb parameters
    vb_params_dict = vb_params_paragami.random()

    # data parameters
    k_approx = np.shape(vb_params_dict['centroid_params']['means'])[0]
    n_obs = np.shape(y)[0]
    dim = np.shape(y)[1]
        
    # intialize centriods
    init_centroids, km_best = init_centroids_w_kmeans(y, 
                                                      k_approx,
                                                      n_kmeans_init = 10, 
                                                      seed = seed)
    
    vb_params_dict['centroid_params']['means'] = np.array(km_best.cluster_centers_)
    
    # intialize ez's: 
    # doesn't actually matter for optimization ...
    # just for plotting the init
    e_z_init = onp.zeros((n_obs, k_approx))
    for n in range(len(km_best.labels_)):
        e_z_init[n, km_best.labels_[n]] = 1.0 
    
    # set stick parameters to one
    vb_params_dict['stick_params']['stick_propn_mean'] = np.ones(k_approx - 1)
    vb_params_dict['stick_params']['stick_propn_info'] = np.ones(k_approx - 1)

    # Set inital inv. covariances
    cluster_info_init = onp.zeros((k_approx, dim, dim))
    for k in range(k_approx):
        indx = onp.argwhere(km_best.labels_ == k).flatten()

        if len(indx) <= (y.shape[1] + 1):
            # if there's less than one datapoint in the cluster,
            # the covariance is not defined.
            cluster_info_init[k, :, :] = onp.eye(dim)
        else:
            resid_k = y[indx, :] - km_best.cluster_centers_[k, :]
            cluster_info_init_ = np.linalg.inv(np.cov(resid_k.T) + \
                                    np.eye(dim) * 1e-4)
            # symmetrize ... there might be some numerical issues otherwise
            cluster_info_init[k, :, :] = 0.5 * (cluster_info_init_ + cluster_info_init_.T)
    
    vb_params_dict['centroid_params']['lambdas'] = np.ones(k_approx)
    
    init_df = dim
    vb_params_dict['centroid_params']['wishart_df'] = np.ones(k_approx) * init_df
    vb_params_dict['centroid_params']['cluster_info'] = np.array(cluster_info_init) / init_df

    init_free_par = vb_params_paragami.flatten(vb_params_dict, free = True)

    return init_free_par, vb_params_dict, e_z_init


def optimize_gmm(y,
                 vb_params_dict,
                 vb_params_paragami,
                 prior_params_dict, 
                 gh_loc, gh_weights, 
                 e_log_phi = None, 
                 run_lbfgs = True,
                 run_newton = True): 
    
    """
    Parameters 
    ----------
    y : ndarray
        The array of datapoints, one observation per row.
    vb_params_dict : dictionary
        A dictionary that contains the initial variational parameters.
    vb_params_paragami : paragami patterned dictionary
        A paragami patterned dictionary that contains the variational parameters.
    get_grad : callable, optional
         Returns the gradient of `get_kl_loss` as 
         function of vb parameters (in flattened space). 
         If none, this is computed automatically using jax derivatives.
    get_hvp : callable, optional
        Returns the hessian vector product as 
        function of vb parameters (in flattened space) and 
        and some vector of equal length as the vb parameters.
        If none, this is computed automatically using jax derivatives.
    run_lbfgs : boolean, optional
        Whether to run LBFGS. At least one of `run_blfgs` and 
        `run_newton` must be true. 
    run_newton : boolean, optional
        Whether to run newton-ncg. At least one of `run_blfgs` and 
        `run_newton` must be true. 
        
    Returns
    ----------
    vb_opt_dict : dictionary
        A dictionary that contains the optimized variational parameters.
    vb_opt : array 
        The unconstrained vector of optimized variational parameters.
    out : 
        The output of scipy.optimize.minimize.
    optim_time : 
        The time elapsed, not including compile times. 
    """

    
    ###################
    # Define loss
    ###################
    def get_kl_loss(vb_params_free): 
        
        vb_params_dict = vb_params_paragami.fold(vb_params_free, free = True)
    
        return get_kl(y,
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
    
    ez_opt = get_optimal_z_from_vb_dict(y, vb_opt_dict, gh_loc, gh_weights)
    
    return vb_opt_dict, vb_opt, ez_opt, out, optim_time
    