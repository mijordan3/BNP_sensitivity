import jax
import jax.numpy as np

import numpy as onp

from copy import deepcopy

from sklearn.cluster import KMeans

from bnpmodeling_runjingdev.bnp_optimization_lib import optimze_kl

from bnpgmm_runjingdev.gmm_clustering_lib import get_kl
from bnpgmm_runjingdev.gmm_posterior_quantities_lib import get_optimal_z_from_vb_dict

def cluster_and_get_k_means_inits(y, 
                                  vb_params_paragami,
                                  n_kmeans_init = 1,
                                  z_init_eps=0.05,
                                  seed = 1):
    """
    Runs k-means to initialize the variational parameters.

    Parameters
    ----------
    y : ndarray
        The array of datapoints, one observation per row.
    vb_params_paragami : paragami Patterned Dictionary
        A paragami patterned dictionary that contains the variational parameters.
    n_kmeans_init : int
        The number of re-initializations for K-means.
    z_init_eps : float
        The weight given to the clusters a data does not belong to
        after running K-means

    Returns
    -------
    vb_params_dict : dictionary
        Dictionary of variational parameters.
    init_free_par : vector
        Vector of the free variational parameters
    e_z_init : ndarray
        Array encoding cluster belongings as found by kmeans
    """

    # get dictionary of vb parameters
    vb_params_dict = vb_params_paragami.random()

    # set seed
    onp.random.seed(seed)

    # data parameters
    k_approx = np.shape(vb_params_dict['cluster_params']['centroids'])[1]
    n_obs = np.shape(y)[0]
    dim = np.shape(y)[1]

    # K means init.
    for i in range(n_kmeans_init):
        km = KMeans(n_clusters = k_approx).fit(y)
        enertia = km.inertia_
        if (i == 0):
            enertia_best = enertia
            km_best = deepcopy(km)
        elif (enertia < enertia_best):
            enertia_best = enertia
            km_best = deepcopy(km)

    e_z_init = onp.full((n_obs, k_approx), z_init_eps)
    for n in range(len(km_best.labels_)):
        e_z_init[n, km_best.labels_[n]] = 1.0 - z_init_eps
    e_z_init /= np.expand_dims(np.sum(e_z_init, axis = 1), axis = 1)

    vb_params_dict['cluster_params']['centroids'] = np.array(km_best.cluster_centers_.T)
    
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

    vb_params_dict['cluster_params']['cluster_info'] = np.array(cluster_info_init)

    init_free_par = vb_params_paragami.flatten(vb_params_dict, free = True)

    return init_free_par, vb_params_dict, e_z_init


def optimize_gmm(y,
                 vb_params_dict,
                 vb_params_paragami,
                 prior_params_dict, 
                 gh_loc, gh_weights, 
                 e_log_phi = None, 
                 run_newton = True): 
    
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
    vb_opt_dict, vb_opt, out, optim_time = optimze_kl(get_kl_loss,
                                                       vb_params_dict, 
                                                       vb_params_paragami, 
                                                       run_newton = run_newton)
    
    ez_opt = get_optimal_z_from_vb_dict(y, vb_opt_dict, gh_loc, gh_weights)
    
    return vb_opt_dict, vb_opt, ez_opt, out, optim_time
    