import jax

import jax.numpy as np
import jax.scipy as sp

from jax import random

import scipy as osp

from bnpmodeling_runjingdev import stick_integration_lib


def get_mixture_weights_from_stick_break_propns(stick_break_propns):
    """
    Computes mixture weights from stick breaking proportions.

    Parameters
    ----------
    stick_break_propns : ndarray
        Array of stick breaking proportions, with sticks along last dimension

    Returns
    -------
    mixture_weights : ndarray
        An array  the same size as stick_break_propns,
        with the mixture weights computed for each row of
        stick breaking proportions.

    """

    # if input is a vector, make it a 1 x k_approx array
    if len(np.shape(stick_break_propns)) == 1:
        stick_break_propns = np.array([stick_break_propns])

    # number of components
    k_approx = np.shape(stick_break_propns)[-1]
    # number of mixtures
    ones_shape = stick_break_propns.shape[0:-1] + (1,)

    stick_break_propns_1m = 1 - stick_break_propns
    stick_remain = np.concatenate((np.ones(ones_shape),
                                   np.cumprod(stick_break_propns_1m, axis = -1)), axis = -1)
    stick_add = np.concatenate((stick_break_propns,
                                np.ones(ones_shape)), axis = -1)

    mixture_weights = (stick_remain * stick_add).squeeze()

    return mixture_weights

def get_mixture_weights_from_logitnorm_params(stick_propn_mean, 
                                              stick_propn_info,
                                              gh_loc,
                                              gh_weights):
    """
    Computes the mixture weights from stick breaking proportions from logit-normal 
    parameters. 

    Parameters
    ----------
    stick_propn_mean : ndarray
        Mean parameters for the logit of the
        stick-breaking proportions, of shape ...  x (k_approx-1)
    stick_propn_info : ndarray
        Info parameters for the logit of the
        stick-breaking proportions, of shape ...  x (k_approx-1)
    gh_loc : vector
        Locations for gauss-hermite quadrature. We need this to compute the
        expectations wrt to stick-breaking proportions.
    gh_weights : vector
        Weights for gauss-hermite quadrature. We need this to compute the
        expectations wrt to stick-breaking proportions.

    
    Returns
    -------
    ndarray
        array of mixture weights, of shape ... x k_approx
    """
    
    e_stick_lengths = \
        stick_integration_lib.get_e_logitnormal(stick_propn_mean, stick_propn_info, gh_loc, gh_weights)
    
    # can just multiply expectations bc variational distribution on sticks 
    # are independent 
    return get_mixture_weights_from_stick_break_propns(e_stick_lengths)

def _sample_stick_propn(stick_propn_mean, stick_propn_info, n_samples, 
                       prng_key = jax.random.PRNGKey(0)): 
    
    # sample stick breaking-proportions from logitnormal parameters
    
    shape = (n_samples, ) + stick_propn_mean.shape 
    normal_samples = jax.random.normal(key = prng_key,
                                       shape = shape)
    
    # sample sticks: shape is n_samples x n_obs x (k_approx - 1) 
    sds = np.expand_dims((1 / np.sqrt(stick_propn_info)), axis = 0)
    means = np.expand_dims(stick_propn_mean, axis = 0)
    
    return sp.special.expit(normal_samples * sds + means)
    

def sample_weights_from_logitnormal_sticks(stick_propn_mean,
                                           stick_propn_info,
                                           n_samples = 1,
                                           prng_key = jax.random.PRNGKey(0)): 
    """
    Samples mixture weights from 
    from logitnormal stick-breaking parameters,
    ``stick_propn_mean`` and ``stick_propn_info``.

    Parameters
    ----------
    stick_propn_mean : ndarray
        Mean parameters for the logit of the
        stick-breaking proportions, of length (k_approx - 1)
    stick_propn_info : ndarray
        parameters for the logit of the
        stick-breaking proportions, of length (k_approx - 1)
    n_samples : int
        number of samples.
    seed : int
        random seed
        
    Returns
    -------
    ndarray
        a n_samples x k_approx array of sampled mixture weights.
    """
    
    assert stick_propn_mean.shape == stick_propn_info.shape
    assert len(stick_propn_mean.shape) == 1
    
    # sample sticks proportions from logitnormal
    # this is n_samples x (k_approx - 1)
    stick_propn_samples = _sample_stick_propn(stick_propn_mean,
                                             stick_propn_info,
                                             n_samples, 
                                             prng_key)
                                        

    # get sampled mixture weights weights
    # this is n_samples x k_approx
    weight_samples = \
        get_mixture_weights_from_stick_break_propns(stick_propn_samples)

    return weight_samples
    
def get_e_num_pred_clusters_from_mixture_weights(mixture_weights, 
                                                 n_obs, 
                                                 threshold): 
    
    """ 
    Given a ... x k_approx array of mixture weights, 
    computes the expected number of clusters 
    in a new dataset of size `n_obs`. 
    
    Parameters
    ----------
    mixture_weights : ndarray
        mixture weights of shape ... x k_approx
    n_obs : int
        Number of observations in a dataset
    threshold : int
        Miniumum number of observations for a cluster to be counted.
    
    Returns
    ----------
    ndarray 
        expected number of predicted clusters with at least 
        ``theshold`` observations, one for each row of 
        ``mixture_weights``. 
    """
    
    assert isinstance(threshold, int)
    
    # probability cluster has no observations
    subtr_weight = (1 - mixture_weights)**(n_obs)
    
    # probability that each cluster has i observations
    for i in range(1, threshold+1):
        subtr_weight += \
            osp.special.comb(n_obs, i) * \
                mixture_weights**i * (1 - mixture_weights)**(n_obs - i)

    return np.sum(1 - subtr_weight, axis = -1)
    

def get_e_num_pred_clusters_from_logit_sticks(stick_propn_mean, 
                                            stick_propn_info,
                                            n_obs, 
                                            threshold = 0,
                                            n_samples = 1,
                                            prng_key = jax.random.PRNGKey(0),
                                            return_samples = False):
    """
    Computes, using Monte Carlo, the expected number of predicted clusters 
    with at least t observations in a new sample of size n_obs. 

    Parameters
    ----------
    stick_propn_mean : ndarray
        Mean parameters for the logit of the
        stick-breaking proportions, of length (k_approx - 1)
    stick_propn_info : ndarray
        parameters for the logit of the
        stick-breaking proportions, of length (k_approx - 1)
    threshold : int
        Miniumum number of observations for a cluster to be counted.
    n_obs : int
        Number of observations in a dataset
    n_samples : int
        Number of Monte Carlo samples used to compute the expected
        number of clusters.
    seed : int
        random seed

    Returns
    -------
    float
        The expected number of clusters with at least ``threshold`` observations
        in a dataset of size n_obs
    """

    weight_samples = sample_weights_from_logitnormal_sticks(stick_propn_mean,
                                                            stick_propn_info,
                                                            n_samples = n_samples,
                                                            prng_key = prng_key)
    
    n_clusters_sampled = get_e_num_pred_clusters_from_mixture_weights(weight_samples, 
                                                                      n_obs = n_obs, 
                                                                      threshold = threshold)

    if return_samples: 
        return n_clusters_sampled
    else: 
        return np.mean(n_clusters_sampled)



def get_e_num_clusters_from_ez_analytic(e_z):
    """
    Analytically computes the expected number of clusters from cluster
    belongings e_z.
    Parameters
    ----------
    e_z : ndarray
        Array whose (n, k)th entry is the probability of the nth
        datapoint belonging to cluster k.
    Returns
    -------
    float
        The expected number of clusters in the dataset.
    """

    k = np.shape(e_z)[1]
    return k - np.sum(np.prod(1 - e_z, axis = 0))



def _sample_ez_from_unif_samples(e_z, unif_samples):
    # e_z is n_obs x k
    # unif_sample should be a matrix of shape n_samples x n_obs
    
    # returns a n_samples x n_obs x k matrix encoding sampled 
    # cluster belongings
    
    
    # these assertions mess up if we use jit ... 
#     assert np.all(e_z >= 0.)
#     assert np.all(e_z <= 1.)
    
    n_obs = e_z.shape[0]
    k_approx = e_z.shape[1]
    
    e_z_cumsum = e_z.cumsum(1)
#     assert np.all(np.abs(e_z_cumsum[:, -1] - 1.) < 1e-8)
        
    e_z_cumsum0 = np.hstack((np.zeros((n_obs, 1)),
                             e_z_cumsum[:, 0:(k_approx-1)]))
    
    assert len(unif_samples.shape) == 2
    assert unif_samples.shape[1] == n_obs

    # get which cluster the sample belongs to
    z_sample = (e_z_cumsum[None, :, :] > unif_samples[:, :, None]) & \
                (e_z_cumsum0[None, :, :] < unif_samples[:, :, None])

    return z_sample

def sample_ez(e_z, 
              n_samples = 1, 
              prng_key = random.PRNGKey(0)): 
    """
    Samples cluster belongings from ez
    ----------
    e_z : ndarray
        Array whose (n, k)th entry is the probability of the nth
        datapoint belonging to cluster k
    n_samples : int
        Number of samples 
    seed : int 
        random seed
        
    Returns
    -------
    ndarray
        a n_samples x n_obs x k_approx array of 
        cluster belongings. 
    """

 
    # e_z is of shape n x k
    # index (n,k) is probability of n-th observation 
    # belonging to cluster k
    
    n_obs = e_z.shape[0]
    
    # draw uniform samples
    unif_samples = random.uniform(key = prng_key, 
                                  shape = (n_samples, n_obs))

    # one-hot encoding of zs from uniform samples
    z_samples_one_hot = _sample_ez_from_unif_samples(e_z, unif_samples)
    
    # shape is n_samples x n x k
    return z_samples_one_hot
    
def get_e_num_clusters_from_ez(e_z,
                               threshold = 0,
                               n_samples = 1,
                               prng_key = random.PRNGKey(0),
                               return_samples = False):
    """
    Computes the expected number of clusters with at least ```threshold``
    observations from cluster belongings e_z.
    Parameters
    ----------
    e_z : ndarray
        Array whose (n, k)th entry is the probability of the nth
        datapoint belonging to cluster k
    threshold : int
        Miniumum number of observations for a cluster to be counted.
    n_samples : int
        Number of Monte Carlo samples used to compute the expected
        number of clusters.
    seed : int 
        random seed
        
    Returns
    -------
    float
        The expected number of clusters with at least 
        ``threshold`` observations
    """

    # z_sample is a n_samples x n_obs x k_approx matrix of cluster belongings
    z_sample = sample_ez(e_z, 
                         n_samples = n_samples, 
                         prng_key = prng_key)
    
    # this is n_samples x k_approx: 
    # for each sample, the number of individuals in each cluster
    counts_sampled = z_sample.sum(1)
    
    # for each sample, the number of clusters above some threshold
    n_clusters_sampled = (counts_sampled > threshold).sum(1)
    
    if return_samples: 
        return n_clusters_sampled
    else: 
        return np.mean(n_clusters_sampled)


