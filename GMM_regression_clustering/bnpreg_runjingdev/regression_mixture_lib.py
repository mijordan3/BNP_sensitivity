import jax
import jax.numpy as np
import jax.scipy as sp

import bnpmodeling_runjingdev.cluster_quantities_lib as cluster_lib
import bnpmodeling_runjingdev.modeling_lib as modeling_lib
import bnpmodeling_runjingdev.functional_sensitivity_lib as func_sens_lib

from bnpgmm_runjingdev.gmm_clustering_lib import get_entropy

import paragami

##########################
# Set up vb parameters
##########################

def get_vb_params_paragami_object(dim, k_approx):
    """
    Returns a paragami patterned dictionary
    that stores the variational parameters.

    Parameters
    ----------
    dim : int
        Dimension of the datapoints.
    k_approx : int
        Number of components in the model.

    Returns
    -------
    vb_params_dict : dictionary
        A dictionary that contains the variational parameters.

    vb_params_paragami : paragami patterned dictionary
        A paragami patterned dictionary that contains the variational parameters.

    """

    vb_params_paragami = paragami.PatternDict()

    # cluster parameters
    # centroids
    vb_params_paragami['centroids'] = \
        paragami.NumericArrayPattern(shape=(dim, k_approx))

    # BNP sticks
    # variational distribution for each stick is logitnormal
    stick_params_paragami = paragami.PatternDict()
    stick_params_paragami['stick_means'] = \
        paragami.NumericArrayPattern(shape = (k_approx - 1,))
    stick_params_paragami['stick_infos'] = \
        paragami.NumericArrayPattern(shape = (k_approx - 1,), lb = 1e-4)

    # add the vb_params
    vb_params_paragami['stick_params'] = stick_params_paragami

    vb_params_dict = vb_params_paragami.random()

    return vb_params_dict, vb_params_paragami

##########################
# Set up prior parameters
##########################
def get_default_prior_params(dim):
    """
    Returns a paragami patterned dictionary
    that stores the prior parameters.
    
    Parameters
    ----------
    dim : int
        Dimension of the datapoints.

    Returns
    -------
    prior_params_dict : dictionary
        A dictionary that contains the prior parameters.

    prior_params_paragami : paragami Patterned Dictionary
        A paragami patterned dictionary that contains the prior parameters.

    """

    prior_params_dict = dict()
    prior_params_paragami = paragami.PatternDict()

    # DP prior parameter
    prior_params_dict['dp_prior_alpha'] = np.array([3.0])
    prior_params_paragami['dp_prior_alpha'] = \
        paragami.NumericArrayPattern(shape=(1, ), lb = 0.0)

    # normal prior on the centroids
    prior_params_dict['prior_centroid_mean'] = np.array([0.0])
    prior_params_paragami['prior_centroid_mean'] = \
        paragami.NumericArrayPattern(shape=(1, ))

    prior_params_dict['prior_centroid_info'] = np.array([0.001])
    prior_params_paragami['prior_centroid_info'] = \
        paragami.NumericArrayPattern(shape=(1, ), lb = 0.0)

    return prior_params_dict, prior_params_paragami

##########################
# Expected prior term
##########################
def get_e_log_prior(stick_means, stick_infos, 
                    centroids,
                    prior_params_dict,
                    gh_loc, gh_weights):
    
    # get expected prior term

    # dp prior
    alpha = prior_params_dict['dp_prior_alpha']
    dp_prior = \
        modeling_lib.get_e_logitnorm_dp_prior(stick_means, stick_infos,
                                              alpha, gh_loc, gh_weights)

    # centroid prior
    prior_mean = prior_params_dict['prior_centroid_mean']
    prior_info = prior_params_dict['prior_centroid_info']

    e_centroid_prior = sp.stats.norm.pdf(centroids, 
                                         loc = prior_mean, 
                                         scale = 1 / np.sqrt(prior_info)).sum()

    return e_centroid_prior + dp_prior

##########################
# Likelihood term
##########################
def get_loglik_obs_by_nk(gamma, gamma_info, centroids):
    # returns a n x k matrix whose nkth entry is
    # the likelihood for the nth observation
    # belonging to the kth cluster
    
    loglik_nk = \
        -0.5 * (-2 * np.einsum('ni,kj,nij->nk', gamma, centroids, gamma_infos) + \
                    np.einsum('ki,kj,nij->nk', centroids, centroids, gamma_infos))
    
    return loglik_nk

##########################
# Optimization over e_z
##########################
def get_z_nat_params(gamma, gamma_info, 
                     stick_means, stick_infos,
                     centroids,
                     gh_loc, gh_weights,
                     use_bnp_prior = True):

    # get likelihood term
    loglik_obs_by_nk = get_loglik_obs_by_nk(gamma, gamma_info, centroids)

    # get weight term
    operand = (stick_means, stick_infos, gh_loc, gh_weights)
    e_log_cluster_probs = jax.lax.cond(use_bnp_prior,
                    operand,
                    lambda x : modeling_lib.get_e_log_cluster_probabilities(*x),
                    operand,
                    lambda x : np.zeros(len(operand[0]) + 1))

    z_nat_param = loglik_obs_by_nk + e_log_cluster_probs

    return z_nat_param, loglik_obs_by_nk

def get_optimal_z(gamma, gamma_info, 
                  stick_means, stick_infos,
                  centroids,
                  gh_loc, gh_weights,
                  use_bnp_prior = True):

    z_nat_param, loglik_obs_by_nk= \
        get_z_nat_params(gamma, gamma_info, 
                         stick_means, stick_infos,
                         centroids,
                         gh_loc, gh_weights,
                         use_bnp_prior)

    log_const = sp.special.logsumexp(z_nat_param, axis=1)
    e_z = np.exp(z_nat_param - np.expand_dims(log_const, axis = 1))

    return e_z, loglik_obs_by_nk


def get_kl(gamma, gamma_info,
           vb_params_dict, prior_params_dict,
           gh_loc, gh_weights,
           e_z = None,
           use_bnp_prior = True):

    """
    Computes the negative ELBO using the data y, at the current variational
    parameters and at the current prior parameters

    Parameters
    ----------
    gamma : ndarray
        The array of regression coefficients (N, d). 
    gamma_info : ndarray 
        The array of information matrices of regression coefficients, 
        shape = (N, d, d).
    vb_params_dict : dictionary
        Dictionary of variational parameters.
    prior_params_dict : dictionary
        Dictionary of prior parameters.
    gh_loc : vector
        Locations for gauss-hermite quadrature. We need this compute the
        expected prior terms.
    gh_weights : vector
        Weights for gauss-hermite quadrature. We need this compute the
        expected prior terms.
    e_z : ndarray (optional)
        The optimal cluster belongings as a function of the variational
        parameters, stored in an array whose (n, k)th entry is the probability
        of the nth datapoint belonging to cluster k.
        If ``None``, we set the optimal z.
    use_bnp_prior : boolean
        Whether or not to use a prior on the cluster mixture weights.
        If True, a DP prior is used.

    Returns
    -------
    kl : float
        The negative elbo.
    """

    # get vb parameters
    stick_means = vb_params_dict['stick_params']['stick_means']
    stick_infos = vb_params_dict['stick_params']['stick_infos']
    centroids = vb_params_dict['centroids']
    
    # get optimal cluster belongings
    e_z_opt, loglik_obs_by_nk = \
            get_optimal_z(gamma, gamma_info, 
                          stick_means, stick_infos,
                          centroids,
                          gh_loc, gh_weights,
                          use_bnp_prior = use_bnp_prior)
    if e_z is None:
        e_z = e_z_opt

    e_loglik_obs = np.sum(e_z * loglik_obs_by_nk)

    # likelihood of z
    if use_bnp_prior:
        e_loglik_ind = modeling_lib.loglik_ind(stick_means, stick_infos, e_z,
                                               gh_loc, gh_weights)
    else:
        e_loglik_ind = 0.

    e_loglik = e_loglik_ind + e_loglik_obs

    # entropy term
    entropy = get_entropy(stick_means, stick_infos, e_z,
                                        gh_loc, gh_weights)

    # prior term
    e_log_prior = get_e_log_prior(stick_means, stick_infos,
                            centroids, cluster_info,
                            prior_params_dict,
                            gh_loc, gh_weights)

    elbo = e_log_prior + entropy + e_loglik

    return -1 * elbo.squeeze()



# ########################
# # Posterior quantities of interest
# #######################
# def get_optimal_z_from_vb_dict(y, vb_params_dict, gh_loc, gh_weights,
#                                use_bnp_prior = True):

#     """
#     Returns the optimal cluster belonging probabilities, given the
#     variational parameters.

#     Parameters
#     ----------
#     y : ndarray
#         The array of datapoints, one observation per row.
#     vb_params_dict : dictionary
#         Dictionary of variational parameters.
#     gh_loc : vector
#         Locations for gauss-hermite quadrature. We need this compute the
#         expected prior terms.
#     gh_weights : vector
#         Weights for gauss-hermite quadrature. We need this compute the
#         expected prior terms.
#     use_bnp_prior : boolean
#         Whether or not to use a prior on the cluster mixture weights.
#         If True, a DP prior is used.

#     Returns
#     -------
#     e_z : ndarray
#         The optimal cluster belongings as a function of the variational
#         parameters, stored in an array whose (n, k)th entry is the probability
#         of the nth datapoint belonging to cluster k

#     """

#     # get global vb parameters
#     stick_means = vb_params_dict['stick_params']['stick_means']
#     stick_infos = vb_params_dict['stick_params']['stick_infos']
#     centroids = vb_params_dict['cluster_params']['centroids']
#     cluster_info = vb_params_dict['cluster_params']['cluster_info']

#     # compute optimal e_z from vb global parameters
#     e_z, _ = get_optimal_z(y, stick_means, stick_infos, centroids, cluster_info,
#                         gh_loc, gh_weights,
#                         use_bnp_prior = use_bnp_prior)

#     return e_z

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
