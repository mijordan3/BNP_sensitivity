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
        paragami.NumericArrayPattern(shape=(k_approx, dim))

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
        -0.5 * (-2 * np.einsum('ni,kj,nij->nk', gamma, centroids, gamma_info) + \
                    np.einsum('ki,kj,nij->nk', centroids, centroids, gamma_info))
    
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

    return z_nat_param

def get_optimal_z(gamma, gamma_info, 
                  stick_means, stick_infos,
                  centroids,
                  gh_loc, gh_weights,
                  use_bnp_prior = True):

    z_nat_param = \
        get_z_nat_params(gamma, gamma_info, 
                         stick_means, stick_infos,
                         centroids,
                         gh_loc, gh_weights,
                         use_bnp_prior)

    log_const = sp.special.logsumexp(z_nat_param, axis=1)
    e_z = np.exp(z_nat_param - np.expand_dims(log_const, axis = 1))

    return e_z, z_nat_param

def get_kl(gamma, gamma_info,
           vb_params_dict, prior_params_dict,
           gh_loc, gh_weights,
           e_z = None,
           use_bnp_prior = True, 
           e_log_phi = None):

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
    e_z_opt, z_nat_param = \
            get_optimal_z(gamma, gamma_info, 
                          stick_means, stick_infos,
                          centroids,
                          gh_loc, gh_weights,
                          use_bnp_prior = use_bnp_prior)
    if e_z is None:
        e_z = e_z_opt
    
    e_loglik = np.sum(e_z * z_nat_param) 


    # entropy term
    entropy = get_entropy(stick_means, stick_infos, e_z,
                          gh_loc, gh_weights)

    # prior term
    e_log_prior = get_e_log_prior(stick_means, stick_infos,
                                  centroids,
                                  prior_params_dict,
                                  gh_loc, gh_weights)

    elbo = e_log_prior + entropy + e_loglik
    
    if e_log_phi is not None:

        e_log_pert = e_log_phi(vb_params_dict['stick_params']['stick_means'],
                               vb_params_dict['stick_params']['stick_infos'])
                                                            
        elbo = elbo + e_log_pert
        
    return -1 * elbo.squeeze()



