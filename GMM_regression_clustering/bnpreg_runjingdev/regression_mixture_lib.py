import jax
import jax.numpy as np
import jax.scipy as sp

import numpy as onp

import bnpmodeling_runjingdev.modeling_lib as modeling_lib

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
        Dimension of the regressors.
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
    
    # sticks
    vb_params_paragami['stick_params'] = modeling_lib.get_stick_paragami_object(k_approx)

    # centroids
    vb_params_paragami['centroids'] = \
        paragami.NumericArrayPattern(shape=(k_approx, dim))
    
    vb_params_paragami['centroids_covar'] = \
        paragami.pattern_containers.PatternArray(array_shape = (k_approx, ), \
                    base_pattern = paragami.PSDSymmetricMatrixPattern(size=dim))
    
    # info of data 
    vb_params_paragami['data_info_alpha'] = \
        paragami.NumericArrayPattern(shape=(k_approx,), lb = 0.)
    
    vb_params_paragami['data_info_beta'] = \
        paragami.NumericArrayPattern(shape=(k_approx,), lb = 0.)

    vb_params_dict = vb_params_paragami.random()

    return vb_params_dict, vb_params_paragami

##########################
# Set up prior parameters
##########################
def get_default_prior_params():
    """
    Returns a paragami patterned dictionary
    that stores the prior parameters.
    
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

    prior_params_dict['prior_centroid_info'] = np.array([0.1])
    prior_params_paragami['prior_centroid_info'] = \
        paragami.NumericArrayPattern(shape=(1, ), lb = 0.0)
        
    # gamma prior on data info
    prior_params_dict['prior_data_info_shape'] = np.array([10.])
    prior_params_paragami['prior_data_info_shape'] = \
        paragami.NumericArrayPattern(shape=(1, ), lb = 0.0)

    prior_params_dict['prior_data_info_scale'] = np.array([0.05])
    prior_params_paragami['prior_data_info_scale'] = \
        paragami.NumericArrayPattern(shape=(1, ), lb = 0.0)
    
    return prior_params_dict, prior_params_paragami

##########################
# entropy term
##########################
def get_entropy(vb_params_dict, e_z,
                gh_loc, gh_weights):
    
    # entropy on memberships
    z_entropy = modeling_lib.multinom_entropy(e_z)
    
    # entropy on sticks
    stick_means = vb_params_dict['stick_params']['stick_means']
    stick_infos = vb_params_dict['stick_params']['stick_infos']
    
    stick_entropy = \
        modeling_lib.get_stick_breaking_entropy(stick_means, stick_infos,
                                gh_loc, gh_weights)
    
    # entropy on centroids
    # negative here because the entropy function expects infos, not covariances
    centroid_entropy = - modeling_lib.\
        multivariate_normal_entropy(vb_params_dict['centroids_covar']) 
    
    # entropy on information 
    info_entropy = modeling_lib.gamma_entropy(vb_params_dict['data_info_alpha'], 
                                              vb_params_dict['data_info_beta'])
    
    return z_entropy + stick_entropy + centroid_entropy + info_entropy

##########################
# prior term 
##########################

def _get_gamma_moments(alpha, beta): 
    
    e_info = modeling_lib.get_e_gamma(alpha, beta)

    e_log_info = modeling_lib.get_e_log_gamma(alpha, beta)
        
    return e_info, e_log_info

def get_e_log_prior(vb_params_dict,
                    prior_params_dict,
                    gh_loc, gh_weights):
    
    # get expected prior term

    # dp prior
    alpha = prior_params_dict['dp_prior_alpha']
    stick_means = vb_params_dict['stick_params']['stick_means']
    stick_infos = vb_params_dict['stick_params']['stick_infos']
    dp_prior = \
        modeling_lib.get_e_logitnorm_dp_prior(stick_means, stick_infos,
                                              alpha, gh_loc, gh_weights)
    
    
    # prior on data info 
    prior_shape = prior_params_dict['prior_data_info_shape']
    prior_scale = prior_params_dict['prior_data_info_scale']
    
    e_info, e_log_info = _get_gamma_moments(vb_params_dict['data_info_alpha'], 
                                            vb_params_dict['data_info_beta'])
    
    data_info_prior = (prior_shape - 1) * e_log_info - e_info / prior_scale
    data_info_prior = data_info_prior.sum()

    # centroid prior
    # these prior parameters are just scalars, which makes things easier
    prior_mean = prior_params_dict['prior_centroid_mean']
    prior_info = prior_params_dict['prior_centroid_info']
    
    centroid_means = vb_params_dict['centroids']
    centroid_covars = vb_params_dict['centroids_covar']
    
    e_centroid_prior = -0.5 * prior_info * \
                        (np.einsum('kii -> k', centroid_covars).sum() + \
                        np.sum(centroid_means ** 2) - \
                        2 * np.sum(centroid_means * prior_mean))
    
    return dp_prior + e_centroid_prior + data_info_prior
    
    
##########################
# Likelihood term
##########################
def get_loglik_obs_by_nk(y, x, vb_params_dict):
    # returns a N by k_approx matrix where the (n,k)th entry is the
    # expected log likelihood of the nth observation when it belongs to
    # component k.

    num_time_points = x.shape[0]
    
    e_info, e_log_info = _get_gamma_moments(vb_params_dict['data_info_alpha'], 
                                            vb_params_dict['data_info_beta'])
    
    print('removing loglik term')
    e_info = e_info * 0.
    e_log_info = e_log_info * 0. - 200
        
    centroids = vb_params_dict['centroids']
    centroids_covar = vb_params_dict['centroids_covar']

    # y is (obs x time points) = (n x t)
    # x is (time points x basis vectors) = (t x b)
    # centroids is (clusters x basis vectors) = (k x b)
    x_times_beta = np.einsum('tb,kb->tk', x, centroids)
    
    
    # this is E(y^2 - 2 y(x\beta))
    # shape is n x k
    linear_term = \
        np.sum(y ** 2, axis=1, keepdims=True) + \
        -2 * np.einsum('nt,tk->nk', y, x_times_beta) 

    # This is E((x\beta)^2) 
    # vector of length k
    e_xbeta2 = np.sum(x_times_beta ** 2, axis=0) + \
                np.einsum('ti, kij, tj -> k', x, centroids_covar, x)
        
    e_xbeta2 = np.expand_dims(e_xbeta2, axis = 0) 

    square_term = -0.5 * np.expand_dims(e_info, axis = 0) * (linear_term + e_xbeta2)

    # We have already summed the (y - mu)^2 terms over time points,so
    # we need to multiply the e_log_info_y by the number of points we've
    # summed over.
    log_info_term = \
        0.5 * np.expand_dims(e_log_info, axis = 0) * \
        num_time_points
        
    return  square_term + log_info_term

##########################
# Optimization over e_z
##########################
def get_z_nat_params(y, x, 
                     vb_params_dict, 
                     gh_loc, gh_weights, 
                     prior_params_dict):

    # get likelihood term
    loglik_obs_by_nk = get_loglik_obs_by_nk(y, x, vb_params_dict) 

    # get weight term    
    stick_means = vb_params_dict['stick_params']['stick_means']
    stick_infos = vb_params_dict['stick_params']['stick_infos']
    
    e_log_cluster_probs = modeling_lib.\
        get_e_log_cluster_probabilities(stick_means, stick_infos,
                                        gh_loc, gh_weights)
    
    z_nat_param = loglik_obs_by_nk + e_log_cluster_probs 

    return z_nat_param

def get_optimal_z(y, x, 
                  vb_params_dict,
                  gh_loc, gh_weights, 
                  prior_params_dict):

    z_nat_param = \
        get_z_nat_params(y, x, 
                         vb_params_dict,
                         gh_loc, gh_weights, 
                         prior_params_dict)

    e_z = jax.nn.softmax(z_nat_param, axis = 1)
    
    return e_z, z_nat_param


def get_kl(y, x,
           vb_params_dict, prior_params_dict,
           gh_loc, gh_weights,
           e_z = None,
           e_log_phi = None):

    """
    Computes the negative ELBO using the data y, at the current variational
    parameters and at the current prior parameters

    Parameters
    ----------
    gamma : ndarray
        The array of observations (num_genes x n_timepoints) 
    x : ndarray 
        The b-spline regression matrix, shape = (n_timepoints, n_basis).
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
        
    # get optimal cluster belongings
    e_z_opt, z_nat_param = \
            get_optimal_z(y, x, 
                          vb_params_dict,
                          gh_loc, gh_weights, 
                          prior_params_dict)
    if e_z is None:
        e_z = e_z_opt
    
    e_loglik = np.sum(e_z * z_nat_param) 
    
    # entropy term
    entropy = get_entropy(vb_params_dict, e_z,
                          gh_loc, gh_weights)

    # prior term
    e_log_prior = get_e_log_prior(vb_params_dict,
                                  prior_params_dict,
                                  gh_loc, gh_weights)
    
    elbo = e_log_prior + entropy + e_loglik
        
    if e_log_phi is not None:

        e_log_pert = e_log_phi(vb_params_dict['stick_params']['stick_means'],
                               vb_params_dict['stick_params']['stick_infos'])
                                                            
        elbo = elbo + e_log_pert
        
    return -1 * elbo.squeeze()



