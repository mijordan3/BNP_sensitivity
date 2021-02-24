import jax
import jax.numpy as np
import jax.scipy as sp

import bnpmodeling_runjingdev.modeling_lib as modeling_lib

import bnpgmm_runjingdev.gmm_clustering_lib as gmm_lib

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

    _, vb_params_paragami = \
        gmm_lib.get_vb_params_paragami_object(dim, k_approx)
    
    # don't need cluster infos's
    # delete cluster params and add only centroids
    vb_params_paragami.__delitem__('cluster_params')
    
    
    # centroids
    vb_params_paragami['centroids'] = \
        paragami.NumericArrayPattern(shape=(k_approx, dim))
    
    # info of data 
    vb_params_paragami['data_info'] = \
        paragami.NumericArrayPattern(shape=(1,), lb = 0.)

    vb_params_dict = vb_params_paragami.random()

    return vb_params_dict, vb_params_paragami



##########################
# Set up prior parameters
##########################
def get_default_prior_params():
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

    prior_params_dict['prior_centroid_info'] = np.array([0.1])
    prior_params_paragami['prior_centroid_info'] = \
        paragami.NumericArrayPattern(shape=(1, ), lb = 0.0)
    
    # normal prior on shifts
    prior_params_dict['prior_shift_mean'] = np.array([0.0])
    prior_params_paragami['prior_shift_mean'] = \
        paragami.NumericArrayPattern(shape=(1, ))

    prior_params_dict['prior_shift_info'] = np.array([0.1])
    prior_params_paragami['prior_shift_info'] = \
        paragami.NumericArrayPattern(shape=(1, ), lb = 0.0)
    
    # gamma prior on data info
    prior_params_dict['prior_data_info_shape'] = np.array([0.1])
    prior_params_paragami['prior_data_info_shape'] = \
        paragami.NumericArrayPattern(shape=(1, ), lb = 0.0)

    prior_params_dict['prior_data_info_rate'] = np.array([10.0])
    prior_params_paragami['prior_data_info_rate'] = \
        paragami.NumericArrayPattern(shape=(1, ), lb = 0.0)
    
    return prior_params_dict, prior_params_paragami


##########################
# Likelihood term
##########################
def get_loglik_obs_by_nk(y, x, centroids, data_info, e_b, e_b2):
    # returns a N by k_approx matrix where the (n,k)th entry is the
    # expected log likelihood of the nth observation when it belongs to
    # component k.

    num_time_points = x.shape[0]

    # y is (obs x time points) = (n x t)
    # x is (time points x basis vectors) = (t x b)
    # beta is (clusters x basis vectors) = (k x b)
    x_times_beta = np.einsum('tb,kb->tk', x, centroids)
    
    # mu_nkt = beta_k^T x_t + b_nk

    # \sum_t (y_nt - mu_nkt) ^ 2 =
    #   \sum_t (y_nt^2 - 2 * y_nt * mu_nkt + mu_nkt ^ 2) =
    #   \sum_t (linear_term +                quad_term)

    # \sum_t E(y^2 - 2 * y * mu) =
    #        E(sum(y_t^2) - 2 * sum(y) * beta_k^T x - 2 * sum(y_t) * b_nk):
    linear_term = \
        np.sum(y ** 2, axis=1, keepdims=True) + \
        -2 * np.einsum('nt,tk->nk', y, x_times_beta) + \
        -2 * np.sum(y, axis = 1, keepdims = True) * e_b

    # This is E(mu^2) =
    #         E[(beta_k^T x)^2 +
    #           2 * b_n * beta_k^T x +
    #           b_n^2]
    quad_term = np.sum(x_times_beta ** 2, axis=0, keepdims=True) + \
                2 * np.sum(x_times_beta, axis=0, keepdims=True) * e_b + \
                e_b2 * num_time_points

    square_term = -0.5 * data_info * (linear_term + quad_term)

    # We have already summed the (y - mu)^2 terms over time points,so
    # we need to multiply the e_log_info_y by the number of points we've
    # summed over.
    log_info_term = \
        0.5 * np.log(data_info) * \
        num_time_points
    
    return  square_term + log_info_term

def get_optimal_shifts(y, x, centroids, data_info, prior_params_dict): 
    num_time_points = x.shape[0]

    # y is (obs x time points) = (n x t)
    # x is (time points x basis vectors) = (t x b)
    # centroids is (clusters x basis vectors) = (k x b)
    x_times_beta = np.einsum('tb,kb->tk', x, centroids)
    
    # ydiff is (y - xbeta), but summed over time 
    ydiff = (np.expand_dims(y, axis = -1) - \
             np.expand_dims(x_times_beta, axis = 0)).sum(1) * data_info
    
    prior_diff = prior_params_dict['prior_shift_info'] * prior_params_dict['prior_shift_mean']
    
    e_b = (ydiff + prior_diff) / (num_time_points  * data_info + prior_params_dict['prior_shift_info'])
    e_b2 = 1 / (num_time_points  * data_info + prior_params_dict['prior_shift_info'])
    
    return e_b, e_b2


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

    e_z = jax.nn.softmax(z_nat_param, axis = 1)
    
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



