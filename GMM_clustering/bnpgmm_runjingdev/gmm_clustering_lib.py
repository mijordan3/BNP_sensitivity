import jax
import jax.numpy as np
import jax.scipy as sp

import bnpmodeling_runjingdev.cluster_quantities_lib as cluster_lib
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
    cluster_params_paragami = paragami.PatternDict()
    
    # centroids (mean and info) are normal-wishart 
    cluster_params_paragami['means'] = \
        paragami.NumericArrayPattern(shape=(k_approx, dim))
    cluster_params_paragami['lambdas'] = \
        paragami.NumericArrayPattern(shape=(k_approx, ), lb = 0.)
    
    # inverse covariances
    cluster_params_paragami['wishart_scale'] = \
        paragami.pattern_containers.PatternArray(array_shape = (k_approx, ), \
                    base_pattern = paragami.PSDSymmetricMatrixPattern(size=dim))
    
    cluster_params_paragami['wishart_df'] = \
        paragami.NumericArrayPattern(shape =(k_approx, ), lb = dim - 1)
    
    
    # BNP sticks
    # variational distribution for each stick is logitnormal
    stick_params_paragami = paragami.PatternDict()
    stick_params_paragami['stick_means'] = \
        paragami.NumericArrayPattern(shape = (k_approx - 1,))
    stick_params_paragami['stick_infos'] = \
        paragami.NumericArrayPattern(shape = (k_approx - 1,), lb = 1e-4)

    # add the vb_params
    vb_params_paragami['centroid_params'] = cluster_params_paragami
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

    Default prior parameters are those set for the experiments in
    "Evaluating Sensitivity to the Stick Breaking Prior in
    Bayesian Nonparametrics"
    https://arxiv.org/abs/1810.06587

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

    # normal-wishart prior on the centroids and cluster info
    prior_params_dict['prior_centroid_mean'] = np.array([0.0])
    prior_params_paragami['prior_centroid_mean'] = \
        paragami.NumericArrayPattern(shape=(1, ))

    prior_params_dict['prior_lambda'] = np.array([1.0])
    prior_params_paragami['prior_lambda'] = \
        paragami.NumericArrayPattern(shape=(1, ), lb = 0.0)

    prior_params_dict['prior_wishart_df'] = np.array([10.0])
    prior_params_paragami['prior_wishart_df'] = \
        paragami.NumericArrayPattern(shape=(1, ), lb = 0.0)

    prior_params_dict['prior_wishart_rate'] = np.eye(dim)
    prior_params_paragami['prior_wishart_rate'] = \
        paragami.PSDSymmetricMatrixPattern(size=dim)

    return prior_params_dict, prior_params_paragami


##################
# Some useful moments
##################
def get_e_log_determinant(centroids_param_dict): 
    # expected log determinant of a matrix sampled
    # from a wishart 
    
    dim = centroids_param_dict['means'].shape[1]
    
    multivariate_gammas = \
        np.array([sp.special.digamma( \
                  (centroids_param_dict['wishart_df'] + 1 - i) / 2) \
                  for i in range(1, dim+1)]).sum(0)
    
    logdet = np.linalg.slogdet(centroids_param_dict['wishart_scale'])[1]
    
    return multivariate_gammas + logdet + dim * np.log(2), logdet   


##########################
# Expected prior term
##########################
def get_e_log_prior(vb_params_dict,
                    e_log_det, 
                    prior_params_dict,
                    gh_loc, gh_weights):
    
    # get expected prior term

    # dp prior
    alpha = prior_params_dict['dp_prior_alpha']
    stick_means = vb_params_dict['stick_params']['stick_means']
    stick_infos = vb_params_dict['stick_params']['stick_infos']
    dp_prior = \
        modeling_lib.get_e_logitnorm_dp_prior(stick_means,
                                              stick_infos,
                                              alpha,
                                              gh_loc,
                                              gh_weights)

    # wishart prior
    centroid_means = vb_params_dict['centroid_params']['means']
    dim = centroid_means.shape[-1]
        
    lambda_term = - dim * prior_params_dict['prior_lambda'] / \
                            vb_params_dict['centroid_params']['lambdas']
    
    
    diff = centroid_means - prior_params_dict['prior_centroid_mean']
    prior_info = vb_params_dict['centroid_params']['wishart_scale'] * \
                    prior_params_dict['prior_lambda']
    means_term = - vb_params_dict['centroid_params']['wishart_df'] * \
                        np.einsum('ki, kij, kj -> k', diff, prior_info, diff)

    # sum over k
    term1 = 0.5 * (e_log_det + lambda_term + means_term).sum()
    
    
    term2 = (prior_params_dict['prior_wishart_df'] - dim - 1)/2 * \
                e_log_det.sum()
    
    tr_winv_scale = \
        np.einsum('ij, kji -> k',
                  prior_params_dict['prior_wishart_rate'], 
                  vb_params_dict['centroid_params']['wishart_scale'])
    
    term3 = -0.5 * (vb_params_dict['centroid_params']['wishart_df'] * \
                    tr_winv_scale).sum()
    
    return dp_prior + term1 + term2 + term3

##########################
# Entropy
##########################
def get_entropy(vb_params_dict, e_log_det, log_det, e_z, gh_loc, gh_weights):
    # get entropy term

    dim = vb_params_dict['centroid_params']['means'].shape[1]
    
    z_entropy = modeling_lib.multinom_entropy(e_z)
    
    stick_means = vb_params_dict['stick_params']['stick_means']
    stick_infos = vb_params_dict['stick_params']['stick_infos']
    stick_entropy = \
        modeling_lib.get_stick_breaking_entropy(stick_means, stick_infos,
                                gh_loc, gh_weights)
    
    # normal entropy
    lambdas = vb_params_dict['centroid_params']['lambdas']
    normal_entropy = (-0.5 * e_log_det - dim / 2 * np.log(lambdas)).sum()
    
    # wishart entropy
    dfs = vb_params_dict['centroid_params']['wishart_df']
    wishart_entropy = log_det * dfs / 2 + \
                        dfs * dim / 2 * np.log(2) + \
                        sp.special.multigammaln(dfs / 2, dim) - \
                        (dfs - dim - 1) / 2 * e_log_det + \
                        dfs * dim / 2
    
    wishart_entropy = wishart_entropy.sum()
    
    return z_entropy + stick_entropy + normal_entropy + wishart_entropy

##########################
# Likelihood term
##########################
def get_loglik_obs_by_nk(y, centroid_params_dict, e_log_det):
    # returns a n x k matrix whose nkth entry is
    # the likelihood for the nth observation
    # belonging to the kth cluster
    
    dim = y.shape[1]

    # expectation of info under a wishart
    e_scale = np.einsum('k, kij -> kij', 
                        centroid_params_dict['wishart_df'], 
                        centroid_params_dict['wishart_scale'])
    
    # expectation of info times mean
    e_scale_mean = np.einsum('kij, kj -> ki', 
                             e_scale, 
                             centroid_params_dict['means'])
    
    e_scale_mean_scale = dim / centroid_params_dict['lambdas'] + \
                            np.einsum('ki, kij, kj -> k',
                                      centroid_params_dict['means'], 
                                      e_scale, 
                                      centroid_params_dict['means'])
    
    # add in the data
    data2_term = np.einsum('ni, kij, nj -> nk', y, e_scale, y)
    cross_term = np.einsum('ni, ki -> nk', y, e_scale_mean)

    squared_term = data2_term - 2 * cross_term + \
                    np.expand_dims(e_scale_mean_scale, axis = 0)
        
    return - 0.5 * squared_term + 0.5 * np.expand_dims(e_log_det, 0)

##########################
# Optimization over e_z
##########################
def get_z_nat_params(y,
                     vb_params_dict, 
                     e_log_det, 
                     gh_loc, gh_weights,
                     use_bnp_prior = True):

    # get likelihood term
    loglik_obs_by_nk = get_loglik_obs_by_nk(y,
                                            vb_params_dict['centroid_params'], 
                                            e_log_det)

    # get weight term
    stick_means = vb_params_dict['stick_params']['stick_means']
    stick_infos = vb_params_dict['stick_params']['stick_infos']
    
    operand = (stick_means, stick_infos, gh_loc, gh_weights)
    e_log_cluster_probs = jax.lax.cond(use_bnp_prior,
                    operand,
                    lambda x : modeling_lib.get_e_log_cluster_probabilities(*x),
                    operand,
                    lambda x : np.zeros(len(operand[0]) + 1))
    
    z_nat_param = loglik_obs_by_nk + e_log_cluster_probs

    return z_nat_param

def get_optimal_z(y, vb_params_dict, e_log_det, 
                    gh_loc, gh_weights,
                    use_bnp_prior = True):

    z_nat_param = \
        get_z_nat_params(y,
                         vb_params_dict,
                         e_log_det,
                         gh_loc,
                         gh_weights,
                         use_bnp_prior)

    log_const = sp.special.logsumexp(z_nat_param, axis=1)
    e_z = np.exp(z_nat_param - np.expand_dims(log_const, axis = 1))

    return e_z, z_nat_param


def get_kl(y, 
           vb_params_dict,
           prior_params_dict,
           gh_loc, 
           gh_weights,
           e_z = None,
           use_bnp_prior = True, 
           e_log_phi = None):

    """
    Computes the negative ELBO using the data y, at the current variational
    parameters and at the current prior parameters

    Parameters
    ----------
    y : ndarray
        The array of datapoints, one observation per row.
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

    # an expensive moment: only compute once here
    e_log_det, log_det = get_e_log_determinant(vb_params_dict['centroid_params'])

    # get optimal cluster belongings
    e_z_opt, z_nat_param = \
            get_optimal_z(y,
                          vb_params_dict,
                          e_log_det,
                          gh_loc,
                          gh_weights,
                          use_bnp_prior = use_bnp_prior)
    if e_z is None:
        e_z = e_z_opt

    e_loglik = np.sum(e_z * z_nat_param)

    # entropy term
    entropy = get_entropy(vb_params_dict, e_log_det, log_det, 
                          e_z, gh_loc, gh_weights)

    # prior term
    e_log_prior = get_e_log_prior(vb_params_dict,
                                  e_log_det,
                                  prior_params_dict,
                                  gh_loc,
                                  gh_weights)

    elbo = e_log_prior + entropy + e_loglik
    
    if e_log_phi is not None:

        e_log_pert = e_log_phi(vb_params_dict['stick_params']['stick_means'],
                               vb_params_dict['stick_params']['stick_infos'])
                                                            
        elbo = elbo + e_log_pert


    return -1 * elbo.squeeze()



