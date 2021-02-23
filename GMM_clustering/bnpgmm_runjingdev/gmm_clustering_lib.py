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
    # centroids
    cluster_params_paragami = paragami.PatternDict()
    
    # TODO: make the first dimension k_approx 
    # this matches w regression_lib, and is just more natural ... 
    cluster_params_paragami['centroids'] = \
        paragami.NumericArrayPattern(shape=(dim, k_approx))
    # inverse covariances
    cluster_params_paragami['cluster_info'] = \
        paragami.pattern_containers.PatternArray(array_shape = (k_approx, ), \
                    base_pattern = paragami.PSDSymmetricMatrixPattern(size=dim))

    # BNP sticks
    # variational distribution for each stick is logitnormal
    stick_params_paragami = paragami.PatternDict()
    stick_params_paragami['stick_means'] = \
        paragami.NumericArrayPattern(shape = (k_approx - 1,))
    stick_params_paragami['stick_infos'] = \
        paragami.NumericArrayPattern(shape = (k_approx - 1,), lb = 1e-4)

    # add the vb_params
    vb_params_paragami['cluster_params'] = cluster_params_paragami
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
    # TODO to be consistent change 'alpha' to 'dp_prior_alpha'
    prior_params_dict['alpha'] = np.array([3.0])
    prior_params_paragami['alpha'] = \
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

##########################
# Expected prior term
##########################
def get_e_log_prior(stick_means, stick_infos, centroids, cluster_info,
                        prior_params_dict,
                        gh_loc, gh_weights):
    # get expected prior term

    # dp prior
    alpha = prior_params_dict['alpha']
    dp_prior = \
        modeling_lib.get_e_logitnorm_dp_prior(stick_means, stick_infos,
                                            alpha, gh_loc, gh_weights)

    # wishart prior
    df = prior_params_dict['prior_wishart_df']
    V_inv = prior_params_dict['prior_wishart_rate']
    e_cluster_info_prior = modeling_lib.get_e_log_wishart_prior(cluster_info, df, V_inv)

    # centroid prior
    prior_mean = prior_params_dict['prior_centroid_mean']
    prior_lambda = prior_params_dict['prior_lambda']

    diff = centroids - prior_mean
    prior_info = cluster_info * prior_lambda
    e_centroid_prior = -0.5 * np.einsum('ji, ijk, ki', diff, prior_info, diff)

    return e_cluster_info_prior + e_centroid_prior + dp_prior

##########################
# Entropy
##########################
def get_entropy(stick_means, stick_infos, e_z, gh_loc, gh_weights):
    # get entropy term

    z_entropy = modeling_lib.multinom_entropy(e_z)
    stick_entropy = \
        modeling_lib.get_stick_breaking_entropy(stick_means, stick_infos,
                                gh_loc, gh_weights)

    return z_entropy + stick_entropy

##########################
# Likelihood term
##########################
def get_loglik_obs_by_nk(y, centroids, cluster_info):
    # returns a n x k matrix whose nkth entry is
    # the likelihood for the nth observation
    # belonging to the kth cluster

    dim = np.shape(y)[1]

    assert np.shape(y)[1] == np.shape(centroids)[0]
    assert np.shape(cluster_info)[0] == np.shape(centroids)[1]
    assert np.shape(cluster_info)[1] == np.shape(centroids)[0]

    data2_term = np.einsum('ni, kij, nj -> nk', y, cluster_info, y)
    cross_term = np.einsum('ni, kij, jk -> nk', y, cluster_info, centroids)
    centroid2_term = np.einsum('ik, kij, jk -> k', centroids, cluster_info, centroids)

    squared_term = data2_term - 2 * cross_term + \
                    np.expand_dims(centroid2_term, axis = 0)

    return - 0.5 * squared_term + 0.5 * np.expand_dims(np.linalg.slogdet(cluster_info)[1], 0)

##########################
# Optimization over e_z
##########################
def get_z_nat_params(y, stick_means, stick_infos, centroids, cluster_info,
                        gh_loc, gh_weights,
                        use_bnp_prior = True):

    # get likelihood term
    loglik_obs_by_nk = get_loglik_obs_by_nk(y, centroids, cluster_info)

    # get weight term
    operand = (stick_means, stick_infos, gh_loc, gh_weights)
    e_log_cluster_probs = jax.lax.cond(use_bnp_prior,
                    operand,
                    lambda x : modeling_lib.get_e_log_cluster_probabilities(*x),
                    operand,
                    lambda x : np.zeros(len(operand[0]) + 1))

    z_nat_param = loglik_obs_by_nk + e_log_cluster_probs

    return z_nat_param

def get_optimal_z(y, stick_means, stick_infos, centroids, cluster_info,
                    gh_loc, gh_weights,
                    use_bnp_prior = True):

    z_nat_param = \
        get_z_nat_params(y, stick_means, stick_infos, centroids, cluster_info,
                                    gh_loc, gh_weights,
                                    use_bnp_prior)

    log_const = sp.special.logsumexp(z_nat_param, axis=1)
    e_z = np.exp(z_nat_param - np.expand_dims(log_const, axis = 1))

    return e_z, z_nat_param


def get_kl(y, vb_params_dict, prior_params_dict,
            gh_loc, gh_weights,
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

    # get vb parameters
    stick_means = vb_params_dict['stick_params']['stick_means']
    stick_infos = vb_params_dict['stick_params']['stick_infos']
    centroids = vb_params_dict['cluster_params']['centroids']
    cluster_info = vb_params_dict['cluster_params']['cluster_info']

    # get optimal cluster belongings
    e_z_opt, z_nat_param = \
            get_optimal_z(y, stick_means, stick_infos, centroids, cluster_info,
                            gh_loc, gh_weights, use_bnp_prior = use_bnp_prior)
    if e_z is None:
        e_z = e_z_opt

    e_loglik = np.sum(e_z * z_nat_param)

    # entropy term
    entropy = get_entropy(stick_means, stick_infos, e_z,
                                        gh_loc, gh_weights)

    # prior term
    e_log_prior = get_e_log_prior(stick_means, stick_infos,
                            centroids, cluster_info,
                            prior_params_dict,
                            gh_loc, gh_weights)

    elbo = e_log_prior + entropy + e_loglik
    
    if e_log_phi is not None:

        e_log_pert = e_log_phi(vb_params_dict['stick_params']['stick_means'],
                               vb_params_dict['stick_params']['stick_infos'])
                                                            
        elbo = elbo + e_log_pert


    return -1 * elbo.squeeze()



