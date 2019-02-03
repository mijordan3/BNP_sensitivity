import autograd
import autograd.numpy as np
import autograd.scipy as sp

import paragami

import dp_modeling_lib

import LinearResponseVariationalBayes.ExponentialFamilies as ef

##########################
# Set up vb parameters
##########################

def get_vb_params_paragami_object(n_obs, n_loci, k_approx):
    """
    Returns a paragami patterned dictionary
    that stores the variational parameters.

    Parameters
    ----------

    Returns
    -------
    vb_params_dict : dictionary
        A dictionary that contains the variational parameters.

    vb_params_paragami : paragami patterned dictionary
        A paragami patterned dictionary that contains the variational parameters.

    """

    vb_params_paragami = paragami.PatternDict()

    # variational beta parameters for population allele frequencies
    vb_params_paragami['pop_freq_beta_params'] = \
        paragami.NumericArrayPattern(shape=(n_loci, k_approx, 2), lb = 0.0)

    # BNP sticks
    # variational distribution for each stick is logitnormal
    vb_params_paragami['ind_mix_stick_propn_mean'] = \
        paragami.NumericArrayPattern(shape = (n_obs, k_approx - 1,))
    vb_params_paragami['ind_mix_stick_propn_info'] = \
        paragami.NumericArrayPattern(shape = (n_obs, k_approx - 1,), lb = 1e-4)

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

    # DP prior parameter for the individual admixtures
    prior_params_dict['dp_prior_alpha'] = np.array([3.0])
    prior_params_paragami['dp_prior_alpha'] = \
        paragami.NumericArrayPattern(shape=(1, ), lb = 0.0)

    # prior on the allele frequencies
    # beta distribution parameters
    prior_params_dict['allele_prior_alpha'] = np.array([1.])
    prior_params_dict['allele_prior_beta'] = np.array([1.])
    prior_params_paragami['allele_prior_alpha'] = \
        paragami.NumericArrayPattern(shape=(1, ), lb = 0.0)
    prior_params_paragami['allele_prior_beta'] = \
        paragami.NumericArrayPattern(shape=(1, ), lb = 0.0)

    return prior_params_dict, prior_params_paragami

##########################
# Expected prior term
##########################
def get_e_log_prior(ind_mix_stick_propn_mean, ind_mix_stick_propn_info,
                        e_log_p, e_log_1mp,
                        dp_prior_alpha, allele_prior_alpha,
                        allele_prior_beta,
                        gh_loc, gh_weights):
    # get expected prior term

    # dp prior on individual mixtures
    ind_mix_dp_prior = \
        dp_modeling_lib.get_e_logitnorm_dp_prior(ind_mix_stick_propn_mean,
                                            ind_mix_stick_propn_info,
                                            dp_prior_alpha, gh_loc, gh_weights)

    # allele frequency prior
    allele_freq_beta_prior = np.sum((allele_prior_alpha - 1) * e_log_p + \
                                    (allele_prior_beta - 1) * e_log_1mp)

    return ind_mix_dp_prior + allele_freq_beta_prior

##########################
# Entropy
##########################
def get_entropy(ind_mix_stick_propn_mean, ind_mix_stick_propn_info,
                    pop_freq_beta_params,
                    e_z, gh_loc, gh_weights):
    # get entropy term

    # entropy on population belongings

    z_entropy = -(np.log(e_z + 1e-12) * e_z).sum()

    # entropy of individual admixtures
    stick_entropy = \
        dp_modeling_lib.get_stick_breaking_entropy(
                                ind_mix_stick_propn_mean,
                                ind_mix_stick_propn_info,
                                gh_loc, gh_weights)

    # beta entropy term
    lk = pop_freq_beta_params.shape[0] * pop_freq_beta_params.shape[1]
    beta_entropy = ef.beta_entropy(tau = pop_freq_beta_params.reshape((lk, 2)))

    return z_entropy + stick_entropy + beta_entropy

##########################
# Likelihood term
##########################
def get_loglik_gene_nlk(g_obs, e_log_p, e_log_1mp):

    genom_loglik_nlk_a = \
        np.einsum('nl, lk -> nlk', g_obs[:, :, 0], e_log_1mp) + \
            np.einsum('nl, lk -> nlk', g_obs[:, :, 1] + g_obs[:, :, 2], e_log_1mp)

    genom_loglik_nlk_b = \
        np.einsum('nl, lk -> nlk', g_obs[:, :, 0] + g_obs[:, :, 1], e_log_1mp) + \
            np.einsum('nl, lk -> nlk', g_obs[:, :, 2], e_log_1mp)

    return np.stack((genom_loglik_nlk_a, genom_loglik_nlk_b), axis = -1)

##########################
# Optimization over e_z
##########################
def get_loglik_cond_z(g_obs, e_log_p, e_log_1mp,
                        ind_mix_stick_propn_mean, ind_mix_stick_propn_info,
                        gh_loc, gh_weights):

    # get likelihood of genes
    loglik_gene_nlk = get_loglik_gene_nlk(g_obs, e_log_p, e_log_1mp)

    # log likelihood of population belongings
    n = ind_mix_stick_propn_mean.shape[0]
    k = ind_mix_stick_propn_mean.shape[1] + 1

    e_log_cluster_probs = \
        dp_modeling_lib.get_e_log_cluster_probabilities(
                        ind_mix_stick_propn_mean, ind_mix_stick_propn_info,
                        gh_loc, gh_weights).reshape(n, 1, k, 1)

    # loglik_obs_by_nlk2 is n_obs x n_loci x k_approx x 2
    loglik_cond_z = loglik_gene_nlk + e_log_cluster_probs

    return loglik_cond_z

def get_z_opt_from_loglik_cond_z(loglik_cond_z):
    # 2nd axis dimension is k
    # recall that loglik_obs_by_nlk2 is n_obs x n_loci x k_approx x 2
    loglik_cond_z = loglik_cond_z - np.max(loglik_cond_z, axis = 2, keepdims = True)

    log_const = sp.misc.logsumexp(loglik_cond_z, axis = 2, keepdims = True)

    return np.exp(loglik_cond_z - log_const)

def get_kl(g_obs, vb_params_dict, prior_params_dict,
                    gh_loc, gh_weights,
                    e_z = None,
                    data_weights = None):

    """
    Computes the negative ELBO using the data y, at the current variational
    parameters and at the current prior parameters

    Parameters
    ----------
    g_obs : ndarray
        The array of one-hot encoded genotypes, of shape (n_obs, n_loci, 3)
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
        parameters, stored in an array whose (n, l, k, i)th entry is the probability
        of the nth datapoint at locus l and chromosome i belonging to cluster k.
        If ``None``, we set the optimal z.
    data_weights : ndarray of shape (number of observations) x 1 (optional)
        Weights for each datapoint in g_obs.

    Returns
    -------
    kl : float
        The negative elbo.
    """
    # get prior parameters
    dp_prior_alpha = prior_params_dict['dp_prior_alpha']
    allele_prior_alpha = prior_params_dict['allele_prior_alpha']
    allele_prior_beta = prior_params_dict['allele_prior_beta']

    # get vb parameters
    ind_mix_stick_propn_mean = vb_params_dict['ind_mix_stick_propn_mean']
    ind_mix_stick_propn_info = vb_params_dict['ind_mix_stick_propn_info']
    pop_freq_beta_params = vb_params_dict['pop_freq_beta_params']

    # expected log beta and expected log(1 - beta)
    e_log_p, e_log_1mp = dp_modeling_lib.get_e_log_beta(pop_freq_beta_params)

    # get optimal cluster belongings
    loglik_cond_z = \
            get_loglik_cond_z(g_obs, e_log_p, e_log_1mp,
                            ind_mix_stick_propn_mean, ind_mix_stick_propn_info,
                            gh_loc, gh_weights)

    e_z_opt = get_z_opt_from_loglik_cond_z(loglik_cond_z)

    if e_z is None:
        e_z = e_z_opt

    # weight data if necessary, and get likelihood of y
    if data_weights is not None:
        raise NotImplementedError()
    else:
        e_loglik = np.sum(e_z * loglik_cond_z)

    assert(np.isfinite(e_loglik))

    # entropy term
    entropy = get_entropy(ind_mix_stick_propn_mean,
                                        ind_mix_stick_propn_info,
                                        pop_freq_beta_params,
                                        e_z, gh_loc, gh_weights).squeeze()
    assert(np.isfinite(entropy))

    # prior term
    e_log_prior = get_e_log_prior(ind_mix_stick_propn_mean, ind_mix_stick_propn_info,
                            e_log_p, e_log_1mp,
                            dp_prior_alpha, allele_prior_alpha,
                            allele_prior_beta,
                            gh_loc, gh_weights).squeeze()

    assert(np.isfinite(e_log_prior))

    elbo = e_log_prior + entropy + e_loglik

    return -1 * elbo
