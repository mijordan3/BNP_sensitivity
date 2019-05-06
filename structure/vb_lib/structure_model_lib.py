import autograd
import autograd.numpy as np
import autograd.scipy as sp

import paragami

from BNP_modeling import cluster_quantities_lib, modeling_lib, optimization_lib

import LinearResponseVariationalBayes.ExponentialFamilies as ef

from sklearn.decomposition import NMF

##########################
# Set up vb parameters
##########################

def get_vb_params_paragami_object(n_obs, n_loci, k_approx,
                                    use_logitnormal_sticks):
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
    if use_logitnormal_sticks:
        # variational distribution for each stick is logitnormal
        vb_params_paragami['ind_mix_stick_propn_mean'] = \
            paragami.NumericArrayPattern(shape = (n_obs, k_approx - 1,))
        vb_params_paragami['ind_mix_stick_propn_info'] = \
            paragami.NumericArrayPattern(shape = (n_obs, k_approx - 1,), lb = 1e-4)
    else:
        # else they are beta distributed
        vb_params_paragami['ind_mix_stick_beta_params'] = \
            paragami.NumericArrayPattern(shape=(n_obs, k_approx - 1, 2), lb = 0.0)

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
def get_e_log_prior(e_log_1m_sticks, e_log_pop_freq, e_log_1m_pop_freq,
                        dp_prior_alpha, allele_prior_alpha,
                        allele_prior_beta):
    # get expected prior term

    # dp prior on individual mixtures
    ind_mix_dp_prior =  (dp_prior_alpha - 1) * np.sum(e_log_1m_sticks)

    # allele frequency prior
    allele_freq_beta_prior = np.sum((allele_prior_alpha - 1) * e_log_pop_freq + \
                                    (allele_prior_beta - 1) * e_log_1m_pop_freq)

    return ind_mix_dp_prior + allele_freq_beta_prior

##########################
# Entropy
##########################
def get_entropy(ind_mix_stick_propn_mean,
                ind_mix_stick_propn_info,
                pop_freq_beta_params,
                e_z, gh_loc, gh_weights,
                use_logitnormal_sticks,
                ind_mix_stick_beta_params = None):
    # get entropy term

    # entropy on population belongings

    z_entropy = -(np.log(e_z + 1e-12) * e_z).sum()

    # entropy of individual admixtures
    if use_logitnormal_sticks:
        stick_entropy = \
            modeling_lib.get_stick_breaking_entropy(
                                    ind_mix_stick_propn_mean,
                                    ind_mix_stick_propn_info,
                                    gh_loc, gh_weights)
    else:
        assert ind_mix_stick_beta_params is not None
        nk = ind_mix_stick_beta_params.shape[0] * ind_mix_stick_beta_params.shape[1]
        stick_entropy = ef.beta_entropy(tau = ind_mix_stick_beta_params.reshape((nk, 2)))

    # beta entropy term
    lk = pop_freq_beta_params.shape[0] * pop_freq_beta_params.shape[1]
    beta_entropy = ef.beta_entropy(tau = pop_freq_beta_params.reshape((lk, 2)))

    return z_entropy + stick_entropy + beta_entropy

##########################
# Likelihood term
##########################
def get_loglik_gene_nlk(g_obs, e_log_pop_freq, e_log_1m_pop_freq):

    genom_loglik_nlk_a = \
        np.einsum('nl, lk -> nlk', g_obs[:, :, 0], e_log_1m_pop_freq) + \
            np.einsum('nl, lk -> nlk', g_obs[:, :, 1] + g_obs[:, :, 2], e_log_pop_freq)

    genom_loglik_nlk_b = \
        np.einsum('nl, lk -> nlk', g_obs[:, :, 0] + g_obs[:, :, 1], e_log_1m_pop_freq) + \
            np.einsum('nl, lk -> nlk', g_obs[:, :, 2], e_log_pop_freq)

    return np.stack((genom_loglik_nlk_a, genom_loglik_nlk_b), axis = -1)

##########################
# Optimization over e_z
##########################
def get_loglik_cond_z(g_obs, e_log_pop_freq, e_log_1m_pop_freq,
                        e_log_cluster_probs):

    # get likelihood of genes
    loglik_gene_nlk = get_loglik_gene_nlk(g_obs, e_log_pop_freq, e_log_1m_pop_freq)

    # log likelihood of population belongings
    n = e_log_cluster_probs.shape[0]
    k = e_log_cluster_probs.shape[1]

    _e_log_cluster_probs = e_log_cluster_probs.reshape(n, 1, k, 1)

    # loglik_obs_by_nlk2 is n_obs x n_loci x k_approx x 2
    loglik_cond_z = loglik_gene_nlk + _e_log_cluster_probs

    return loglik_cond_z

def get_z_opt_from_loglik_cond_z(loglik_cond_z):
    # 2nd axis dimension is k
    # recall that loglik_obs_by_nlk2 is n_obs x n_loci x k_approx x 2
    loglik_cond_z = loglik_cond_z - np.max(loglik_cond_z, axis = 2, keepdims = True)

    log_const = sp.misc.logsumexp(loglik_cond_z, axis = 2, keepdims = True)

    return np.exp(loglik_cond_z - log_const)


def get_e_joint_loglik_from_nat_params(g_obs, e_z,
                                    e_log_pop_freq, e_log_1m_pop_freq,
                                    e_log_sticks, e_log_1m_sticks,
                                    dp_prior_alpha, allele_prior_alpha,
                                    allele_prior_beta,
                                    obs_weights = None,
                                    loci_weights = None,
                                    return_ez = False):
    if obs_weights is not None:
        raise NotImplementedError()
    if loci_weights is not None:
        raise NotImplementedError()

    # log likelihood of individual population belongings
    e_log_cluster_probs = \
        modeling_lib.get_e_log_cluster_probabilities_from_e_log_stick(
                            e_log_sticks, e_log_1m_sticks)

    loglik_cond_z = \
            get_loglik_cond_z(g_obs, e_log_pop_freq,
                                e_log_1m_pop_freq, e_log_cluster_probs)

    if e_z is None:
        # set at optimal e_z
        e_z = get_z_opt_from_loglik_cond_z(loglik_cond_z)

    e_loglik = np.sum(e_z * loglik_cond_z)

    assert(np.isfinite(e_loglik))

    # prior term
    e_log_prior = get_e_log_prior(e_log_1m_sticks,
                            e_log_pop_freq, e_log_1m_pop_freq,
                            dp_prior_alpha, allele_prior_alpha,
                            allele_prior_beta).squeeze()

    assert(np.isfinite(e_log_prior))

    if return_ez:
        return e_log_prior + e_loglik, e_z
    else:
        return e_log_prior + e_loglik


def get_kl(g_obs, vb_params_dict, prior_params_dict,
                    use_logitnormal_sticks,
                    gh_loc = None, gh_weights = None,
                    e_z = None,
                    obs_weights = None,
                    loci_weights = None):

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

    Returns
    -------
    kl : float
        The negative elbo.
    """
    # get prior parameters
    dp_prior_alpha = prior_params_dict['dp_prior_alpha']
    allele_prior_alpha = prior_params_dict['allele_prior_alpha']
    allele_prior_beta = prior_params_dict['allele_prior_beta']

    e_log_sticks, e_log_1m_sticks, \
        e_log_pop_freq, e_log_1m_pop_freq = \
            get_moments_from_vb_params_dict(g_obs, vb_params_dict,
                                    use_logitnormal_sticks = use_logitnormal_sticks,
                                    gh_loc = gh_loc,
                                    gh_weights = gh_weights)
    # joint log likelihood
    log_lik, e_z = get_e_joint_loglik_from_nat_params(g_obs, e_z,
                                e_log_pop_freq, e_log_1m_pop_freq,
                                e_log_sticks, e_log_1m_sticks,
                                dp_prior_alpha, allele_prior_alpha,
                                allele_prior_beta,
                                obs_weights = obs_weights,
                                loci_weights = loci_weights,
                                return_ez = True)

    # entropy term
    pop_freq_beta_params = vb_params_dict['pop_freq_beta_params']
    if use_logitnormal_sticks:
        entropy = get_entropy(vb_params_dict['ind_mix_stick_propn_mean'],
                                vb_params_dict['ind_mix_stick_propn_info'],
                                pop_freq_beta_params,
                                e_z, gh_loc, gh_weights,
                                use_logitnormal_sticks = True).squeeze()
    else:
        beta_params = vb_params_dict['ind_mix_stick_beta_params']
        entropy = get_entropy(None, None,
                            pop_freq_beta_params,
                            e_z, gh_loc, gh_weights,
                            use_logitnormal_sticks = False,
                            ind_mix_stick_beta_params = beta_params).squeeze()

    assert(np.isfinite(entropy))

    elbo = log_lik + entropy

    return -1 * elbo

def get_moments_from_vb_params_dict(g_obs, vb_params_dict,
                                    use_logitnormal_sticks,
                                    gh_loc = None,
                                    gh_weights = None):
    # get expected sticks
    if use_logitnormal_sticks:
        assert gh_loc is not None
        assert gh_weights is not None

        ind_mix_stick_propn_mean = vb_params_dict['ind_mix_stick_propn_mean']
        ind_mix_stick_propn_info = vb_params_dict['ind_mix_stick_propn_info']

        e_log_sticks, e_log_1m_sticks = \
            ef.get_e_log_logitnormal(
                lognorm_means = ind_mix_stick_propn_mean,
                lognorm_infos = ind_mix_stick_propn_info,
                gh_loc = gh_loc,
                gh_weights = gh_weights)
    else:
        ind_mix_stick_beta_params = vb_params_dict['ind_mix_stick_beta_params']
        e_log_sticks, e_log_1m_sticks = \
            modeling_lib.get_e_log_beta(ind_mix_stick_beta_params)

    # population beta parameters
    pop_freq_beta_params = vb_params_dict['pop_freq_beta_params']
    e_log_pop_freq, e_log_1m_pop_freq = \
        modeling_lib.get_e_log_beta(pop_freq_beta_params)

    return e_log_sticks, e_log_1m_sticks, \
                e_log_pop_freq, e_log_1m_pop_freq

###############
# functions for initializing
def cluster_and_get_init(g_obs, k):
    # g_obs should be n_obs x n_loci x 3,
    # a one-hot encoding of genotypes
    assert len(g_obs.shape) == 3

    # convert one-hot encoding to probability of A genotype, {0, 0.5, 1}
    x = g_obs.argmax(axis = 2) / 2

    # run NMF
    model = NMF(n_components=k, init='random')
    init_ind_admix_propn_unscaled = model.fit_transform(x)
    init_pop_allele_freq_unscaled = model.components_.T

    # divide by largest allele frequency, so all numbers between 0 and 1
    denom_pop_allele_freq = np.max(init_pop_allele_freq_unscaled)
    init_pop_allele_freq = init_pop_allele_freq_unscaled / \
                                denom_pop_allele_freq

    # normalize rows
    denom_ind_admix_propn = \
        init_ind_admix_propn_unscaled.sum(axis = 1, keepdims = True)
    init_ind_admix_propn = \
        init_ind_admix_propn_unscaled / denom_ind_admix_propn
    # clip again and renormalize
    init_ind_admix_propn = init_ind_admix_propn.clip(0.05, 0.95)
    init_ind_admix_propn = init_ind_admix_propn / \
                            init_ind_admix_propn.sum(axis = 1, keepdims = True)

    return init_ind_admix_propn, init_pop_allele_freq.clip(0.05, 0.95)

def set_init_vb_params(g_obs, k_approx, vb_params_dict,
                        use_logitnormal_sticks):
    # get initial admixtures, and population frequencies
    init_ind_admix_propn, init_pop_allele_freq = \
            cluster_and_get_init(g_obs, k_approx)

    # set bnp parameters for individual admixture
    # set mean to be logit(stick_breaking_propn), info to be 1
    stick_break_propn = \
        cluster_quantities_lib.get_stick_break_propns_from_mixture_weights(init_ind_admix_propn)
    if use_logitnormal_sticks:
        ind_mix_stick_propn_mean = np.log(stick_break_propn) - np.log(1 - stick_break_propn)
        ind_mix_stick_propn_info = np.ones(stick_break_propn.shape)
        vb_params_dict['ind_mix_stick_propn_mean'] = ind_mix_stick_propn_mean
        vb_params_dict['ind_mix_stick_propn_info'] = ind_mix_stick_propn_info
    else:
        ind_mix_stick_beta_param1 = np.ones(stick_break_propn.shape)
        ind_mix_stick_beta_param2 = (1 - stick_break_propn) / stick_break_propn
        vb_params_dict['ind_mix_stick_beta_params'] = \
            np.concatenate((ind_mix_stick_beta_param1[:, :, None],
                            ind_mix_stick_beta_param2[:, :, None]), axis = 2)

    # set beta paramters for population paramters
    # set beta = 1, alpha to have the correct mean
    pop_freq_beta_params1 = init_pop_allele_freq / (1 - init_pop_allele_freq)
    pop_freq_beta_params2 = np.ones(init_pop_allele_freq.shape)
    pop_freq_beta_params = np.concatenate((pop_freq_beta_params1[:, :, None],
                                       pop_freq_beta_params2[:, :, None]), axis = 2)

    vb_params_dict['pop_freq_beta_params'] = pop_freq_beta_params

    return vb_params_dict

def assert_optimizer(g_obs, vb_opt_dict, vb_params_paragami,
                        prior_params_dict, gh_loc, gh_weights,
                        use_logitnormal_sticks):
    # this function checks that vb_opt_dict are at a kl optimum for the given
    # prior parameters

    # get loss as a function of vb parameters
    get_free_vb_params_loss = paragami.FlattenFunctionInput(
                                    original_fun=get_kl,
                                    patterns = vb_params_paragami,
                                    free = True,
                                    argnums = 1)
    # cache other parameters
    get_free_vb_params_loss_cached = \
        lambda x : get_free_vb_params_loss(g_obs, x, prior_params_dict, use_logitnormal_sticks,
                                        gh_loc, gh_weights)

    grad_get_loss = autograd.grad(get_free_vb_params_loss_cached)
    linf_grad = np.max(np.abs(grad_get_loss(vb_params_paragami.flatten(vb_opt_dict, free = True))))

    assert  linf_grad < 1e-5, 'error: {}'.format(linf_grad)
