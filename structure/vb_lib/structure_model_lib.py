import jax
import jax.numpy as np
import jax.scipy as sp

import numpy as onp

import paragami

import bnpmodeling_runjingdev.functional_sensitivity_lib as func_sens_lib

from bnpmodeling_runjingdev import cluster_quantities_lib, modeling_lib
import bnpmodeling_runjingdev.exponential_families as ef

from sklearn.decomposition import NMF

import warnings

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
    n_obs : integer
        The number of observations
    n_loci : integer
        The number of loci per observation
    k_approx : integer
        The number of components in the model
    use_logitnormal_sticks : boolean
        Whether to use a logitnormal approximation to infer the sticks.

    Returns
    -------
    vb_params_dict : dictionary
        A dictionary that contains the variational parameters.
        The beta parameters are for population frequencies are
        stored in 'pop_freq_beta_params'.
        If use_logitnormal_sticks = True, then we model the sticks
        for the individual admixtures using logitnormals,
        whose means and infos are stored in 'ind_mix_stick_propn_mean'
        and 'ind_mix_stick_propn_info'.
        Else, we use a beta approximation to the sticks, and
        these are stored in 'ind_mix_stick_beta_params'

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
            paragami.NumericArrayPattern(shape = (n_obs, k_approx - 1,),
                                            lb = 1e-4)
    else:
        # else they are beta distributed
        vb_params_paragami['ind_mix_stick_beta_params'] = \
            paragami.NumericArrayPattern(shape=(n_obs, k_approx - 1, 2),
                                            lb = 0.0)

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
    allele_freq_beta_prior = (allele_prior_alpha - 1) * np.sum(e_log_pop_freq) + \
                            (allele_prior_beta - 1) * np.sum(e_log_1m_pop_freq)

    return ind_mix_dp_prior + allele_freq_beta_prior

##########################
# Entropy
##########################
def get_entropy(vb_params_dict, gh_loc, gh_weights):

    # entropy of individual admixtures
    use_logitnormal_sticks = 'ind_mix_stick_propn_mean' in vb_params_dict.keys()
    if use_logitnormal_sticks:
        stick_entropy = \
            modeling_lib.get_stick_breaking_entropy(
                                    vb_params_dict['ind_mix_stick_propn_mean'],
                                    vb_params_dict['ind_mix_stick_propn_info'],
                                    gh_loc, gh_weights)
    else:
        ind_mix_stick_beta_params = vb_params_dict['ind_mix_stick_beta_params']
        nk = ind_mix_stick_beta_params.shape[0] * \
                ind_mix_stick_beta_params.shape[1]
        stick_entropy = \
            ef.beta_entropy(tau = ind_mix_stick_beta_params.reshape((nk, 2)))

    # beta entropy term
    pop_freq_beta_params = vb_params_dict['pop_freq_beta_params']
    lk = pop_freq_beta_params.shape[0] * pop_freq_beta_params.shape[1]
    beta_entropy = ef.beta_entropy(tau = pop_freq_beta_params.reshape((lk, 2)))

    return stick_entropy + beta_entropy

##########################
# Likelihood term
##########################
def get_loglik_gene_nl(g_obs_nl, e_log_pop_freq_l, e_log_1m_pop_freq_l):

    g_obs_nl0 = g_obs_nl[0]
    g_obs_nl1 = g_obs_nl[1]
    g_obs_nl2 = g_obs_nl[2]

    loglik_a = \
        g_obs_nl0 * e_log_1m_pop_freq_l + \
            (g_obs_nl1 + g_obs_nl2) * e_log_pop_freq_l

    loglik_b = \
        (g_obs_nl0 + g_obs_nl1) * e_log_1m_pop_freq_l + \
            g_obs_nl2 * e_log_pop_freq_l

    # returns k_approx x 2 array
    return np.stack((loglik_a, loglik_b), axis = -1)

def get_e_loglik_nl(g_obs_nl, e_log_pop_freq_l, e_log_1m_pop_freq_l,
                    e_log_cluster_probs_n, detach_ez):
    
    # returns z-optimized log-likelihood for observation-n at locus-l

    # get loglikelihood of observations at loci n,l
    loglik_gene_nl = get_loglik_gene_nl(g_obs_nl, e_log_pop_freq_l, e_log_1m_pop_freq_l)

    # add individual belongings
    loglik_cond_z_nl = np.expand_dims(e_log_cluster_probs_n, axis = 1) + loglik_gene_nl

    # individal x chromosome belongings
    e_z_nl = jax.nn.softmax(loglik_cond_z_nl, axis = 0)

    if detach_ez:
        e_z_nl = jax.lax.stop_gradient(e_z_nl)

    # log likelihood
    loglik_nl = np.sum(loglik_cond_z_nl * e_z_nl)

    # entropy term: save this because the z's won't be available later
    # compute the entropy
    z_entropy_nl = (sp.special.entr(e_z_nl)).sum()

    return np.array([loglik_nl, z_entropy_nl])

# def get_e_loglik(g_obs,
#                     e_log_pop_freq, e_log_1m_pop_freq, \
#                     e_log_sticks, e_log_1m_sticks,
#                     detach_ez): 

#     n_obs = g_obs.shape[0]
#     n_loci = g_obs.shape[1]
    
#     e_log_cluster_probs = \
#         modeling_lib.get_e_log_cluster_probabilities_from_e_log_stick(
#                             e_log_sticks, e_log_1m_sticks)
#     def body_fun(val, i): 
#         n = i % n_obs 
#         l = i // n_obs
#         return get_e_loglik_nl(g_obs[n, l], e_log_pop_freq[l], e_log_1m_pop_freq[l],
#                         e_log_cluster_probs[n], detach_ez) + val

#     scan_fun = lambda val, x : (body_fun(val, x), None)
    
#     init_val = np.array([0., 0.])
#     out = jax.lax.scan(scan_fun, init_val,
#                         xs = np.arange(n_obs * n_loci))[0]

#     e_loglik = out[0]
#     z_entropy = out[1]
    
#     return e_loglik, z_entropy 

def get_e_loglik_n(g_obs_n, e_log_pop_freq, e_log_1m_pop_freq, \
                    e_log_cluster_probs_n,
                    detach_ez):
    
    # inner loop throug loci
    
    body_fun = lambda val, x : get_e_loglik_nl(x[0], x[1], x[2],
                                        e_log_cluster_probs_n, detach_ez) + \
                                        val

    scan_fun = lambda val, x : (body_fun(val, x), None)

    init_val = np.array([0., 0.])

    out = jax.lax.scan(scan_fun, init_val,
                        xs = (g_obs_n, e_log_pop_freq, e_log_1m_pop_freq))[0]

    e_loglik_n = out[0]
    z_entropy_n = out[1]

    return e_loglik_n, z_entropy_n

def get_e_loglik(g_obs,
                    e_log_pop_freq, e_log_1m_pop_freq, \
                    e_log_sticks, e_log_1m_sticks,
                    detach_ez): 
    
    # outer loop through n
    
    e_log_cluster_probs = \
        modeling_lib.get_e_log_cluster_probabilities_from_e_log_stick(
                            e_log_sticks, e_log_1m_sticks)

    body_fun = lambda val, x : get_e_loglik_n(x[0], 
                                            e_log_pop_freq, e_log_1m_pop_freq,
                                            x[1], detach_ez) + \
                                            val

    scan_fun = lambda val, x : (body_fun(val, x), None)

    init_val = np.array([0., 0.])
    out = jax.lax.scan(scan_fun, init_val,
                        xs = (g_obs,
                              e_log_cluster_probs))[0]

    e_loglik = out[0]
    z_entropy = out[1]

    return e_loglik, z_entropy


def get_e_joint_loglik_from_nat_params(g_obs,
                                    e_log_pop_freq, e_log_1m_pop_freq,
                                    e_log_sticks, e_log_1m_sticks,
                                    dp_prior_alpha, allele_prior_alpha,
                                    allele_prior_beta,
                                    detach_ez = False):

    e_loglik, z_entropy = get_e_loglik(g_obs,
                                        e_log_pop_freq, e_log_1m_pop_freq, \
                                        e_log_sticks, e_log_1m_sticks,
                                        detach_ez = detach_ez)

    # prior term
    e_log_prior = get_e_log_prior(e_log_1m_sticks,
                            e_log_pop_freq, e_log_1m_pop_freq,
                            dp_prior_alpha, allele_prior_alpha,
                            allele_prior_beta).squeeze()

    return e_log_prior + e_loglik, z_entropy


def get_kl(g_obs, vb_params_dict, prior_params_dict,
                    gh_loc = None, gh_weights = None,
                    log_phi = None,
                    epsilon = 1.,
                    detach_ez = False):

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
    use_logitnormal_sticks : boolean
        Whether to use a logitnormal approximation to infer the sticks.
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
    obs_weights: ndarray
        weights for the individual observations
    loci_weights: ndarray
        weights for the loci

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
            get_moments_from_vb_params_dict(vb_params_dict,
                                    gh_loc = gh_loc,
                                    gh_weights = gh_weights)
    # joint log likelihood
    e_loglik, z_entropy = get_e_joint_loglik_from_nat_params(g_obs,
                                    e_log_pop_freq, e_log_1m_pop_freq,
                                    e_log_sticks, e_log_1m_sticks,
                                    dp_prior_alpha, allele_prior_alpha,
                                    allele_prior_beta,
                                    detach_ez = detach_ez)

    # entropy term
    entropy = get_entropy(vb_params_dict, gh_loc, gh_weights) + z_entropy

    elbo = e_loglik + entropy

    # prior perturbation
    if log_phi is not None:

        assert gh_loc is not None
        assert gh_weights is not None

        assert 'ind_mix_stick_propn_info' in vb_params_dict.keys()
        assert 'ind_mix_stick_propn_mean' in vb_params_dict.keys()

        e_log_pert = func_sens_lib.get_e_log_perturbation(log_phi,
                                vb_params_dict['ind_mix_stick_propn_mean'],
                                vb_params_dict['ind_mix_stick_propn_info'],
                                epsilon, gh_loc, gh_weights, sum_vector=True)
        elbo = elbo - e_log_pert

    return -1 * elbo

def get_moments_from_vb_params_dict(vb_params_dict,
                                    gh_loc = None,
                                    gh_weights = None):

    use_logitnormal_sticks = 'ind_mix_stick_propn_mean' in vb_params_dict.keys()
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
def cluster_and_get_init(g_obs, k, seed):
    # g_obs should be n_obs x n_loci x 3,
    # a one-hot encoding of genotypes
    assert len(g_obs.shape) == 3

    # convert one-hot encoding to probability of A genotype, {0, 0.5, 1}
    x = g_obs.argmax(axis = 2) / 2

    # run NMF
    model = NMF(n_components=k, init='random', random_state = seed)
    init_ind_admix_propn_unscaled = model.fit_transform(onp.array(x))
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

    return np.array(init_ind_admix_propn), \
            np.array(init_pop_allele_freq.clip(0.05, 0.95))

def set_init_vb_params(g_obs, k_approx, vb_params_dict,
                        seed):
    # get initial admixtures, and population frequencies
    init_ind_admix_propn, init_pop_allele_freq = \
            cluster_and_get_init(g_obs, k_approx, seed = seed)

    # set bnp parameters for individual admixture
    # set mean to be logit(stick_breaking_propn), info to be 1
    stick_break_propn = \
        cluster_quantities_lib.get_stick_break_propns_from_mixture_weights(init_ind_admix_propn)

    use_logitnormal_sticks = 'ind_mix_stick_propn_mean' in vb_params_dict.keys()
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
