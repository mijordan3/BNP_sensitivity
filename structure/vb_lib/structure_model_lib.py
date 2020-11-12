import jax
import jax.numpy as np
import jax.scipy as sp

from jax.experimental import loops

import paragami

import bnpmodeling_runjingdev.functional_sensitivity_lib as func_sens_lib

import bnpmodeling_runjingdev.exponential_families as ef
from bnpmodeling_runjingdev import cluster_quantities_lib, modeling_lib

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
        paragami.NumericArrayPattern(shape=(n_loci, k_approx, 2), 
                                     lb = 0.0)

    # BNP sticks
    ind_admix_params_paragami = paragami.PatternDict()
    if use_logitnormal_sticks:
        # variational distribution for each stick is logitnormal
        ind_admix_params_paragami['stick_means'] = \
            paragami.NumericArrayPattern(shape = (n_obs, k_approx - 1,))
        ind_admix_params_paragami['stick_infos'] = \
            paragami.NumericArrayPattern(shape = (n_obs, k_approx - 1,),
                                            lb = 1e-4)
    else:
        # else they are beta distributed
        ind_admix_params_paragami['stick_beta'] = \
            paragami.NumericArrayPattern(shape=(n_obs, k_approx - 1, 2),
                                            lb = 0.0)
    vb_params_paragami['ind_admix_params'] = ind_admix_params_paragami
    
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
    use_logitnormal_sticks = 'stick_means' in vb_params_dict['ind_admix_params'].keys()
    if use_logitnormal_sticks:
        stick_entropy = \
            modeling_lib.get_stick_breaking_entropy(
                                    vb_params_dict['ind_admix_params']['stick_means'],
                                    vb_params_dict['ind_admix_params']['stick_infos'],
                                    gh_loc, gh_weights)
    else:
        ind_mix_stick_beta_params = vb_params_dict['ind_admix_params']['stick_beta']
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
def get_e_loglik_gene_nk(g_obs_l, e_log_pop_freq_l, e_log_1m_pop_freq_l):

    g_obs_l0 = g_obs_l[:, 0]
    g_obs_l1 = g_obs_l[:, 1]
    g_obs_l2 = g_obs_l[:, 2]

    loglik_a = \
        np.outer(g_obs_l0, e_log_1m_pop_freq_l) + \
            np.outer(g_obs_l1 + g_obs_l2, e_log_pop_freq_l)

    loglik_b = \
        np.outer(g_obs_l0 + g_obs_l1, e_log_1m_pop_freq_l) + \
            np.outer(g_obs_l2, e_log_pop_freq_l)


    return np.stack((loglik_a, loglik_b), axis = -1)

def get_optimal_ezl(g_obs_l, e_log_pop_freq_l, e_log_1m_pop_freq_l,
                    e_log_cluster_probs): 
    
    # get loglikelihood of observations at loci l
    loglik_gene_l = get_e_loglik_gene_nk(g_obs_l, e_log_pop_freq_l, e_log_1m_pop_freq_l)

    # add individual belongings
    loglik_cond_z_l = np.expand_dims(e_log_cluster_probs, axis = 2) + loglik_gene_l

    # individal x chromosome belongings
    e_z_l = jax.nn.softmax(loglik_cond_z_l, axis = 1)
    
    return loglik_cond_z_l, e_z_l
    
def get_e_loglik_l(g_obs_l, e_log_pop_freq_l, e_log_1m_pop_freq_l,
                    e_log_cluster_probs, detach_ez):
    # returns z-optimized log-likelihood for locus-l
    
    loglik_cond_z_l, e_z_l = \
        get_optimal_ezl(g_obs_l, e_log_pop_freq_l, e_log_1m_pop_freq_l,
                    e_log_cluster_probs)
    
    if detach_ez:
        e_z_l = jax.lax.stop_gradient(e_z_l)

    # log likelihood
    loglik_l = np.sum(loglik_cond_z_l * e_z_l)

    # entropy term: save this because the z's won't be available later
    # compute the entropy
    z_entropy_l = (sp.special.entr(e_z_l)).sum()

    return loglik_l, z_entropy_l

def get_e_loglik(g_obs, e_log_pop_freq, e_log_1m_pop_freq, \
                    e_log_sticks, e_log_1m_sticks,
                    detach_ez):


    e_log_cluster_probs = \
        modeling_lib.get_e_log_cluster_probabilities_from_e_log_stick(
                            e_log_sticks, e_log_1m_sticks)

    with loops.Scope() as s:
        s.e_loglik = 0.
        s.z_entropy = 0.
        for l in s.range(g_obs.shape[1]):
            e_loglik_l, z_entropy_l = get_e_loglik_l(g_obs[:, l],
                                    e_log_pop_freq[l], e_log_1m_pop_freq[l],
                                    e_log_cluster_probs, detach_ez)

            s.e_loglik += e_loglik_l
            s.z_entropy += z_entropy_l

    return s.e_loglik, s.z_entropy


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
                    e_log_phi = None,
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
    if e_log_phi is not None:

        e_log_pert = e_log_phi(vb_params_dict['ind_admix_params']['stick_means'],
                               vb_params_dict['ind_admix_params']['stick_infos'])
                                                            
        elbo = elbo + e_log_pert
        
    return -1 * elbo


######################
# Some helpful functions to get posterior moments
######################

def get_moments_from_vb_params_dict(vb_params_dict,
                                    gh_loc = None,
                                    gh_weights = None):

    use_logitnormal_sticks = 'stick_means' in vb_params_dict['ind_admix_params'].keys()
    # get expected sticks
    if use_logitnormal_sticks:
        assert gh_loc is not None
        assert gh_weights is not None

        ind_mix_stick_propn_mean = vb_params_dict['ind_admix_params']['stick_means']
        ind_mix_stick_propn_info = vb_params_dict['ind_admix_params']['stick_infos']

        e_log_sticks, e_log_1m_sticks = \
            ef.get_e_log_logitnormal(
                lognorm_means = ind_mix_stick_propn_mean,
                lognorm_infos = ind_mix_stick_propn_info,
                gh_loc = gh_loc,
                gh_weights = gh_weights)
    else:
        ind_mix_stick_beta_params = vb_params_dict['ind_admix_params']['stick_beta']
        e_log_sticks, e_log_1m_sticks = \
            modeling_lib.get_e_log_beta(ind_mix_stick_beta_params)

    # population beta parameters
    pop_freq_beta_params = vb_params_dict['pop_freq_beta_params']
    e_log_pop_freq, e_log_1m_pop_freq = \
        modeling_lib.get_e_log_beta(pop_freq_beta_params)

    return e_log_sticks, e_log_1m_sticks, \
                e_log_pop_freq, e_log_1m_pop_freq

def get_e_num_pred_clusters(stick_means, stick_infos, gh_loc, gh_weights, 
                    key, n_samples = 10000): 
    
    # If I sample one more loci for every individual in my dataset, 
    # how many clusters would I expect to see?
    
    # sample standard normal
    shape = (n_samples, ) + stick_means.shape 
    normal_samples = jax.random.normal(key, shape)
    
    # sample sticks: shape is n_samples x n_obs x k_approx 
    sds = np.expand_dims((1 / np.sqrt(stick_infos)), axis = 0)
    means = np.expand_dims(stick_means, axis = 0)
    sticks_sampled = sp.special.expit(normal_samples * sds + means)
    
    # get mixture weights
    ind_admix_sampled = \
        cluster_quantities_lib.\
            get_mixture_weights_from_stick_break_propns(sticks_sampled)
        
    # get expected number of clusters 
    # loop over n_samples 
    e_num_clusters_sampled = \
        jax.lax.map(cluster_quantities_lib.get_e_num_clusters_from_ez,
                    ind_admix_sampled)
    
    return e_num_clusters_sampled.mean()