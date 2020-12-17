import jax
import jax.numpy as np
import jax.scipy as sp

from jax.experimental import loops

import paragami

import bnpmodeling_runjingdev.functional_sensitivity_lib as func_sens_lib

import bnpmodeling_runjingdev.exponential_families as ef
from bnpmodeling_runjingdev import cluster_quantities_lib, modeling_lib

from numpy.polynomial.hermite import hermgauss

import warnings

##########################
# Set up vb parameters
##########################

def get_vb_params_paragami_object(n_obs, n_loci, k_approx,
                                    use_logitnormal_sticks, 
                                    seed = 0):
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
                                            lb = 0.0)
    else:
        # else they are beta distributed
        ind_admix_params_paragami['stick_beta'] = \
            paragami.NumericArrayPattern(shape=(n_obs, k_approx - 1, 2),
                                            lb = 0.0)
    vb_params_paragami['ind_admix_params'] = ind_admix_params_paragami
    
    vb_params_dict = vb_params_paragami.random(key = jax.random.PRNGKey(0))

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
def get_loglik_gene_nlk(g_obs, e_log_pop_freq, e_log_1m_pop_freq):

    genom_loglik_nlk_a = \
        np.einsum('nl, lk -> nlk', g_obs[:, :, 0], e_log_1m_pop_freq) + \
            np.einsum('nl, lk -> nlk', g_obs[:, :, 1] + \
                                        g_obs[:, :, 2], e_log_pop_freq)

    genom_loglik_nlk_b = \
        np.einsum('nl, lk -> nlk', g_obs[:, :, 0] + \
                                    g_obs[:, :, 1], e_log_1m_pop_freq) + \
            np.einsum('nl, lk -> nlk', g_obs[:, :, 2], e_log_pop_freq)
    
    # this is n_obs x n_loci x k_aprox x 2
    return np.stack((genom_loglik_nlk_a, genom_loglik_nlk_b), axis = -1)

def get_loglik_cond_z(g_obs, e_log_pop_freq, e_log_1m_pop_freq,
                        e_log_cluster_probs):

    # get likelihood of genes
    loglik_gene_nlk = get_loglik_gene_nlk(g_obs, e_log_pop_freq, \
                                            e_log_1m_pop_freq)

    # log likelihood of population belongings
    n = e_log_cluster_probs.shape[0]
    k = e_log_cluster_probs.shape[1]

    # loglik_obs_by_nlk2 is n_obs x n_loci x k_approx x 2
    loglik_cond_z = loglik_gene_nlk + e_log_cluster_probs.reshape(n, 1, k, 1)

    return loglik_cond_z

def get_z_opt_from_loglik_cond_z(loglik_cond_z):
    # 2nd axis dimension is k
    # recall that loglik_obs_by_nlk2 is n_obs x n_loci x k_approx x 2

    return jax.nn.softmax(loglik_cond_z, axis = 2)


def get_e_joint_loglik_from_nat_params(g_obs,
                                       e_log_pop_freq, e_log_1m_pop_freq,
                                       e_log_sticks, e_log_1m_sticks,
                                       dp_prior_alpha, allele_prior_alpha,
                                       allele_prior_beta,
                                       ez = None):
    
    e_log_cluster_probs = \
        modeling_lib.get_e_log_cluster_probabilities_from_e_log_stick(
                            e_log_sticks, e_log_1m_sticks)
    
    loglik_cond_z = get_loglik_cond_z(g_obs, 
                                      e_log_pop_freq,
                                      e_log_1m_pop_freq,
                                      e_log_cluster_probs)
    if ez is None: 
        ez = get_z_opt_from_loglik_cond_z(loglik_cond_z)
    
    e_loglik = (loglik_cond_z * ez).sum()
    
    # prior term
    e_log_prior = get_e_log_prior(e_log_1m_sticks,
                            e_log_pop_freq, e_log_1m_pop_freq,
                            dp_prior_alpha, allele_prior_alpha,
                            allele_prior_beta).squeeze()
        
    return e_log_prior + e_loglik, ez


def get_kl(g_obs, 
           vb_params_dict, 
           prior_params_dict,
           gh_loc = None, 
           gh_weights = None,
           ez = None,
           e_log_phi = None):

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
        Locations for gauss-hermite quadrature.
        Required if sticks are modeled using a logit-normal.
    gh_weights : vector
        Weights for gauss-hermite quadrature. 
        Required if sticks are modeled using a logit-normal.
    e_log_phi : callable
        Function with arguments stick_means and stick_infos 
        and returns the expected log-multiplicative perturbation.
        If not None, sticks must be modeled using a logit-normal.
    detach_ez : boolean
        Does not affect the KL but will affect derivatives.
        If True, derivatives are computed with 
        the expected cluster belongings
        (in unconstrained space) fixed. 
        Otherwise, the resulting derivative will be 
        the total derivative (differentiating through 
        the optimality of the cluster belongings). 
        
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
    e_loglik, ez = get_e_joint_loglik_from_nat_params(g_obs,
                                    e_log_pop_freq, e_log_1m_pop_freq,
                                    e_log_sticks, e_log_1m_sticks,
                                    dp_prior_alpha, allele_prior_alpha,
                                    allele_prior_beta,
                                    ez = ez)

    # entropy term
    entropy = get_entropy(vb_params_dict, gh_loc, gh_weights) + \
                sp.special.entr(ez).sum()
    
    
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

def get_e_num_clusters(g_obs, vb_params_dict, gh_loc, gh_weights, 
                        threshold = 0,
                        n_samples = 1000,
                        seed = 2342): 
    
    e_log_sticks, e_log_1m_sticks, \
        e_log_pop_freq, e_log_1m_pop_freq = \
            get_moments_from_vb_params_dict(vb_params_dict,
                                    gh_loc = gh_loc,
                                    gh_weights = gh_weights)
    e_log_cluster_probs = \
        modeling_lib.get_e_log_cluster_probabilities_from_e_log_stick(
                            e_log_sticks, e_log_1m_sticks)
    
    n_obs = g_obs.shape[0]
    k_approx = e_log_cluster_probs.shape[-1]
        
    with loops.Scope() as s:
        s.sampled_counts = np.zeros((n_samples, k_approx))
        
        for l in s.range(g_obs.shape[1]):
            # e_z_l is shaped as n_obs x k_approx x 2
            _, e_z_l = get_optimal_ezl(g_obs[:, l],
                                    e_log_pop_freq[l], e_log_1m_pop_freq[l],
                                    e_log_cluster_probs)
            
            # combine first and last dimension
            e_z_l = e_z_l.transpose((0, 2, 1)).\
                            reshape((n_obs * 2, k_approx))

            # sample counts
            unif_samples = jax.random.uniform(key = jax.random.PRNGKey(seed + l), 
                                              shape = (n_samples, e_z_l.shape[0]))
            
            # this is n_samples x (n_obs * 2) x k_approx
            z_samples = cluster_quantities_lib.\
                            _get_onehot_clusters_from_ez_and_unif_samples(e_z_l, 
                                                                          unif_samples)
            
            # this is n_samples x k_approx
            sampled_counts_l = z_samples.sum(1)
            
            s.sampled_counts += sampled_counts_l
    
    n_clusters_sampled = (s.sampled_counts > threshold).sum(1)
    
    # return s.sampled_counts
    # return n_clusters_sampled
    return n_clusters_sampled.mean()

def get_e_num_pred_clusters(stick_means, stick_infos, gh_loc, gh_weights, 
                            n_samples = 1000,
                            key = jax.random.PRNGKey(0)): 
    
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
    e_num_clusters_sampled = cluster_quantities_lib.\
            get_e_num_clusters_from_ez_2d(ind_admix_sampled)
    
    return e_num_clusters_sampled.mean()


#####################
# Functions to save / load a structure fit
#####################
def save_structure_fit(outfile, vb_params_dict, vb_params_paragami, 
                       prior_params_dict, gh_deg, **kwargs): 
    
    paragami.save_folded(outfile,
                         vb_params_dict,
                         vb_params_paragami,
                         dp_prior_alpha = prior_params_dict['dp_prior_alpha'],
                         allele_prior_alpha = prior_params_dict['allele_prior_alpha'],
                         allele_prior_beta = prior_params_dict['allele_prior_beta'],
                         gh_deg = gh_deg,
                         **kwargs)

def load_structure_fit(fit_file): 
    
    # load vb params dict and other meta data
    vb_params_dict, vb_params_paragami, meta_data = \
        paragami.load_folded(fit_file)
    
    # gauss-hermite parameters
    gh_deg = int(meta_data['gh_deg'])
    gh_loc, gh_weights = hermgauss(gh_deg)

    gh_loc = np.array(gh_loc)
    gh_weights = np.array(gh_weights)
    
    # load prior parameters
    prior_params_dict, prior_params_paragami = \
        get_default_prior_params()

    prior_params_dict['dp_prior_alpha'] = np.array(meta_data['dp_prior_alpha'])
    prior_params_dict['allele_prior_alpha'] = np.array(meta_data['allele_prior_alpha'])
    prior_params_dict['allele_prior_beta'] = np.array(meta_data['allele_prior_beta'])

    return vb_params_dict, vb_params_paragami, \
            prior_params_dict, prior_params_paragami, \
                gh_loc, gh_weights, meta_data

#####################
# hyper-parameter objective functions: 
# NOTE these are **added** to the **KL**
#####################
def alpha_objective_fun(vb_params_dict, alpha, gh_loc, gh_weights): 
    
    # term of objective function that depends on 
    # the dp prior alpha
    
    means = vb_params_dict['ind_admix_params']['stick_means']
    infos = vb_params_dict['ind_admix_params']['stick_infos']

    e_log_1m_sticks = \
        ef.get_e_log_logitnormal(
            lognorm_means = means,
            lognorm_infos = infos,
            gh_loc = gh_loc,
            gh_weights = gh_weights)[1]

    return - (alpha - 1) * np.sum(e_log_1m_sticks)

