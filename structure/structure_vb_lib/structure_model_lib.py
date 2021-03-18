import jax
import jax.numpy as np
import jax.scipy as sp

import paragami

from bnpmodeling_runjingdev import modeling_lib
import bnpmodeling_runjingdev.exponential_families as ef

from numpy.polynomial.hermite import hermgauss

##########################
# Set up vb parameters
##########################

def get_vb_params_paragami_object(n_obs, n_loci, n_allele, k_approx,
                                  prng_key = jax.random.PRNGKey(0)):
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

    Returns
    -------
    vb_params_dict : dictionary
        A dictionary that contains the variational parameters.

    vb_params_paragami : paragami patterned dictionary
        A paragami patterned dictionary that contains the variational parameters.

    """

    vb_params_paragami = paragami.PatternDict()

    # variational beta parameters for population allele frequencies
    vb_params_paragami['pop_freq_dirichlet_params'] = \
        paragami.NumericArrayPattern(shape=(k_approx, n_loci, n_allele), 
                                     lb = 0.0)
    
    # population-breaking stick parameters
    pop_stick_params_paragami = paragami.PatternDict()
    pop_stick_params_paragami['stick_means'] = \
        paragami.NumericArrayPattern(shape = (k_approx - 1,))
    pop_stick_params_paragami['stick_infos'] = \
        paragami.NumericArrayPattern(shape = (k_approx - 1,),
                                        lb = 0.0)
    vb_params_paragami['pop_stick_params'] = pop_stick_params_paragami
    
    # variational multinomial parameters for the individual-level 
    # population indices
    vb_params_paragami['pop_indx_multinom_params'] = \
        paragami.SimplexArrayPattern(simplex_size = k_approx, 
                                     array_shape = (n_obs, k_approx))

    # individual stick-breaking
    ind_admix_params_paragami = paragami.PatternDict()
    
    # variational distribution for each stick is logitnormal
    ind_admix_params_paragami['stick_means'] = \
        paragami.NumericArrayPattern(shape = (n_obs, k_approx - 1,))
    ind_admix_params_paragami['stick_infos'] = \
        paragami.NumericArrayPattern(shape = (n_obs, k_approx - 1,),
                                        lb = 0.0)
    vb_params_paragami['ind_admix_params'] = ind_admix_params_paragami
    
    vb_params_dict = vb_params_paragami.random(key = prng_key)

    return vb_params_dict, vb_params_paragami


##########################
# Set up prior parameters
##########################
def get_default_prior_params(n_allele):
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

    # DP prior parameter population-breaking proportions
    prior_params_dict['dp_prior_alpha'] = np.array([6.0])
    prior_params_paragami['dp_prior_alpha'] = \
        paragami.NumericArrayPattern(shape=(1, ), lb = 0.0)
    
    # DP prior parameter on individual-breaking proportions
    prior_params_dict['ind_dp_prior_alpha'] = np.array([6.0])
    prior_params_paragami['ind_dp_prior_alpha'] = \
        paragami.NumericArrayPattern(shape=(1, ), lb = 0.0)

    # prior on the allele frequencies
    # beta distribution parameters
    prior_params_dict['allele_prior_lambda_vec'] = np.ones(n_allele)
    prior_params_paragami['allele_prior_lambda_vec'] = \
        paragami.NumericArrayPattern(shape=(n_allele, ), lb = 0.0)

    return prior_params_dict, prior_params_paragami

##########################
# Expected prior term
##########################
def get_e_log_prior(vb_params_dict, moments_dict, prior_params_dict):
    
    # get expected prior term

    # dp prior on stick-breaking
    ind_mix_dp_prior =  (prior_params_dict['ind_dp_prior_alpha'] - 1) * \
                            np.sum(moments_dict['e_log_1m_ind_sticks'])
    
    pop_mix_dp_prior =  (prior_params_dict['dp_prior_alpha'] - 1) * \
                            np.sum(moments_dict['e_log_1m_pop_sticks'])

    # allele frequency prior
    # sum over k_approx and n_loci
    # this is len(n_allele)
    _e_log_pop_freq_allele = moments_dict['e_log_pop_freq'].sum(0).sum(0)
    allele_freq_beta_prior = ((prior_params_dict['allele_prior_lambda_vec'] - 1) * _e_log_pop_freq_allele).sum()
    
    # prior on individual level multinomial
    ind_multi_prior = (vb_params_dict['pop_indx_multinom_params'].sum(0).sum(0) * \
                       moments_dict['e_log_pop_cluster_probs']).sum()
    
    return ind_mix_dp_prior + pop_mix_dp_prior + allele_freq_beta_prior + ind_multi_prior

##########################
# Entropy
##########################
def get_entropy(vb_params_dict, e_z, gh_loc, gh_weights):
    
    # entropy of stick-breaking distributions
    ind_stick_entropy = \
        modeling_lib.get_stick_breaking_entropy(
                                vb_params_dict['ind_admix_params']['stick_means'],
                                vb_params_dict['ind_admix_params']['stick_infos'],
                                gh_loc, gh_weights)
    
    pop_stick_entropy = \
        modeling_lib.get_stick_breaking_entropy(
                                vb_params_dict['pop_stick_params']['stick_means'],
                                vb_params_dict['pop_stick_params']['stick_infos'],
                                gh_loc, gh_weights)
    
    # dirichlet entropy term
    pop_freq_dir_params = vb_params_dict['pop_freq_dirichlet_params']
    dir_entropy = ef.dirichlet_entropy(pop_freq_dir_params.transpose((2, 0, 1))).sum()
    
    # z entropy 
    z_entropy = modeling_lib.multinom_entropy(e_z)
    
    # individual level multinomial entropy 
    ind_multinom_entropy = modeling_lib.multinom_entropy(vb_params_dict['pop_indx_multinom_params'])
    
    return ind_stick_entropy + pop_stick_entropy + dir_entropy + z_entropy + ind_multinom_entropy

##########################
# Likelihood term
##########################
def get_e_loglik_gene_nlk(g_obs, e_log_pop_freq, pop_indx_multinom_params):
    
    # e_log_pop_freq is k_approx x n_loci x n_allele
    # pop_indx_multinom_params is n_obs x k_approx x k_approx, 
    # where the last dimension sums to one
    
#     n_obs = pop_indx_multinom_params.shape[0]
#     k_approx = pop_indx_multinom_params.shape[-1]
    
#     print('setting cs')
#     pop_indx_multinom_params = np.stack((np.eye(k_approx), ) * n_obs)
    
    # this is n_obs x k_approx x n_loci x n_allele
    e_log_pop_freq_innerprod = np.einsum('ijk, klm -> ijlm',
                                         pop_indx_multinom_params,
                                         e_log_pop_freq)
        
    # this is n_obs x n_loci x n_allele x k_approx
    e_log_pop_freq_t = e_log_pop_freq_innerprod.transpose((0, 2, 3, 1))
    
    # this is n_obs x n_loci x 1 x n_allele x k_approx
    e_log_pop_freq_t = np.expand_dims(e_log_pop_freq_t, 2)
    
    # this is n_obs x n_loci x 2 x n_allele x k_approx
    outer_prod = np.expand_dims(g_obs, -1) * e_log_pop_freq_t
    
    # sum over n_allele
    # return something that is n_obs x n_loci x 2 x k_approx
    return outer_prod.sum(3)

#     # g_obs is n_obs x n_loci x 2 x n_allele
#     return np.einsum('nlij, nklj -> nlik', 
#                      g_obs, 
#                      e_log_pop_freq_innerprod)
    

def get_z_nat_params(g_obs, e_log_pop_freq, pop_indx_multinom_params, e_log_ind_cluster_probs): 
    
    # this is n_obs x n_loci x 2 x k_approx
    loglik_gene_nlk = get_e_loglik_gene_nlk(g_obs, 
                                            e_log_pop_freq,
                                            pop_indx_multinom_params)

    # make this this is n_obs x 1 x 1 x k_approx
    _e_log_ind_cluster_probs = np.expand_dims(e_log_ind_cluster_probs, axis = 1)
    _e_log_ind_cluster_probs = np.expand_dims(_e_log_ind_cluster_probs, axis = 1)
                                            
    # add individual belongings
    return _e_log_ind_cluster_probs + loglik_gene_nlk

def get_optimal_z(g_obs, e_log_pop_freq, pop_indx_multinom_params, e_log_ind_cluster_probs):

    z_nat_param = \
        get_z_nat_params(g_obs, e_log_pop_freq, pop_indx_multinom_params, e_log_ind_cluster_probs)

    e_z = jax.nn.softmax(z_nat_param, -1)

    return e_z, z_nat_param


def get_kl(g_obs, 
           vb_params_dict, 
           prior_params_dict,
           gh_loc, 
           gh_weights,
           e_log_phi = None,
           e_z = None):

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
        
    Returns
    -------
    kl : float
        The negative elbo.
    """

    # get moments
    moments_dict = get_moments_from_vb_params_dict(vb_params_dict,
                                                   gh_loc = gh_loc,
                                                   gh_weights = gh_weights)
                                            
    
    # joint log likelihood
    e_z_opt, z_nat_param = \
        get_optimal_z(g_obs, 
                      moments_dict['e_log_pop_freq'], 
                      vb_params_dict['pop_indx_multinom_params'], 
                      moments_dict['e_log_ind_cluster_probs'])
    
    if e_z is None:
        e_z = e_z_opt
    
    e_loglik = np.sum(e_z * z_nat_param)
    
    # entropy term
    entropy = get_entropy(vb_params_dict, e_z, gh_loc, gh_weights) 
    
    # prior term 
    e_log_prior = get_e_log_prior(vb_params_dict, 
                                  moments_dict,
                                  prior_params_dict)
    
    elbo = e_loglik + entropy + e_log_prior

    # prior perturbation
    if e_log_phi is not None:

        e_log_pert = e_log_phi(vb_params_dict['ind_admix_params']['stick_means'],
                               vb_params_dict['ind_admix_params']['stick_infos'])
                                                            
        elbo = elbo + e_log_pert
        
    return -1 * elbo

######################
# a useful function to get posterior moments
######################

def _get_stick_moments_from_dict(stick_params_dict, gh_loc, gh_weights): 
    stick_means = stick_params_dict['stick_means']
    stick_infos = stick_params_dict['stick_infos']
    
    e_log_sticks, e_log_1m_sticks = \
        ef.get_e_log_logitnormal(
            lognorm_means = stick_means,
            lognorm_infos = stick_infos,
            gh_loc = gh_loc,
            gh_weights = gh_weights)
    
    e_log_cluster_probs = \
        modeling_lib.get_e_log_cluster_probabilities_from_e_log_stick(
                            e_log_sticks, e_log_1m_sticks)
    
    return e_log_sticks, e_log_1m_sticks, e_log_cluster_probs
                                            

def get_moments_from_vb_params_dict(vb_params_dict,
                                    gh_loc,
                                    gh_weights):
    
    # expected log individual stick lengths
    e_log_ind_sticks, e_log_1m_ind_sticks, e_log_ind_cluster_probs = \
        _get_stick_moments_from_dict(vb_params_dict['ind_admix_params'], 
                                     gh_loc, 
                                     gh_weights)
                                            
    # population dirichlet moments
    pop_freq_dir_params = vb_params_dict['pop_freq_dirichlet_params']
    e_log_pop_freq = \
        modeling_lib.get_e_log_dirichlet(pop_freq_dir_params)
    
    # population stick-breaking moments
    e_log_pop_sticks, e_log_1m_pop_sticks, e_log_pop_cluster_probs = \
        _get_stick_moments_from_dict(vb_params_dict['pop_stick_params'], 
                                     gh_loc, 
                                     gh_weights)
    
    moments_dict = dict({'e_log_ind_sticks' : e_log_ind_sticks, 
                         'e_log_1m_ind_sticks' : e_log_1m_ind_sticks,
                         'e_log_ind_cluster_probs' : e_log_ind_cluster_probs, 
                         'e_log_pop_sticks' : e_log_pop_sticks, 
                         'e_log_1m_pop_sticks' : e_log_1m_pop_sticks,
                         'e_log_pop_cluster_probs' : e_log_pop_cluster_probs, 
                         'e_log_pop_freq': e_log_pop_freq})    
    
    return moments_dict


#####################
# Functions to save a structure fit
#####################
def save_structure_fit(outfile, vb_params_dict, vb_params_paragami, 
                       prior_params_dict, gh_deg, **kwargs): 
    
    paragami.save_folded(outfile,
                         vb_params_dict,
                         vb_params_paragami,
                         dp_prior_alpha = prior_params_dict['dp_prior_alpha'],
                         allele_prior_lambda_vec = prior_params_dict['allele_prior_lambda_vec'],
                         gh_deg = gh_deg,
                         **kwargs)


def load_structure_fit(fit_file): 
    
    # load vb params dict and other meta data
    vb_params_dict, vb_params_paragami, meta_data = \
        paragami.load_folded(fit_file)
    
    n_alleles = vb_params_dict['pop_freq_dirichlet_params'].shape[-1]
    
    # gauss-hermite parameters
    gh_deg = int(meta_data['gh_deg'])
    gh_loc, gh_weights = hermgauss(gh_deg)

    gh_loc = np.array(gh_loc)
    gh_weights = np.array(gh_weights)
    
    # load prior parameters
    prior_params_dict, prior_params_paragami = \
        get_default_prior_params(n_alleles)

    prior_params_dict['dp_prior_alpha'] = np.array(meta_data['dp_prior_alpha'])
    prior_params_dict['allele_prior_lambda_vec'] = np.array(meta_data['allele_prior_lambda_vec'])

    return vb_params_dict, vb_params_paragami, \
            prior_params_dict, prior_params_paragami, \
                gh_loc, gh_weights, meta_data
