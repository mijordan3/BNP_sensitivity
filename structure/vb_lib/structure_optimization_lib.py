import jax
import jax.numpy as np
import jax.scipy as sp

import numpy as onp
from sklearn.decomposition import NMF

import paragami

from vb_lib import structure_model_lib
from vb_lib.cavi_lib import update_pop_beta

import bnpmodeling_runjingdev.exponential_families as ef
from bnpmodeling_runjingdev import cluster_quantities_lib, modeling_lib

from bnpmodeling_runjingdev.optimization_lib import OptimizationObjectiveJaxtoNumpy

###############
# functions for initializing
###############
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

def set_nmf_init_vb_params(g_obs, k_approx, vb_params_dict, seed):
    # set the vb parameters at the NMF results 
    
    # get initial admixtures, and population frequencies
    init_ind_admix_propn, init_pop_allele_freq = \
            cluster_and_get_init(g_obs, k_approx, seed = seed)

    # set bnp parameters for individual admixture
    # set mean to be logit(stick_breaking_propn), info to be 1
    stick_break_propn = \
        cluster_quantities_lib.get_stick_break_propns_from_mixture_weights(init_ind_admix_propn)

    use_logitnormal_sticks = 'stick_means' in vb_params_dict['ind_admix_params'].keys()
    if use_logitnormal_sticks:
        ind_mix_stick_propn_mean = np.log(stick_break_propn) - np.log(1 - stick_break_propn)
        ind_mix_stick_propn_info = np.ones(stick_break_propn.shape)
        vb_params_dict['ind_admix_params']['stick_means'] = ind_mix_stick_propn_mean
        vb_params_dict['ind_admix_params']['stick_infos'] = ind_mix_stick_propn_info
    else:
        ind_mix_stick_beta_param1 = np.ones(stick_break_propn.shape) 
        ind_mix_stick_beta_param2 = (1 - stick_break_propn) / stick_break_propn
        vb_params_dict['ind_admix_params']['stick_beta'] = \
            np.concatenate((ind_mix_stick_beta_param1[:, :, None],
                            ind_mix_stick_beta_param2[:, :, None]), axis = 2)

    # set beta paramters for population paramters
    # set beta = 1, alpha to have the correct mean
    beta0 = 1
    pop_freq_beta_params1 = beta0 * init_pop_allele_freq / (1 - init_pop_allele_freq)
    pop_freq_beta_params2 = np.ones(init_pop_allele_freq.shape) * beta0
    pop_freq_beta_params = np.concatenate((pop_freq_beta_params1[:, :, None],
                                       pop_freq_beta_params2[:, :, None]), axis = 2)

    vb_params_dict['pop_freq_beta_params'] = pop_freq_beta_params

    return vb_params_dict

def set_init_vb_params(g_obs, k_approx, vb_params_dict, 
                       prior_params_dict,
                       gh_loc = None, gh_weights = None,
                       seed = 1,
                       n_iter = 5): 
    
    # get nmf init
    vb_params_dict = \
        set_nmf_init_vb_params(g_obs, k_approx, vb_params_dict, seed)
    
    # update population frequency betas 
    # and implicity the ezs. 
    # these updates are closed form, and relatively fast
    
    # get initial moments from vb_params
    e_log_sticks, e_log_1m_sticks, \
        e_log_pop_freq, e_log_1m_pop_freq = \
            structure_model_lib.get_moments_from_vb_params_dict(
                vb_params_dict, gh_loc, gh_weights)

    e_log_cluster_probs = \
        modeling_lib.get_e_log_cluster_probabilities_from_e_log_stick(
            e_log_sticks, e_log_1m_sticks)
    
    # a few cavi updates
    for i in range(n_iter): 
        vb_params_dict['pop_freq_beta_params'],\
            e_log_pop_freq, e_log_1m_pop_freq = \
                update_pop_beta(g_obs,
                                e_log_pop_freq, e_log_1m_pop_freq,
                                e_log_cluster_probs,
                                prior_params_dict)
    return vb_params_dict

    
    

def get_ind_admix_params_psloss(g_obs, ind_admix_params, 
                                e_log_pop_freq, e_log_1m_pop_freq, 
                                prior_params_dict, 
                                gh_loc, gh_weights,
                                detach_ez = True): 

    # returns the terms of the KL that depend on the 
    # individual admixture parameters
    # Hence the KL is not correct, but its derivatives 
    # wrt to ind_admix_params are correct
    
    # TODO handle log-phi's
    
    # data parameters
    n_obs = g_obs.shape[0]
    k_approx = e_log_pop_freq.shape[1]
    
    
    # get expecations
    e_log_sticks, e_log_1m_sticks = \
            ef.get_e_log_logitnormal(
                lognorm_means = ind_admix_params['stick_means'],
                lognorm_infos = ind_admix_params['stick_infos'],
                gh_loc = gh_loc,
                gh_weights = gh_weights)
    
    e_log_cluster_probs = \
        modeling_lib.get_e_log_cluster_probabilities_from_e_log_stick(
            e_log_sticks, e_log_1m_sticks)
    
    # sum the e_z's over loci
    body_fun = lambda val, x :\
                    structure_model_lib.get_optimal_ezl(x[0], x[1], x[2],
                                        e_log_cluster_probs)[1].sum(-1) + val
    
    scan_fun = lambda val, x : (body_fun(val, x), None)
    
    init_val = np.zeros((n_obs, k_approx))
    ez_nxk = jax.lax.scan(scan_fun, init_val,
                        xs = (g_obs.transpose((1, 0, 2)),
                                e_log_pop_freq, e_log_1m_pop_freq))[0]
    if detach_ez:
        ez_nxk = jax.lax.stop_gradient(ez_nxk)

    # log-likelihood term 
    loglik_ind = (ez_nxk * e_log_cluster_probs).sum()
    
    # entropy term
    stick_entropy = \
            modeling_lib.get_stick_breaking_entropy(
                                    ind_admix_params['stick_means'],
                                    ind_admix_params['stick_infos'],
                                    gh_loc, gh_weights)
    
    # prior term
    ind_mix_dp_prior =  (prior_params_dict['dp_prior_alpha'] - 1) * np.sum(e_log_1m_sticks)
    
    return - (loglik_ind + ind_mix_dp_prior + stick_entropy).squeeze()

def get_ind_admix_params_loss(g_obs, 
                            ind_admix_params, 
                            pop_freq_beta_params, 
                            prior_params_dict, 
                            gh_loc, gh_weights, 
                            detach_ez = True):
    
    # used for testing the pseudo-loss above 
        
    vb_params_dict = dict({'pop_freq_beta_params':pop_freq_beta_params,
                           'ind_admix_params': ind_admix_params})

    return structure_model_lib.get_kl(g_obs,
                                        vb_params_dict,
                                        prior_params_dict,
                                        gh_loc, gh_weights,
                                        detach_ez = detach_ez)


########################
# A helper function to get
# objective functions and gradients
def define_structure_objective(g_obs, vb_params_dict,
                                vb_params_paragami,
                                prior_params_dict,
                                gh_loc = None, gh_weights = None,
                                log_phi = None, epsilon = 0., 
                                compile_hvp = False):

    # set up loss
    _kl_fun_free = paragami.FlattenFunctionInput(
                                original_fun=structure_model_lib.get_kl,
                                patterns = vb_params_paragami,
                                free = True,
                                argnums = 1)

    kl_fun_free = lambda x : _kl_fun_free(g_obs, x, prior_params_dict,
                                                     gh_loc, gh_weights,
                                                     log_phi, epsilon)

    # initial free parameters
    init_vb_free = vb_params_paragami.flatten(vb_params_dict, free = True)
    
    # define objective
    optim_objective = OptimizationObjectiveJaxtoNumpy(kl_fun_free, 
                                                     init_vb_free, 
                                                      compile_hvp = compile_hvp, 
                                                      print_every = 1,
                                                      log_every = 0)
    
    return optim_objective, init_vb_free
