import jax
import jax.numpy as np
import jax.scipy as sp

import paragami

from vb_lib import structure_model_lib

import bnpmodeling_runjingdev.exponential_families as ef
import bnpmodeling_runjingdev.modeling_lib as modeling_lib

from bnpmodeling_runjingdev.optimization_lib import OptimizationObjectiveJaxtoNumpy


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
