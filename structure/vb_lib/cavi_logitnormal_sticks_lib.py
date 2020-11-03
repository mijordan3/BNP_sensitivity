import jax
from jax import random
import jax.numpy as np
import jax.scipy as sp

import paragami

import numpy as onp

from vb_lib import structure_model_lib
from vb_lib.cavi_lib import update_pop_beta
from vb_lib.structure_optimization_lib import StickObjective

from bnpmodeling_runjingdev import modeling_lib
import bnpmodeling_runjingdev.exponential_families as ef

import time

def run_stoch_cavi(g_obs, vb_params_dict,
                    vb_params_paragami,
                    prior_params_dict,
                    gh_loc, gh_weights,
                    seed = 1,
                    batchsize = 100,
                    f_tol = 1e-3,
                    max_iter = 1000,
                    kappa = 0.9,
                    print_every = 1,
                    debug = False):
    """
    Runs coordinate ascent on the VB parameters.

    Parameters
    ----------
    g_obs : ndarray
        Array of size (n_obs x n_loci x 3), giving a one-hot encoding of
        genotypes
    vb_params_dict : dictionary
        A dictionary that contains the variational parameters.
    prior_params_dict : dictionary
        A dictionary that contains the prior parameters.

    Returns
    -------
    vb_params_dict : dictionary
        A dictionary that contains the optimized variational parameters.
    """
    
    ###################
    # some cleaning
    ###################
    # change vb param dict to numpy arrays 
    # instead of jax.numpy arrays 
    # (we will be doing array assignment)
    vb_params_dict['ind_admix_params']['stick_means'] = \
        onp.array(vb_params_dict['ind_admix_params']['stick_means'])
    vb_params_dict['ind_admix_params']['stick_infos'] = \
        onp.array(vb_params_dict['ind_admix_params']['stick_infos'])
    
    n_obs = g_obs.shape[0]
    
    if batchsize > n_obs:
        batchsize = n_obs
        
    ########################
    # get initial moments from vb_params
    ########################
    e_log_pop_freq, e_log_1m_pop_freq = \
        modeling_lib.get_e_log_beta(vb_params_dict['pop_freq_beta_params'])
    
    kl_old = 1e16
    x_old = 1e16
    kl_vec = []
    
    ########################
    # set up KL function
    ########################
    _get_kl = lambda vb_params_dict : \
                structure_model_lib.get_kl(g_obs, 
                                           vb_params_dict,
                                           prior_params_dict, 
                                          gh_loc, gh_weights)
    get_kl = jax.jit(_get_kl)
    def check_kl(vb_params_dict, kl_old):
        kl = get_kl(vb_params_dict)
        kl_diff = kl_old - kl
        assert kl_diff >= 0, kl_diff
        return kl
    
    #####################
    # Function to get stick moments
    #####################
    def _get_e_log_cluster_probs(ind_stick_params): 
        e_log_sticks, e_log_1m_sticks = \
            ef.get_e_log_logitnormal(
                lognorm_means = ind_stick_params['stick_means'],
                lognorm_infos = ind_stick_params['stick_infos'],
                gh_loc = gh_loc,
                gh_weights = gh_weights)
        return modeling_lib.get_e_log_cluster_probabilities_from_e_log_stick(
                        e_log_sticks, e_log_1m_sticks)
    
    get_e_log_cluster_probs = jax.jit(_get_e_log_cluster_probs)
    update_pop_beta_jitted = jax.jit(update_pop_beta)
    
    ########################
    # set up stick objective
    ########################
    _, vb_params_paragami_n = \
        structure_model_lib.get_vb_params_paragami_object(n_obs = 1,
                                                          n_loci = g_obs.shape[1],
                                                          k_approx = e_log_pop_freq.shape[1],
                                                          use_logitnormal_sticks = True)

    stick_objective_n = StickObjective(g_obs[0:1], 
                                        vb_params_paragami_n,
                                        prior_params_dict, 
                                        gh_loc, gh_weights, 
                                        compute_hess = True)

    for i in range(0, max_iter):
        if i == 0: 
            print('Compiling ... first iteration of cavi might be slow')
            t0 = time.time()
        if i == 1:
            # start timing after first iteration 
            # so everything has compiled. 
            
            t0 = time.time()
            time_vec = [t0]

        #####################
        # subsample individuals
        #####################
        key = random.PRNGKey(seed + i)
        indx_sampled = random.choice(key, g_obs.shape[0],
                                         (batchsize,), 
                                         replace = False)
        g_obs_sub = g_obs[indx_sampled]
        
        #####################
        # Update individual admixtures 
        #####################
        
        # subset vb parameters
        ind_admix_params_sub = dict({'stick_means': \
                                         vb_params_dict['ind_admix_params']['stick_means'][indx_sampled, :], 
                                     'stick_infos': \
                                         vb_params_dict['ind_admix_params']['stick_infos'][indx_sampled, :]})
        
        # optimize sticks w newton
        ind_admix_update = update_ind_admix_sticks(g_obs_sub, ind_admix_params_sub, 
                                                    stick_objective_n, 
                                                    e_log_pop_freq, e_log_1m_pop_freq)
        
        # update parameters
        vb_params_dict['ind_admix_params']['stick_means'][indx_sampled, :] = \
            ind_admix_update['stick_means']
        vb_params_dict['ind_admix_params']['stick_infos'][indx_sampled, :] = \
            ind_admix_update['stick_infos']
        
        if debug:
            kl_old = check_kl(vb_params_dict, kl_old)
        
        # update moments for the batch
        e_log_cluster_probs = get_e_log_cluster_probs(ind_admix_update)
        
        #####################
        # update population frequency parameters
        #####################
        beta_update, _e_log_pop_freq, _e_log_1m_pop_freq = \
                update_pop_beta_jitted(g_obs_sub, e_log_pop_freq, e_log_1m_pop_freq, 
                                       e_log_cluster_probs, prior_params_dict, 
                                       data_weight = n_obs / batchsize)
        
        rho_t = (1 + i)**(-kappa)
        e_log_pop_freq = e_log_pop_freq * (1 - rho_t) + _e_log_pop_freq * rho_t 
        e_log_1m_pop_freq = e_log_1m_pop_freq * (1 - rho_t) + _e_log_1m_pop_freq * rho_t 
        vb_params_dict['pop_freq_beta_params'] = \
                        vb_params_dict['pop_freq_beta_params'] * (1 - rho_t) + \
                        beta_update * rho_t 
        
        if i == 0: 
            kl_old = get_kl(vb_params_dict)
            print('done compiling. Elapsed: {0:3g}'.format(time.time() - t0))
            
        else: 
            if (i % print_every) == 0 or debug:
                kl = 1.0 # get_kl(vb_params_dict)
                kl_vec.append(kl)
                time_vec.append(time.time())
                kl_old = kl

                print('iteration [{}]; kl:{}; elapsed: {}secs'.format(i,
                                            round(kl, 6),
                                            round(time_vec[-1] - time_vec[-2], 4)))

def update_ind_admix_sticks(g_obs, ind_admix_params, stick_objective_n, 
                            e_log_pop_freq, e_log_1m_pop_freq): 
    
    vb_params_dict_n = dict({'ind_admix_params' : 
                         dict({'stick_means': 0, 
                               'stick_infos': 0})})
    
    for n in range(g_obs.shape[0]):

        vb_params_dict_n['ind_admix_params']['stick_means'] = \
            ind_admix_params['stick_means'][n]
        vb_params_dict_n['ind_admix_params']['stick_infos'] = \
            ind_admix_params['stick_infos'][n]

        # run netwon ... 
        stick_updates, out = stick_objective_n.optimize_sticks(
                                            g_obs[n:(n+1)], 
                                            vb_params_dict_n['ind_admix_params'],
                                            e_log_pop_freq, e_log_1m_pop_freq)

        # update the parameters
        ind_admix_params['stick_means'][n] = stick_updates['stick_means']
        ind_admix_params['stick_infos'][n] = stick_updates['stick_infos']
    
    return ind_admix_params


    
#         if (i % print_every) == 0 or debug:
#             kl = check_kl(vb_params_dict, kl_old)
#             kl_vec.append(kl)
#             time_vec.append(time.time())
#             kl_old = kl

#             print('iteration [{}]; kl:{}; elapsed: {}secs'.format(i,
#                                         round(kl, 6),
#                                         round(time_vec[-1] - time_vec[-2], 4)))

#         x_diff = flatten_vb_params(vb_params_dict) - x_old

#         if np.abs(x_diff).max() < x_tol:
#             print('CAVI done.')
#             break

#         x_old = flatten_vb_params(vb_params_dict)

#     if i == (max_iter - 1):
#         print('Done. Warning, max iterations reached. ')

#     vb_opt = flatten_vb_params(vb_params_dict)

#     print('Elapsed: {} steps in {} seconds'.format(
#             i, round(time.time() - t0, 2)))

#     return vb_params_dict, vb_opt, np.array(kl_vec), np.array(time_vec) - t0

