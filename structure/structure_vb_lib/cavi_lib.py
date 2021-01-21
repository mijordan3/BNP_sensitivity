import jax

import jax.numpy as np
import jax.scipy as sp

from structure_vb_lib import structure_model_lib

from bnpmodeling_runjingdev import cluster_quantities_lib, modeling_lib
from bnpmodeling_runjingdev.functional_sensitivity_lib import get_e_log_perturbation

import bnpmodeling_runjingdev.exponential_families as ef

from paragami import FlattenFunctionInput

import time

from copy import deepcopy

# using autograd to get natural paramters
# for testing only 
joint_loglik = lambda *x : structure_model_lib.\
                    get_e_joint_loglik_from_nat_params(*x, detach_ez=True)


# get natural beta parameters for population frequencies
get_pop_beta_update1_ag = jax.jacobian(joint_loglik, argnums=1)
get_pop_beta_update2_ag = jax.jacobian(joint_loglik, argnums=2)

# get natural beta parameters for admixture sticks
get_stick_update1_ag = jax.jacobian(joint_loglik, argnums=3)
get_stick_update2_ag = jax.jacobian(joint_loglik, argnums=4)

def _get_ez_l_from_moments(g_obs_l,
                           e_log_pop_freq_l,
                           e_log_1m_pop_freq_l,
                           e_log_cluster_probs): 
    
    e_z_l_free = structure_model_lib.get_loglik_cond_z_l(g_obs_l,
                                                   e_log_pop_freq_l,
                                                   e_log_1m_pop_freq_l,
                                                   e_log_cluster_probs)
    
    return structure_model_lib.get_ez_from_ezfree(e_z_l_free, detach_ez = True)[0]
    
def _update_pop_beta_l(g_obs_l, e_log_pop_freq_l, e_log_1m_pop_freq_l, 
                       e_log_cluster_probs, allele_prior_alpha, allele_prior_beta, 
                       data_weight): 
    
    g_obs_l0 = g_obs_l[:, 0]
    g_obs_l1 = g_obs_l[:, 1]
    g_obs_l2 = g_obs_l[:, 2]
    
    e_z_l = _get_ez_l_from_moments(g_obs_l,
                                   e_log_pop_freq_l,
                                   e_log_1m_pop_freq_l,
                                   e_log_cluster_probs)
    
    beta_param_l1 = (np.dot(g_obs_l1 + g_obs_l2, e_z_l[:, :, 0]) + \
                        np.dot(g_obs_l2, e_z_l[:, :, 1])) * data_weight + \
                        (allele_prior_alpha - 1) 
    
    beta_param_l2 = (np.dot(g_obs_l0, e_z_l[:, :, 0]) + \
                        np.dot(g_obs_l0 + g_obs_l1, e_z_l[:, :, 1])) * data_weight + \
                            (allele_prior_beta - 1) 
    
    return np.stack([beta_param_l1, beta_param_l2]).transpose((1, 0))

def update_pop_beta(g_obs, e_log_pop_freq, e_log_1m_pop_freq, 
                       e_log_cluster_probs, prior_params_dict, 
                       data_weight = 1): 
    
    # prior parameters
    allele_prior_alpha = prior_params_dict['allele_prior_alpha']
    allele_prior_beta = prior_params_dict['allele_prior_beta']
        
    # the per-loci update function
    f = lambda x : \
            _update_pop_beta_l(x[0], x[1], x[2], 
                       e_log_cluster_probs, allele_prior_alpha, allele_prior_beta, 
                       data_weight)
    
    # the for-loop to update
    beta_update = jax.lax.map(f, 
                              (g_obs.transpose((1, 0, 2)), 
                               e_log_pop_freq, 
                               e_log_1m_pop_freq)) + 1
    
    # get moments 
    e_log_pop_freq, e_log_1m_pop_freq = \
        modeling_lib.get_e_log_beta(beta_update)
        
    return beta_update, e_log_pop_freq, e_log_1m_pop_freq


def update_ind_admix_beta(g_obs, e_log_pop_freq, e_log_1m_pop_freq, 
                            e_log_cluster_probs, prior_params_dict): 
    
    
    n_obs = g_obs.shape[0]
    k_approx = e_log_pop_freq.shape[1]
    
    # prior parameters
    dp_prior_alpha = prior_params_dict['dp_prior_alpha']
        
    # sum the e_z's over loci
    body_fun = lambda val, x :\
                    _get_ez_l_from_moments(x[0], x[1], x[2],
                                           e_log_cluster_probs).sum(-1) + val
    
    scan_fun = lambda val, x : (body_fun(val, x), None)
    
    init_val = np.zeros((n_obs, k_approx))
        
    out = jax.lax.scan(scan_fun, init_val,
                        xs = (g_obs.transpose((1, 0, 2)),
                                e_log_pop_freq, e_log_1m_pop_freq))[0]
    
    # get beta updates
    beta_update1 = out[:, 0:(k_approx-1)] + 1
    
    tmp = out[:, 1:k_approx]
    beta_update2 = np.cumsum(np.flip(tmp, axis = 1), axis = 1) + \
                        (dp_prior_alpha - 1) + 1
    beta_update2 = np.flip(beta_update2, axis = 1)
    
    beta_update = np.stack([beta_update1, beta_update2]).transpose((1, 2, 0))
    
    # update moments
    e_log_sticks, e_log_1m_sticks = \
            modeling_lib.get_e_log_beta(beta_update)
    
    e_log_cluster_probs = \
        modeling_lib.get_e_log_cluster_probabilities_from_e_log_stick(
            e_log_sticks, e_log_1m_sticks)

    return beta_update, e_log_cluster_probs

def run_cavi(g_obs, vb_params_dict,
                vb_params_paragami,
                prior_params_dict,
                x_tol = 1e-3,
                max_iter = 1000,
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

    # get initial moments from vb_params
    e_log_sticks, e_log_1m_sticks, \
        e_log_pop_freq, e_log_1m_pop_freq = \
            structure_model_lib.get_moments_from_vb_params_dict(
                vb_params_dict)
    
    e_log_cluster_probs = \
        modeling_lib.get_e_log_cluster_probabilities_from_e_log_stick(
            e_log_sticks, e_log_1m_sticks)

    kl_old = 1e16
    x_old = 1e16
    kl_vec = []

    # set up KL function
    _get_kl = lambda vb_params_dict : \
                structure_model_lib.get_kl(g_obs, 
                                           vb_params_dict,
                                           prior_params_dict)
    _get_kl = jax.jit(_get_kl)
    def check_kl(vb_params_dict, kl_old):
        kl = _get_kl(vb_params_dict)
        kl_diff = kl_old - kl
        assert kl_diff > 0, kl_diff
        return kl
    
    flatten_vb_params = lambda x : vb_params_paragami.flatten(x, free = True, validate_value = False)
    flatten_vb_params = jax.jit(flatten_vb_params)

    # compile cavi functions  
    print('Compiling cavi functions ...')
    t0 = time.time()
    update_pop_beta_jitted = jax.jit(update_pop_beta)
    update_ind_admix_beta_jitted = jax.jit(update_ind_admix_beta)
    
    out = update_pop_beta_jitted(g_obs,
                               e_log_pop_freq,
                               e_log_1m_pop_freq, 
                               e_log_cluster_probs,
                               prior_params_dict)
    _ = out[0].block_until_ready()
    
    out = update_ind_admix_beta_jitted(g_obs,
                                     e_log_pop_freq,
                                     e_log_1m_pop_freq,
                                     e_log_cluster_probs, 
                                     prior_params_dict)
    _ = out[0].block_until_ready()
    _ = _get_kl(vb_params_dict).block_until_ready()
    _ = flatten_vb_params(vb_params_dict).block_until_ready()
    print('CAVI compile time: {0:.3g}sec'.format(time.time() - t0))

    print('\n running CAVI ...')
    time_vec = [time.time()]
    for i in range(1, max_iter):
        
        # update indivual admixtures
        vb_params_dict['ind_admix_params']['stick_beta'], \
            e_log_cluster_probs = \
                update_ind_admix_beta_jitted(g_obs, 
                                  e_log_pop_freq, e_log_1m_pop_freq, 
                                  e_log_cluster_probs, prior_params_dict)

        if debug:
            kl_old = check_kl(vb_params_dict, kl_old)

        # update population frequency parameters
        vb_params_dict['pop_freq_beta_params'], \
            e_log_pop_freq, e_log_1m_pop_freq = \
                update_pop_beta_jitted(g_obs, e_log_pop_freq, e_log_1m_pop_freq, 
                       e_log_cluster_probs, prior_params_dict)

        if (i % print_every) == 0 or debug:
            kl = check_kl(vb_params_dict, kl_old)
            kl_vec.append(kl)
            time_vec.append(time.time())
            kl_old = kl

            print('iteration [{}]; kl:{}; elapsed: {}secs'.format(i,
                                        round(kl, 6),
                                        round(time_vec[-1] - time_vec[-2], 4)))

        x_diff = flatten_vb_params(vb_params_dict) - x_old

        if np.abs(x_diff).max() < x_tol:
            print('CAVI done.')
            break

        x_old = flatten_vb_params(vb_params_dict)

    if i == (max_iter - 1):
        print('Done. Warning, max iterations reached. ')

    vb_opt = flatten_vb_params(vb_params_dict)
    
    final_kl = _get_kl(vb_params_dict)
    kl_vec.append(final_kl)
    time_vec.append(time.time())
    print('final KL: {:.6f}'.format(final_kl))
    print('Elapsed: {} steps in {:.2f} seconds'.format(i, time_vec[-1] - time_vec[0]))

    return vb_params_dict, vb_opt, np.array(kl_vec), np.array(time_vec)[1:] - time_vec[0]


