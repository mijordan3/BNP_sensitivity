import jax

import jax.numpy as np
import jax.scipy as sp

from vb_lib import structure_model_lib

from bnpmodeling_runjingdev import cluster_quantities_lib, modeling_lib
from bnpmodeling_runjingdev.functional_sensitivity_lib import get_e_log_perturbation

import bnpmodeling_runjingdev.exponential_families as ef

from paragami import FlattenFunctionInput

import time

from copy import deepcopy

# using autograd to get natural paramters
# for testing only 
joint_loglik = lambda *x : structure_model_lib.\
                    get_e_joint_loglik_from_nat_params(*x, detach_ez=True)[0]


# get natural beta parameters for population frequencies
get_pop_beta_update1_ag = jax.jacobian(joint_loglik, argnums=1)
get_pop_beta_update2_ag = jax.jacobian(joint_loglik, argnums=2)

# get natural beta parameters for admixture sticks
get_stick_update1_ag = jax.jacobian(joint_loglik, argnums=3)
get_stick_update2_ag = jax.jacobian(joint_loglik, argnums=4)

def _update_pop_beta_l(g_obs_l, e_log_pop_freq_l, e_log_1m_pop_freq_l, 
                       e_log_cluster_probs, allele_prior_alpha, allele_prior_beta): 
    
    g_obs_l0 = g_obs_l[:, 0]
    g_obs_l1 = g_obs_l[:, 1]
    g_obs_l2 = g_obs_l[:, 2]
    
    _, e_z_l = structure_model_lib.get_optimal_ezl(g_obs_l, e_log_pop_freq_l, e_log_1m_pop_freq_l,
                                                    e_log_cluster_probs)
    
    beta_param_l1 = np.dot(g_obs_l1 + g_obs_l2, e_z_l[:, :, 0]) + \
                        np.dot(g_obs_l2, e_z_l[:, :, 1]) + (allele_prior_alpha - 1) 
    
    beta_param_l2 = np.dot(g_obs_l0, e_z_l[:, :, 0]) + \
                        np.dot(g_obs_l0 + g_obs_l1, e_z_l[:, :, 1]) + (allele_prior_beta - 1) 
    
    return np.stack([beta_param_l1, beta_param_l2]).transpose((1, 0))

def update_pop_beta(g_obs, vb_params_dict, prior_params_dict, 
                       gh_loc = None, gh_weights = None): 
    
    # prior parameters
    allele_prior_alpha = prior_params_dict['allele_prior_alpha']
    allele_prior_beta = prior_params_dict['allele_prior_beta']
    
    # get initial moments from vb_params
    e_log_sticks, e_log_1m_sticks, \
        e_log_pop_freq, e_log_1m_pop_freq = \
            structure_model_lib.get_moments_from_vb_params_dict(
                vb_params_dict, gh_loc, gh_weights)

    e_log_cluster_probs = \
        modeling_lib.get_e_log_cluster_probabilities_from_e_log_stick(
                            e_log_sticks, e_log_1m_sticks)
    
    # the per-loci update function
    f = lambda x : \
            _update_pop_beta_l(x[0], x[1], x[2], 
                       e_log_cluster_probs, allele_prior_alpha, allele_prior_beta)
    
    # the for-loop
    updated_beta_params = jax.lax.map(f, 
                                      (g_obs.transpose((1, 0, 2)), 
                                       e_log_pop_freq, 
                                       e_log_1m_pop_freq))
    
    return updated_beta_params



# def update_pop_beta(g_obs,
#                     e_log_pop_freq, e_log_1m_pop_freq,
#                     e_log_sticks, e_log_1m_sticks,
#                     dp_prior_alpha, allele_prior_alpha,
#                     allele_prior_beta):
#     # update population frequency parameters

#     beta_param1 = get_pop_beta_update1(g_obs,
#                     e_log_pop_freq, e_log_1m_pop_freq,
#                     e_log_sticks, e_log_1m_sticks,
#                     dp_prior_alpha, allele_prior_alpha,
#                     allele_prior_beta) + 1.0
#     beta_param2 = get_pop_beta_update2(g_obs,
#                     e_log_pop_freq, e_log_1m_pop_freq,
#                     e_log_sticks, e_log_1m_sticks,
#                     dp_prior_alpha, allele_prior_alpha,
#                     allele_prior_beta) + 1.0

#     beta_params = np.concatenate((beta_param1[:, :, None],
#                                 beta_param2[:, :, None]), axis = 2)

#     e_log_pop_freq, e_log_1m_pop_freq = \
#         modeling_lib.get_e_log_beta(beta_params)

#     return e_log_pop_freq, e_log_1m_pop_freq, beta_params


# def update_stick_beta(g_obs,
#                     e_log_pop_freq, e_log_1m_pop_freq,
#                     e_log_sticks, e_log_1m_sticks,
#                     dp_prior_alpha, allele_prior_alpha,
#                     allele_prior_beta):

#     # update individual admixtures
#     beta_param1 = get_stick_update1(g_obs,
#                 e_log_pop_freq, e_log_1m_pop_freq,
#                 e_log_sticks, e_log_1m_sticks,
#                 dp_prior_alpha, allele_prior_alpha,
#                 allele_prior_beta) + 1.0

#     beta_param2 = get_stick_update2(g_obs,
#                     e_log_pop_freq, e_log_1m_pop_freq,
#                     e_log_sticks, e_log_1m_sticks,
#                     dp_prior_alpha, allele_prior_alpha,
#                     allele_prior_beta) + 1.0

#     beta_params = np.concatenate((beta_param1[:, :, None],
#                                     beta_param2[:, :, None]), axis = 2)

#     e_log_sticks, e_log_1m_sticks = modeling_lib.get_e_log_beta(beta_params)

#     return e_log_sticks, e_log_1m_sticks, beta_params

# def run_cavi(g_obs, vb_params_dict,
#                 vb_params_paragami,
#                 prior_params_dict,
#                 gh_loc = None, gh_weights = None,
#                 log_phi = None, epsilon = 0.,
#                 x_tol = 1e-3,
#                 max_iter = 1000,
#                 print_every = 1,
#                 debug = False):
#     """
#     Runs coordinate ascent on the VB parameters.

#     Parameters
#     ----------
#     g_obs : ndarray
#         Array of size (n_obs x n_loci x 3), giving a one-hot encoding of
#         genotypes
#     vb_params_dict : dictionary
#         A dictionary that contains the variational parameters.
#     prior_params_dict : dictionary
#         A dictionary that contains the prior parameters.

#     Returns
#     -------
#     vb_params_dict : dictionary
#         A dictionary that contains the optimized variational parameters.
#     """

#     # prior parameters
#     dp_prior_alpha = prior_params_dict['dp_prior_alpha']
#     allele_prior_alpha = prior_params_dict['allele_prior_alpha']
#     allele_prior_beta = prior_params_dict['allele_prior_beta']

#     # get initial moments from vb_params
#     e_log_sticks, e_log_1m_sticks, \
#         e_log_pop_freq, e_log_1m_pop_freq = \
#             structure_model_lib.get_moments_from_vb_params_dict(
#                 vb_params_dict, gh_loc, gh_weights)

#     kl_old = 1e16
#     x_old = 1e16
#     kl_vec = []

#     use_logitnormal_sticks = 'ind_mix_stick_propn_mean' in vb_params_dict.keys()

#     # set up KL function
#     _get_kl = lambda vb_params_dict : \
#                 structure_model_lib.get_kl(g_obs, vb_params_dict,
#                                             prior_params_dict,
#                                             gh_loc = gh_loc,
#                                             gh_weights = gh_weights,
#                                             log_phi = log_phi,
#                                             epsilon = epsilon)

#     _get_kl = jax.jit(_get_kl)
#     def check_kl(vb_params_dict, kl_old):
#         kl = _get_kl(vb_params_dict)
#         kl_diff = kl_old - kl
#         assert kl_diff > 0, kl_diff
#         return kl

#     flatten_vb_params = lambda x : vb_params_paragami.flatten(x, free = True, validate_value = False)
#     flatten_vb_params = jax.jit(flatten_vb_params)

#     # compile cavi functions    
#     t0 = time.time()
#     update_pop_beta = jax.jit(update_pop_beta)
#     _ = update_pop_beta(g_obs,
#                     e_log_pop_freq, e_log_1m_pop_freq,
#                     e_log_sticks, e_log_1m_sticks,
#                     dp_prior_alpha, allele_prior_alpha,
#                     allele_prior_beta)
#     _ = _get_kl(vb_params_dict)
#     _ = flatten_vb_params(vb_params_dict)
#     if use_logitnormal_sticks:
#         stick_obj_fun, stick_mean_grad_fun, stick_info_grad_fun = \
#             prepare_logitnormal_stick_updates(g_obs,
#                                       vb_params_paragami,
#                                       prior_params_dict,
#                                       gh_loc, gh_weights,
#                                       log_phi = log_phi,
#                                       epsilon = epsilon)
#     else:
#         update_stick_beta = jax.jit(update_stick_beta)
        
#         _ = update_stick_beta(g_obs,
#                         e_log_pop_freq, e_log_1m_pop_freq,
#                         e_log_sticks, e_log_1m_sticks,
#                         dp_prior_alpha, allele_prior_alpha,
#                         allele_prior_beta)

#     print('CAVI compile time: {0:.3g}sec'.format(time.time() - t0))

#     print('\n running CAVI ...')
#     t0 = time.time()
#     time_vec = [t0]
#     for i in range(1, max_iter):

#         # update sticks
#         if use_logitnormal_sticks:
#             e_log_sticks, e_log_1m_sticks, \
#                 vb_params_dict['ind_mix_stick_propn_mean'], \
#                     vb_params_dict['ind_mix_stick_propn_info'] = \
#                         update_logitnormal_sticks(stick_obj_fun,
#                                                     stick_mean_grad_fun,
#                                                     stick_info_grad_fun,
#                                                     gh_loc,
#                                                     gh_weights,
#                                                     vb_params_dict,
#                                                     vb_params_paragami)
#         else:
#             e_log_sticks, e_log_1m_sticks, \
#                 vb_params_dict['ind_mix_stick_beta_params'] = \
#                     update_stick_beta(g_obs,
#                                     e_log_pop_freq, e_log_1m_pop_freq,
#                                     e_log_sticks, e_log_1m_sticks,
#                                     dp_prior_alpha, allele_prior_alpha,
#                                     allele_prior_beta)

#         if debug:
#             kl_old = check_kl(vb_params_dict, kl_old)

#         # update population frequency parameters
#         e_log_pop_freq, e_log_1m_pop_freq, \
#             vb_params_dict['pop_freq_beta_params'] = \
#                 update_pop_beta(g_obs,
#                                 e_log_pop_freq, e_log_1m_pop_freq,
#                                 e_log_sticks, e_log_1m_sticks,
#                                 dp_prior_alpha, allele_prior_alpha,
#                                 allele_prior_beta)

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


# #################
# # Functions to update logitnormal sticks
# #################
# def prepare_logitnormal_stick_updates(g_obs,
#                                       vb_params_paragami,
#                                       prior_params_dict,
#                                       gh_loc, gh_weights,
#                                       log_phi,
#                                       epsilon):

#     # set up objective function
#     stick_loss_flattened = \
#             FlattenFunctionInput(original_fun =_get_logitnormal_sticks_loss,
#                     patterns = [vb_params_paragami['ind_mix_stick_propn_mean'],
#                                 vb_params_paragami['ind_mix_stick_propn_info']],
#                     free = True,
#                     argnums = [1, 2])

#     stick_obj_fun = lambda stick_mean_free, stick_info_free, pop_freq_beta_params : \
#                         stick_loss_flattened(g_obs,
#                                                 stick_mean_free,
#                                                 stick_info_free,
#                                                 pop_freq_beta_params,
#                                                 prior_params_dict,
#                                                 gh_loc, gh_weights,
#                                                 log_phi,
#                                                 epsilon)

#     # set up gradients
#     stick_mean_grad_fun = jax.jit(jax.grad(stick_obj_fun, argnums = 0))
#     stick_info_grad_fun = jax.jit(jax.grad(stick_obj_fun, argnums = 1))
#     stick_obj_fun_jitted = jax.jit(stick_obj_fun)

#     # compile gradients
#     stick_mean_free = vb_params_paragami['ind_mix_stick_propn_mean'].flatten(\
#             vb_params_paragami['ind_mix_stick_propn_mean'].random(), free = True)
#     stick_info_free = vb_params_paragami['ind_mix_stick_propn_info'].flatten(\
#             vb_params_paragami['ind_mix_stick_propn_info'].random(), free = True)
#     pop_freq_beta_params = vb_params_paragami['pop_freq_beta_params'].random()

#     _ = stick_obj_fun_jitted(stick_mean_free, stick_info_free, pop_freq_beta_params)
#     _ = stick_mean_grad_fun(stick_mean_free, stick_info_free, pop_freq_beta_params)
#     _ = stick_info_grad_fun(stick_mean_free, stick_info_free, pop_freq_beta_params)

#     return stick_obj_fun_jitted, stick_mean_grad_fun, stick_info_grad_fun

# def _get_logitnormal_sticks_loss(g_obs,
#                                     stick_mean,
#                                     stick_info,
#                                     pop_freq_beta_params,
#                                     prior_params_dict,
#                                     gh_loc, gh_weights,
#                                     log_phi, epsilon):

#     vb_params_dict = dict({'pop_freq_beta_params':pop_freq_beta_params,
#                           'ind_mix_stick_propn_mean': stick_mean,
#                           'ind_mix_stick_propn_info': stick_info})

#     return structure_model_lib.get_kl(g_obs, vb_params_dict,
#                                         prior_params_dict,
#                                         gh_loc, gh_weights,
#                                         log_phi,
#                                         epsilon,
#                                         detach_ez = False)

# def update_logitnormal_sticks(stick_obj_fun,
#                                 stick_mean_grad_fun,
#                                 stick_info_grad_fun,
#                                 gh_loc, gh_weights,
#                                 vb_params_dict,
#                                 vb_params_paragami):

#     # we use a logitnormal approximation to the sticks : thus, updates
#     # can't be computed in closed form. We take a gradient step satisfying wolfe conditions

#     stick_mean = vb_params_dict['ind_mix_stick_propn_mean']
#     stick_info = vb_params_dict['ind_mix_stick_propn_info']
#     pop_freq_beta_params = vb_params_dict['pop_freq_beta_params']

#     # initial parameters
#     init_stick_mean_free = vb_params_paragami['ind_mix_stick_propn_mean'].\
#                                 flatten(stick_mean, free = True)
#     init_stick_info_free = vb_params_paragami['ind_mix_stick_propn_info'].\
#                                 flatten(stick_info, free = True)

#     # initial loss
#     init_ps_loss = stick_obj_fun(init_stick_mean_free,
#                                     init_stick_info_free,
#                                     pop_freq_beta_params)

#     grad_stick_mean = stick_mean_grad_fun(init_stick_mean_free,
#                                     init_stick_info_free,
#                                     pop_freq_beta_params)

#     grad_stick_info = stick_info_grad_fun(init_stick_mean_free,
#                                     init_stick_info_free,
#                                     pop_freq_beta_params)

#     # direction of step
#     step1 = - grad_stick_mean
#     step2 = - grad_stick_info

#     # choose stepsize
#     kl_new = 1e16
#     counter = 0.
#     rho = 0.5
#     alpha = 1.0 / rho

#     correction = np.sum(grad_stick_mean * step1) + np.sum(grad_stick_info * step2)

#     # for my sanity
#     assert correction < 0, correction
#     while (kl_new > (init_ps_loss + 1e-4 * alpha * correction)):
#         alpha *= rho

#         update_stick_mean_free = init_stick_mean_free + alpha * step1
#         update_stick_info_free = init_stick_info_free + alpha * step2

#         kl_new = stick_obj_fun(update_stick_mean_free,
#                                 update_stick_info_free,
#                                 pop_freq_beta_params)

#         counter += 1

#         if counter > 10:
#             print('could not find stepsize for stick optimizer')
#             break

#     # return parameters
#     update_stick_mean = vb_params_paragami['ind_mix_stick_propn_mean'].\
#                             fold(update_stick_mean_free, free = True)

#     update_stick_info = vb_params_paragami['ind_mix_stick_propn_info'].\
#                             fold(update_stick_info_free, free = True)

#     e_log_sticks, e_log_1m_sticks = \
#         ef.get_e_log_logitnormal(\
#             lognorm_means = update_stick_mean,
#             lognorm_infos = update_stick_info,
#             gh_loc = gh_loc,
#             gh_weights = gh_weights)

#     return e_log_sticks, e_log_1m_sticks, \
#                 update_stick_mean, update_stick_info
