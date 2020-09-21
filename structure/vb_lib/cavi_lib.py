import jax

import jax.numpy as np
import jax.scipy as sp

from vb_lib import structure_model_lib

from bnpmodeling_runjingdev import cluster_quantities_lib, modeling_lib
from bnpmodeling_runjingdev.functional_sensitivity_lib import get_e_log_perturbation

import bnpmodeling_runjingdev.exponential_families as ef

import time

from copy import deepcopy

# using autograd to get natural paramters
# get natural beta parameters for population frequencies
joint_loglik = lambda *x : structure_model_lib.\
                get_e_joint_loglik_from_nat_params(*x, set_optimal_z=False)[0]

get_pop_beta_update1 = jax.jit(jax.jacobian(joint_loglik, argnums=2))
get_pop_beta_update2 = jax.jit(jax.jacobian(joint_loglik, argnums=3))

# get natural beta parameters for admixture sticks
get_stick_update1 = jax.jit(jax.jacobian(joint_loglik, argnums=4))
get_stick_update2 = jax.jit(jax.jacobian(joint_loglik, argnums=5))

@jax.jit
def update_z(g_obs, e_log_sticks, e_log_1m_sticks, e_log_pop_freq,
                                e_log_1m_pop_freq):
    e_log_cluster_probs = \
            modeling_lib.get_e_log_cluster_probabilities_from_e_log_stick(
                                e_log_sticks, e_log_1m_sticks)

    loglik_cond_z = structure_model_lib.get_loglik_cond_z(g_obs, e_log_pop_freq,
                                e_log_1m_pop_freq, e_log_cluster_probs)

    return structure_model_lib.get_z_opt_from_loglik_cond_z(loglik_cond_z)

@jax.jit
def update_pop_beta(g_obs, e_z,
                    e_log_sticks, e_log_1m_sticks,
                    dp_prior_alpha, allele_prior_alpha,
                    allele_prior_beta):
    # update population frequency parameters

    n_loci = g_obs.shape[1]
    n_pop = e_z.shape[2]
    constant = np.zeros((n_loci, n_pop))
    assert g_obs.shape[0] == e_z.shape[0]
    assert g_obs.shape[1] == e_z.shape[1]
    assert e_log_sticks.shape[1] == (n_pop - 1)
    assert e_log_sticks.shape == e_log_1m_sticks.shape

    beta_param1 = get_pop_beta_update1(g_obs, e_z,
                    constant, constant,
                    e_log_sticks, e_log_1m_sticks,
                    dp_prior_alpha, allele_prior_alpha,
                    allele_prior_beta) + 1.0
    beta_param2 = get_pop_beta_update2(g_obs, e_z,
                    constant, constant,
                    e_log_sticks, e_log_1m_sticks,
                    dp_prior_alpha, allele_prior_alpha,
                    allele_prior_beta) + 1.0

    beta_params = np.concatenate((beta_param1[:, :, None],
                                beta_param2[:, :, None]), axis = 2)

    e_log_pop_freq, e_log_1m_pop_freq = \
        modeling_lib.get_e_log_beta(beta_params)

    return e_log_pop_freq, e_log_1m_pop_freq, beta_params


@jax.jit
def update_stick_beta(g_obs, e_z,
                    e_log_pop_freq, e_log_1m_pop_freq,
                    dp_prior_alpha, allele_prior_alpha,
                    allele_prior_beta):

    # constant
    # this is the shape of e_log_sticks
    constant = np.zeros((g_obs.shape[0], e_z.shape[2] - 1))

    # for my sanity, check these shapes ...
    assert g_obs.shape[0] == e_z.shape[0]
    assert e_log_pop_freq.shape[0] == e_z.shape[1]
    assert e_log_pop_freq.shape[1] == e_z.shape[2]
    assert e_log_pop_freq.shape == e_log_1m_pop_freq.shape

    # update individual admixtures
    beta_param1 = get_stick_update1(g_obs, e_z,
                e_log_pop_freq, e_log_1m_pop_freq,
                constant, constant,
                dp_prior_alpha, allele_prior_alpha,
                allele_prior_beta) + 1.0

    beta_param2 = get_stick_update2(g_obs, e_z,
                    e_log_pop_freq, e_log_1m_pop_freq,
                    constant, constant,
                    dp_prior_alpha, allele_prior_alpha,
                    allele_prior_beta) + 1.0

    beta_params = np.concatenate((beta_param1[:, :, None],
                                    beta_param2[:, :, None]), axis = 2)

    e_log_sticks, e_log_1m_sticks = modeling_lib.get_e_log_beta(beta_params)

    return e_log_sticks, e_log_1m_sticks, beta_params

def run_cavi(g_obs, vb_params_dict,
                vb_params_paragami,
                prior_params_dict,
                use_logitnormal_sticks,
                gh_loc = None, gh_weights = None,
                log_phi = None, epsilon = 0.,
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

    # prior parameters
    dp_prior_alpha = prior_params_dict['dp_prior_alpha']
    allele_prior_alpha = prior_params_dict['allele_prior_alpha']
    allele_prior_beta = prior_params_dict['allele_prior_beta']

    # get initial moments from vb_params
    e_log_sticks, e_log_1m_sticks, \
        e_log_pop_freq, e_log_1m_pop_freq = \
            structure_model_lib.get_moments_from_vb_params_dict(
                vb_params_dict, use_logitnormal_sticks,
                gh_loc, gh_weights)

    kl_old = 1e16
    x_old = 1e16
    kl_vec = []

    # set up stick functions
    if use_logitnormal_sticks:
        stick_obj_fun = lambda stick_mean_free, stick_info_free, e_z : \
                            _get_logitnormal_sticks_psloss(g_obs,
                                                        e_z,
                                                        stick_mean_free,
                                                        stick_info_free,
                                                        vb_params_paragami,
                                                        prior_params_dict,
                                                        gh_loc, gh_weights,
                                                        log_phi,
                                                        epsilon)
        stick_mean_grad_fun = jax.jit(jax.grad(stick_obj_fun, argnums = 0))
        stick_info_grad_fun = jax.jit(jax.grad(stick_obj_fun, argnums = 1))
        stick_obj_fun = jax.jit(stick_obj_fun)

    # set up KL function
    _get_kl = lambda vb_params_dict, e_z : \
                structure_model_lib.get_kl(g_obs, vb_params_dict,
                                            prior_params_dict,
                                            use_logitnormal_sticks,
                                            gh_loc = gh_loc,
                                            gh_weights = gh_weights,
                                            log_phi = log_phi,
                                            epsilon = epsilon,
                                            e_z = e_z,
                                            set_optimal_z = False)

    _get_kl = jax.jit(_get_kl)
    def check_kl(vb_params_dict, e_z, kl_old):
        kl = _get_kl(vb_params_dict, e_z)
        kl_diff = kl_old - kl
        assert kl_diff > 0
        return kl

    flatten_vb_params = lambda x : vb_params_paragami.flatten(x, free = True, validate_value = False)
    flatten_vb_params = jax.jit(flatten_vb_params)

    # compile cavi functions
    t0 = time.time()
    e_z = update_z(g_obs, e_log_sticks, e_log_1m_sticks, e_log_pop_freq,
                            e_log_1m_pop_freq)
    _ = update_pop_beta(g_obs, e_z,
                    e_log_sticks, e_log_1m_sticks,
                    dp_prior_alpha, allele_prior_alpha,
                    allele_prior_beta)
    _ = _get_kl(vb_params_dict, e_z)
    _ = flatten_vb_params(vb_params_dict)
    if use_logitnormal_sticks:
        stick_mean = vb_params_paragami['ind_mix_stick_propn_mean'].\
                        flatten(vb_params_dict['ind_mix_stick_propn_mean'], free = True)
        stick_info = vb_params_paragami['ind_mix_stick_propn_info'].\
                        flatten(vb_params_dict['ind_mix_stick_propn_info'], free = True)
        _ = stick_obj_fun(stick_mean, stick_info, e_z)
        _ = stick_mean_grad_fun(stick_mean, stick_info, e_z)
        _ = stick_info_grad_fun(stick_mean, stick_info, e_z)
    else:
        _ = update_stick_beta(g_obs, e_z,
                        e_log_pop_freq, e_log_1m_pop_freq,
                        dp_prior_alpha, allele_prior_alpha,
                        allele_prior_beta)

    print('CAVI compile time: {0:.3g}sec'.format(time.time() - t0))

    print('\n running CAVI ...')
    t0 = time.time()
    time_vec = [t0]
    for i in range(1, max_iter):

        # update z
        e_z = update_z(g_obs, e_log_sticks, e_log_1m_sticks, e_log_pop_freq,
                                e_log_1m_pop_freq)

        # update sticks
        if use_logitnormal_sticks:
            e_log_sticks, e_log_1m_sticks, \
                vb_params_dict['ind_mix_stick_propn_mean'], \
                    vb_params_dict['ind_mix_stick_propn_info'] = \
                        update_logitnormal_sticks(stick_obj_fun,
                                                    stick_mean_grad_fun,
                                                    stick_info_grad_fun,
                                                    e_z,
                                                    gh_loc,
                                                    gh_weights,
                                                    vb_params_dict,
                                                    vb_params_paragami)
        else:
            e_log_sticks, e_log_1m_sticks, \
                vb_params_dict['ind_mix_stick_beta_params'] = \
                    update_stick_beta(g_obs, e_z,
                                    e_log_pop_freq, e_log_1m_pop_freq,
                                    dp_prior_alpha, allele_prior_alpha,
                                    allele_prior_beta)

        # update population frequency parameters
        e_log_pop_freq, e_log_1m_pop_freq, \
            vb_params_dict['pop_freq_beta_params'] = \
                update_pop_beta(g_obs, e_z,
                                e_log_sticks, e_log_1m_sticks,
                                dp_prior_alpha, allele_prior_alpha,
                                allele_prior_beta)

        if (i % print_every) == 0 or debug:
            kl = check_kl(vb_params_dict, e_z, kl_old)
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

    print('Elapsed: {} steps in {} seconds'.format(
            i, round(time.time() - t0, 2)))

    return vb_params_dict, vb_opt, e_z, np.array(kl_vec), np.array(time_vec) - t0


# just a useful function
def get_ez_from_vb_params_dict(g_obs, vb_params_dict, use_logitnormal_sticks,
                                gh_loc, gh_weights):
    if use_logitnormal_sticks:
        assert gh_loc is not None
        assert gh_weights is not None

    e_log_sticks, e_log_1m_sticks, \
        e_log_pop_freq, e_log_1m_pop_freq = \
            structure_model_lib.get_moments_from_vb_params_dict(
                                    vb_params_dict, use_logitnormal_sticks,
                                    gh_loc, gh_weights)

    return update_z(g_obs, e_log_sticks, e_log_1m_sticks, e_log_pop_freq,
                            e_log_1m_pop_freq)

#################
# Functions to update logitnormal sticks
#################
def _get_logitnormal_sticks_psloss(g_obs,
                                    e_z,
                                    stick_mean_free,
                                    stick_info_free,
                                    vb_params_paragami,
                                    prior_params_dict,
                                    gh_loc, gh_weights,
                                    log_phi, epsilon):
    # this function only returns the terms in the loss that depend on the
    # logitnormal sticks. The gradients wrt to the sticks are correct,
    # though the loss itself is not

    stick_mean = vb_params_paragami['ind_mix_stick_propn_mean'].\
                    fold(stick_mean_free, free = True)
    stick_info = vb_params_paragami['ind_mix_stick_propn_info'].\
                    fold(stick_info_free, free = True)

    e_log_sticks, e_log_1m_sticks = \
            ef.get_e_log_logitnormal(
                lognorm_means = stick_mean,
                lognorm_infos = stick_info,
                gh_loc = gh_loc,
                gh_weights = gh_weights)

    e_log_cluster_probs = \
        modeling_lib.get_e_log_cluster_probabilities_from_e_log_stick(
                            e_log_sticks, e_log_1m_sticks)

    dp_prior = (prior_params_dict['dp_prior_alpha'].squeeze() - 1) * np.sum(e_log_1m_sticks)
    stick_entropy = \
            modeling_lib.get_stick_breaking_entropy(
                                    stick_mean,
                                    stick_info,
                                    gh_loc, gh_weights)

    n = e_z.shape[0]
    k = e_z.shape[2]

    loglik_term = (e_log_cluster_probs.reshape(n, 1, k, 1) * e_z).sum()

    # perturbed term
    if log_phi is not None:
        e_log_pert = get_e_log_perturbation(log_phi,
                                stick_mean, stick_info,
                                epsilon, gh_loc, gh_weights, sum_vector=True)
    else:
        e_log_pert = 0.0

    return - (loglik_term + dp_prior + stick_entropy) + e_log_pert

def _get_logitnormal_sticks_loss(g_obs,
                    e_z,
                    stick_mean_free_params,
                    stick_info_free_params,
                    vb_params_dict,
                    vb_params_paragami,
                    prior_params_dict,
                    gh_loc, gh_weights):

    # returns the kl loss as a function of stick parameters
    # can be used to test the function above, to make sure gradients are the same


    stick_mean = vb_params_paragami['ind_mix_stick_propn_mean'].\
                    fold(stick_mean_free_params, free = True)
    stick_info = vb_params_paragami['ind_mix_stick_propn_info'].\
                    fold(stick_info_free_params, free = True)

    vb_params_dict['ind_mix_stick_propn_mean'] = stick_mean
    vb_params_dict['ind_mix_stick_propn_info'] = stick_info

    use_logitnormal_sticks = True
    return structure_model_lib.get_kl(g_obs, vb_params_dict, prior_params_dict,
                        use_logitnormal_sticks,
                        gh_loc, gh_weights,
                        e_z = e_z)

def update_logitnormal_sticks(stick_obj_fun,
                                stick_mean_grad_fun,
                                stick_info_grad_fun,
                                e_z,
                                gh_loc, gh_weights,
                                vb_params_dict,
                                vb_params_paragami):

    # we use a logitnormal approximation to the sticks : thus, updates
    # can't be computed in closed form. We take a gradient step satisfying wolfe conditions

    stick_mean = vb_params_dict['ind_mix_stick_propn_mean']
    stick_info = vb_params_dict['ind_mix_stick_propn_info']

    # initial parameters
    init_stick_mean_free = vb_params_paragami['ind_mix_stick_propn_mean'].\
                                flatten(stick_mean, free = True)
    init_stick_info_free = vb_params_paragami['ind_mix_stick_propn_info'].\
                                flatten(stick_info, free = True)

    # initial loss
    init_ps_loss = stick_obj_fun(init_stick_mean_free,
                                    init_stick_info_free,
                                    e_z)

    grad_stick_mean = stick_mean_grad_fun(init_stick_mean_free,
                                    init_stick_info_free,
                                    e_z)

    grad_stick_info = stick_info_grad_fun(init_stick_mean_free,
                                    init_stick_info_free,
                                    e_z)

    # direction of step
    step1 = - grad_stick_mean
    step2 = - grad_stick_info

    # choose stepsize
    kl_new = 1e16
    counter = 0.
    rho = 0.5
    alpha = 1.0 / rho

    correction = np.sum(grad_stick_mean * step1) + np.sum(grad_stick_info * step2)

    # for my sanity
    assert correction < 0
    while (kl_new > (init_ps_loss + 1e-4 * alpha * correction)):
        alpha *= rho

        update_stick_mean_free = init_stick_mean_free + alpha * step1
        update_stick_info_free = init_stick_info_free + alpha * step2

        kl_new = stick_obj_fun(update_stick_mean_free,
                                update_stick_info_free,
                                e_z)

        counter += 1

        if counter > 10:
            print('could not find stepsize for stick optimizer')
            break

    # return parameters
    update_stick_mean = vb_params_paragami['ind_mix_stick_propn_mean'].\
                            fold(update_stick_mean_free, free = True)

    update_stick_info = vb_params_paragami['ind_mix_stick_propn_info'].\
                            fold(update_stick_info_free, free = True)

    e_log_sticks, e_log_1m_sticks = \
        ef.get_e_log_logitnormal(\
            lognorm_means = update_stick_mean,
            lognorm_infos = update_stick_info,
            gh_loc = gh_loc,
            gh_weights = gh_weights)

    return e_log_sticks, e_log_1m_sticks, \
                update_stick_mean, update_stick_info
