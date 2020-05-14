import autograd

import autograd.numpy as np
import autograd.scipy as sp

from vb_lib import structure_model_lib

from bnpmodeling_runjingdev import cluster_quantities_lib, modeling_lib

import LinearResponseVariationalBayes.ExponentialFamilies as ef

import time

from copy import deepcopy

# using autograd to get natural paramters
# get natural beta parameters for population frequencies
get_pop_beta_update1 = autograd.jacobian(
            structure_model_lib.get_e_joint_loglik_from_nat_params, argnum=2)
get_pop_beta_update2 = autograd.jacobian(
        structure_model_lib.get_e_joint_loglik_from_nat_params, argnum=3)

# get natural beta parameters for admixture sticks
get_stick_update1 = autograd.jacobian(
    structure_model_lib.get_e_joint_loglik_from_nat_params, argnum=4)
get_stick_update2 = autograd.jacobian(
    structure_model_lib.get_e_joint_loglik_from_nat_params, argnum=5)


def update_z(g_obs, e_log_sticks, e_log_1m_sticks, e_log_pop_freq,
                                e_log_1m_pop_freq):
    e_log_cluster_probs = \
            modeling_lib.get_e_log_cluster_probabilities_from_e_log_stick(
                                e_log_sticks, e_log_1m_sticks)

    loglik_cond_z = structure_model_lib.get_loglik_cond_z(g_obs, e_log_pop_freq,
                                e_log_1m_pop_freq, e_log_cluster_probs)

    return structure_model_lib.get_z_opt_from_loglik_cond_z(loglik_cond_z)

def update_pop_beta(g_obs, e_z,
                    e_log_sticks, e_log_1m_sticks,
                    dp_prior_alpha, allele_prior_alpha,
                    allele_prior_beta,
                    obs_weights = None,
                    return_moments = True):
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
                    allele_prior_beta,
                    obs_weights = obs_weights) + 1.0
    beta_param2 = get_pop_beta_update2(g_obs, e_z,
                    constant, constant,
                    e_log_sticks, e_log_1m_sticks,
                    dp_prior_alpha, allele_prior_alpha,
                    allele_prior_beta,
                    obs_weights = obs_weights) + 1.0

    beta_params = np.concatenate((beta_param1[:, :, None],
                                beta_param2[:, :, None]), axis = 2)

    if return_moments:
        e_log_pop_freq, e_log_1m_pop_freq = \
            modeling_lib.get_e_log_beta(beta_params)
        return e_log_pop_freq, e_log_1m_pop_freq, beta_params
    else:
        return beta_params

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
    if use_logitnormal_sticks:
        assert gh_loc is not None
        assert gh_weights is not None

        assert 'ind_mix_stick_propn_info' in vb_params_dict.keys()
        assert 'ind_mix_stick_propn_mean' in vb_params_dict.keys()

    # get prior parameters
    dp_prior_alpha = prior_params_dict['dp_prior_alpha']
    allele_prior_alpha = prior_params_dict['allele_prior_alpha']
    allele_prior_beta = prior_params_dict['allele_prior_beta']

    # get initial moments from vb_params
    e_log_sticks, e_log_1m_sticks, \
        e_log_pop_freq, e_log_1m_pop_freq = \
            structure_model_lib.get_moments_from_vb_params_dict(
                vb_params_dict, use_logitnormal_sticks,
                gh_loc, gh_weights)

    kl_old = np.Inf
    x_old = np.Inf
    kl_vec = []

    t0 = time.time()

    time_vec = [t0]

    _get_kl = lambda vb_params_dict, e_z : \
                structure_model_lib.get_kl(g_obs, vb_params_dict,
                                            prior_params_dict,
                                            use_logitnormal_sticks,
                                            e_z = e_z,
                                            gh_loc = gh_loc,
                                            gh_weights = gh_weights)


    for i in range(1, max_iter):
        # update z
        e_z = update_z(g_obs, e_log_sticks, e_log_1m_sticks, e_log_pop_freq,
                                e_log_1m_pop_freq)
        if debug:
            kl = _get_kl(vb_params_dict, e_z)
            kl_diff = kl_old - kl
            assert kl_diff > 0
            kl_old = kl


        # update individual admixtures
        if use_logitnormal_sticks:
            vb_params_dict['ind_mix_stick_propn_mean'], \
                vb_params_dict['ind_mix_stick_propn_info'] = \
                    update_logitnormal_sticks(g_obs, e_z,
                                    vb_params_dict['ind_mix_stick_propn_mean'],
                                    vb_params_dict['ind_mix_stick_propn_info'],
                                    vb_params_paragami, prior_params_dict,
                                    gh_loc, gh_weights)

            e_log_sticks, e_log_1m_sticks = \
                ef.get_e_log_logitnormal(\
                    lognorm_means = vb_params_dict['ind_mix_stick_propn_mean'],
                    lognorm_infos = vb_params_dict['ind_mix_stick_propn_info'],
                    gh_loc = gh_loc,
                    gh_weights = gh_weights)

        else:
            e_log_sticks, e_log_1m_sticks, \
                vb_params_dict['ind_mix_stick_beta_params'] = \
                    update_stick_beta(g_obs, e_z,
                                        e_log_pop_freq, e_log_1m_pop_freq,
                                        dp_prior_alpha, allele_prior_alpha,
                                        allele_prior_beta)
        if debug:
            kl = _get_kl(vb_params_dict, e_z)
            kl_diff = kl_old - kl
            assert kl_diff > 0
            kl_old = kl

        # update population frequencies
        e_log_pop_freq, e_log_1m_pop_freq, \
            vb_params_dict['pop_freq_beta_params'] = \
                update_pop_beta(g_obs, e_z,
                                e_log_sticks, e_log_1m_sticks,
                                dp_prior_alpha, allele_prior_alpha,
                                allele_prior_beta)

        if (i % print_every) == 0 or debug:
            kl = _get_kl(vb_params_dict, e_z)

            kl_vec.append(kl)
            time_vec.append(time.time())

            print('iteration [{}]; kl:{}; elapsed: {}secs'.format(i,
                                        round(kl, 6),
                                        round(time_vec[-1] - time_vec[-2], 4)))

            kl_diff = kl_old - kl
            assert kl_diff > 0

            kl_old = kl

        x_diff = vb_params_paragami.flatten(vb_params_dict, free = True) - x_old

        if np.abs(x_diff).max() < x_tol:
            print('CAVI done. Termination after {} steps in {} seconds'.format(
                    i, round(time.time() - t0, 2)))
            break

        x_old = vb_params_paragami.flatten(vb_params_dict, free = True)

    if i == (max_iter - 1):
        print('Done. Warning, max iterations reached. ')

    vb_opt = vb_params_paragami.flatten(vb_params_dict, free = True)
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
                                    stick_mean_free_params,
                                    stick_info_free_params,
                                    vb_params_paragami,
                                    prior_params_dict,
                                    gh_loc, gh_weights):
    # this function only returns the terms in the loss that depend on the
    # logitnormal sticks. The gradients wrt to the sticks are correct,
    # though the loss itself is not

    stick_mean = vb_params_paragami['ind_mix_stick_propn_mean'].\
                    fold(stick_mean_free_params, free = True)
    stick_info = vb_params_paragami['ind_mix_stick_propn_info'].\
                    fold(stick_info_free_params, free = True)

    e_log_sticks, e_log_1m_sticks = \
            ef.get_e_log_logitnormal(
                lognorm_means = stick_mean,
                lognorm_infos = stick_info,
                gh_loc = gh_loc,
                gh_weights = gh_weights)

    e_log_cluster_probs = \
        modeling_lib.get_e_log_cluster_probabilities_from_e_log_stick(
                            e_log_sticks, e_log_1m_sticks)

    dp_prior = (prior_params_dict['dp_prior_alpha'] - 1) * np.sum(e_log_1m_sticks)
    stick_entropy = \
            modeling_lib.get_stick_breaking_entropy(
                                    stick_mean,
                                    stick_info,
                                    gh_loc, gh_weights)

    n = e_z.shape[0]
    k = e_z.shape[2]

    loglik_term = (e_log_cluster_probs.reshape(n, 1, k, 1) * e_z).sum()

    return - (loglik_term + dp_prior + stick_entropy)

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

def update_logitnormal_sticks(g_obs, e_z, stick_mean, stick_info,
                                vb_params_paragami, prior_params_dict,
                                gh_loc, gh_weights):

    # we use a logitnormal approximation to the sticks : thus, updates
    # can't be computed in closed form. We take a gradient step satisfying wolfe conditions

    # initial parameters
    init_stick_mean_free = vb_params_paragami['ind_mix_stick_propn_mean'].\
                                flatten(stick_mean, free = True)
    init_stick_info_free = vb_params_paragami['ind_mix_stick_propn_info'].\
                                flatten(stick_info, free = True)

    # initial loss
    init_ps_loss = _get_logitnormal_sticks_psloss(g_obs, e_z,
                                        init_stick_mean_free,
                                        init_stick_info_free,
                                        vb_params_paragami,
                                        prior_params_dict,
                                        gh_loc, gh_weights)
    # get gradient
    get_grad_stick_mean = autograd.grad(_get_logitnormal_sticks_psloss, argnum = 2)
    get_grad_stick_info = autograd.grad(_get_logitnormal_sticks_psloss, argnum = 3)

    grad_stick_mean = get_grad_stick_mean(g_obs, e_z,
                                        init_stick_mean_free,
                                        init_stick_info_free,
                                        vb_params_paragami,
                                        prior_params_dict,
                                        gh_loc, gh_weights)

    grad_stick_info = get_grad_stick_info(g_obs, e_z,
                                        init_stick_mean_free,
                                        init_stick_info_free,
                                        vb_params_paragami,
                                        prior_params_dict,
                                        gh_loc, gh_weights)

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

        kl_new = _get_logitnormal_sticks_psloss(g_obs, e_z,
                                        update_stick_mean_free,
                                        update_stick_info_free,
                                        vb_params_paragami,
                                        prior_params_dict,
                                        gh_loc, gh_weights)

        counter += 1

        if counter > 10:
            print('could not find stepsize for stick optimizer')
            break

    update_stick_mean = vb_params_paragami['ind_mix_stick_propn_mean'].\
                            fold(update_stick_mean_free, free = True)

    update_stick_info = vb_params_paragami['ind_mix_stick_propn_info'].\
                            fold(update_stick_info_free, free = True)

    return update_stick_mean, update_stick_info

###############
# functions to run stochastic VI
###############
def update_local_params(g_obs_sampled,
                            e_log_sticks_sampled,
                            e_log_1m_sticks_sampled,
                            pop_beta_params,
                            prior_params_dict,
                            e_log_pop_freq = None,
                            e_log_1m_pop_freq = None,
                            x_tol = 1e-2,
                            maxiter = 100,
                            debug = False):


    # prior parameters
    dp_prior_alpha = prior_params_dict['dp_prior_alpha']
    allele_prior_alpha = prior_params_dict['allele_prior_alpha']
    allele_prior_beta = prior_params_dict['allele_prior_beta']

    kl_old = 1e16
    x_old = 1e16

    if (e_log_pop_freq is None) or (e_log_1m_pop_freq is None):
        e_log_pop_freq, e_log_1m_pop_freq = \
            modeling_lib.get_e_log_beta(pop_beta_params)

    for i in range(maxiter):
        # update e_z
        e_z_sampled = update_z(g_obs_sampled, e_log_sticks_sampled, \
                                e_log_1m_sticks_sampled,
                                e_log_pop_freq,
                                e_log_1m_pop_freq)

        # update individual sticks
        e_log_sticks_sampled, e_log_1m_sticks_sampled, \
            stick_beta_params_sampled = \
                update_stick_beta(g_obs_sampled, e_z_sampled,
                                    e_log_pop_freq, e_log_1m_pop_freq,
                                    dp_prior_alpha, allele_prior_alpha,
                                    allele_prior_beta)

        if debug:
            # get kl
            joint_log_lik = \
                structure_model_lib.get_e_joint_loglik_from_nat_params(g_obs_sampled, \
                                e_z_sampled,
                                e_log_pop_freq, e_log_1m_pop_freq,
                                e_log_sticks_sampled, e_log_1m_sticks_sampled,
                                dp_prior_alpha, allele_prior_alpha,
                                allele_prior_beta)

            entropy = structure_model_lib.get_entropy(None, None,
                                pop_beta_params,
                                e_z_sampled, None, None,
                                use_logitnormal_sticks = False,
                                ind_mix_stick_beta_params = stick_beta_params_sampled).squeeze()

            kl = - joint_log_lik - entropy

            kl_diff = kl_old - kl
            assert kl_diff > 0, kl_diff

            kl_old = kl

        x_diff = stick_beta_params_sampled - x_old
        if np.abs(x_diff).max() < x_tol:
            break

        x_old = stick_beta_params_sampled

    return e_z_sampled, e_log_sticks_sampled, \
                e_log_1m_sticks_sampled, stick_beta_params_sampled

def run_svi(g_obs, vb_params_dict,
                prior_params_dict,
                e_z,
                use_logitnormal_sticks,
                batchsize,
                kappa = 0.9,
                x_tol = 1e-3,
                max_iter = 1000,
                print_every = 1,
                debug_local_updates = False,
                local_x_tol = 1e-3):
    """
    Runs stochastic coordinate ascent on the VB parameters. This is only implemented
    for the beta approximation to the stick-breaking distribution.

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

    if use_logitnormal_sticks:
        raise NotImplementedError()

    # get prior parameters
    dp_prior_alpha = prior_params_dict['dp_prior_alpha']
    allele_prior_alpha = prior_params_dict['allele_prior_alpha']
    allele_prior_beta = prior_params_dict['allele_prior_beta']

    n_obs = g_obs.shape[0]
    obs_weights = np.ones(batchsize) * n_obs / batchsize

    t0 = time.time()

    # initial population moments
    e_log_pop_freq, e_log_1m_pop_freq = \
        modeling_lib.get_e_log_beta(vb_params_dict['pop_freq_beta_params'])

    kl_vec = []
    time_vec = [t0]
    x_old = 1e16

    for i in range(1, max_iter):

        # sample individual
        indx = np.random.choice(n_obs, batchsize, replace = False)

        g_obs_sampled = g_obs[indx]

        # inital moments for sampled individual
        stick_beta_params_sampled = vb_params_dict['ind_mix_stick_beta_params'][indx]
        e_log_sticks_sampled, e_log_1m_sticks_sampled = \
            modeling_lib.get_e_log_beta(stick_beta_params_sampled)

        # update local parameters
        e_z_sampled, e_log_sticks_sampled, \
            e_log_1m_sticks_sampled, stick_beta_params_sampled = \
                update_local_params(g_obs_sampled,
                                e_log_sticks_sampled, e_log_1m_sticks_sampled,
                                vb_params_dict['pop_freq_beta_params'],
                                prior_params_dict,
                                e_log_pop_freq = e_log_pop_freq,
                                e_log_1m_pop_freq = e_log_1m_pop_freq,
                                debug = debug_local_updates,
                                x_tol = local_x_tol)

        e_z[indx] = e_z_sampled
        vb_params_dict['ind_mix_stick_beta_params'][indx] = \
                stick_beta_params_sampled

        # update population frequencies
        pop_beta_params_new = \
            update_pop_beta(g_obs_sampled, e_z_sampled, e_log_sticks_sampled, \
                                e_log_1m_sticks_sampled,
                                dp_prior_alpha, allele_prior_alpha,
                                allele_prior_beta,
                                obs_weights = obs_weights,
                                return_moments = False)

        rho = (i + 1)**(-kappa)
        vb_params_dict['pop_freq_beta_params'] = \
            pop_beta_params_new * rho + \
                        vb_params_dict['pop_freq_beta_params'] * (1 - rho)

        e_log_pop_freq, e_log_1m_pop_freq = \
            modeling_lib.get_e_log_beta(vb_params_dict['pop_freq_beta_params'])

        if (i % print_every) == 0:
            kl = structure_model_lib.get_kl(g_obs, vb_params_dict, prior_params_dict,
                                use_logitnormal_sticks,
                                e_z = e_z)
            kl_vec.append(kl)
            time_vec.append(time.time())

            print('iteration [{}]; kl:{}; elapsed: {}secs'.format(i,
                                        round(kl, 6),
                                        round(time_vec[-1] - time_vec[-2], 4)))

        x_diff = vb_params_dict['pop_freq_beta_params'] - x_old

        if np.abs(x_diff).max() < x_tol:
            print('SVI done. Termination after {} steps in {} seconds'.format(
                    i, round(time.time() - t0, 2)))
            break

        x_old = vb_params_dict['pop_freq_beta_params']

    if i == (max_iter - 1):
        print('Done. Warning, max iterations reached. ')

    return e_z, vb_params_dict, np.array(kl_vec), np.array(time_vec) - t0
