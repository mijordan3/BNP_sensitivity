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
                    e_log_pop_freq, e_log_1m_pop_freq,
                    e_log_sticks, e_log_1m_sticks,
                    dp_prior_alpha, allele_prior_alpha,
                    allele_prior_beta):
    # update population frequency parameters

    beta_param1 = get_pop_beta_update1(g_obs, e_z,
                    e_log_pop_freq, e_log_1m_pop_freq,
                    e_log_sticks, e_log_1m_sticks,
                    dp_prior_alpha, allele_prior_alpha,
                    allele_prior_beta) + 1.0
    beta_param2 = get_pop_beta_update2(g_obs, e_z,
                    e_log_pop_freq, e_log_1m_pop_freq,
                    e_log_sticks, e_log_1m_sticks,
                    dp_prior_alpha, allele_prior_alpha,
                    allele_prior_beta) + 1.0

    beta_params = np.concatenate((beta_param1[:, :, None],
                                beta_param2[:, :, None]), axis = 2)
    e_log_pop_freq, e_log_1m_pop_freq = modeling_lib.get_e_log_beta(beta_params)

    return e_log_pop_freq, e_log_1m_pop_freq, beta_params

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
                prior_params_dict,
                use_logitnormal_sticks,
                f_tol = 1e-6,
                max_iter = 1000,
                print_every = 1):
    """
    Runs coordinate ascent on the VB parameters. This is only implemented
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
    vb_opt_dict : dictionary
        A dictionary that contains the optimized variational parameters.
    """

    vb_opt_dict = deepcopy(vb_params_dict)

    # get prior parameters
    dp_prior_alpha = prior_params_dict['dp_prior_alpha']
    allele_prior_alpha = prior_params_dict['allele_prior_alpha']
    allele_prior_beta = prior_params_dict['allele_prior_beta']

    # get initial moments from vb_params
    e_log_sticks, e_log_1m_sticks, \
        e_log_pop_freq, e_log_1m_pop_freq = \
            structure_model_lib.get_moments_from_vb_params_dict(g_obs, \
                                    vb_params_dict, use_logitnormal_sticks)

    kl_old = np.Inf

    t0 = time.time()
    for i in range(1, max_iter):
        # update z
        e_z = update_z(g_obs, e_log_sticks, e_log_1m_sticks, e_log_pop_freq,
                                e_log_1m_pop_freq)

        # update individual admixtures
        e_log_sticks, e_log_1m_sticks, stick_beta_params = \
            update_stick_beta(g_obs, e_z,
                                e_log_pop_freq, e_log_1m_pop_freq,
                                dp_prior_alpha, allele_prior_alpha,
                                allele_prior_beta)

        # update population frequencies
        e_log_pop_freq, e_log_1m_pop_freq, pop_beta_params = \
            update_pop_beta(g_obs, e_z,
                            e_log_pop_freq, e_log_1m_pop_freq,
                            e_log_sticks, e_log_1m_sticks,
                            dp_prior_alpha, allele_prior_alpha,
                            allele_prior_beta)

        # get kl:
        joint_log_lik = \
            structure_model_lib.get_e_joint_loglik_from_nat_params(g_obs, e_z,
                            e_log_pop_freq, e_log_1m_pop_freq,
                            e_log_sticks, e_log_1m_sticks,
                            dp_prior_alpha, allele_prior_alpha,
                            allele_prior_beta)
        entropy = structure_model_lib.get_entropy(None, None,
                            pop_beta_params,
                            e_z, None, None,
                            use_logitnormal_sticks = False,
                            ind_mix_stick_beta_params = stick_beta_params).squeeze()

        kl = - joint_log_lik - entropy

        if (i % print_every) == 0:
            print('iteration [{}]; kl:{}'.format(i, round(kl, 6)))

        kl_diff = kl_old - kl
        assert kl_diff > 0

        if kl_diff < f_tol:
            print('CAVI done. Termination after {} steps in {} seconds'.format(
                    i, round(time.time() - t0, 2)))
            break

        kl_old = kl

    if i == (max_iter - 1):
        print('Done. Warning, max iterations reached. ')

    # Set VB parameters
    if use_logitnormal_sticks:
        # convert beta params to logitnormal
        raise NotImplementedError()
    else:
        vb_opt_dict['pop_freq_beta_params'] = pop_beta_params
        vb_opt_dict['ind_mix_stick_beta_params'] = stick_beta_params

    return e_z, vb_opt_dict
