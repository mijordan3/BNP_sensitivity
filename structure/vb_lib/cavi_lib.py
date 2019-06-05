import autograd

import autograd.numpy as np
import autograd.scipy as sp

from vb_lib import structure_model_lib

from bnpmodeling_runjingdev import cluster_quantities_lib, modeling_lib

import LinearResponseVariationalBayes.ExponentialFamilies as ef

import time

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
                    e_log_sticks, e_log_1m_sticks,
                    dp_prior_alpha, allele_prior_alpha,
                    allele_prior_beta):

    # update individual admixtures

    beta_param1 = get_stick_update1(g_obs, e_z,
                e_log_pop_freq, e_log_1m_pop_freq,
                e_log_sticks, e_log_1m_sticks,
                dp_prior_alpha, allele_prior_alpha,
                allele_prior_beta) + 1.0

    beta_param2 = get_stick_update2(g_obs, e_z,
                    e_log_pop_freq, e_log_1m_pop_freq,
                    e_log_sticks, e_log_1m_sticks,
                    dp_prior_alpha, allele_prior_alpha,
                    allele_prior_beta) + 1.0

    beta_params = np.concatenate((beta_param1[:, :, None],
                                    beta_param2[:, :, None]), axis = 2)

    e_log_sticks, e_log_1m_sticks = modeling_lib.get_e_log_beta(beta_params)

    return e_log_sticks, e_log_1m_sticks, beta_params

def run_cavi(g_obs, e_log_pop_freq, e_log_1m_pop_freq,
                e_log_sticks, e_log_1m_sticks,
                prior_params_dict,
                f_tol = 1e-6,
                max_iter = 1000):
    """
    Runs coordinate ascent on the VB parameters. This is only implemented
    for the beta approximation to the stick-breaking distribution.

    Parameters
    ----------
    g_obs : ndarray
        Array of size (n_obs x n_loci x 3), giving a one-hot encoding of
        genotypes
    e_log_pop_freq : ndarray
        Array of size n_loci x n_pop specifying the expected log
        population frequencies
    e_log_1m_pop_freq : ndarray
        Array of size n_loci x n_pop specifying the expected
        log(1 - population frequency)
    e_log_sticks : ndarray
        Array of size n_obs x n_pop specifying the expected
        log sticks of the individual admixtures
     e_log_1m_sticks : ndarray
         Array of size n_obs x n_pop specifying the expected
         log(1 - sticks) of the individual admixtures
    prior_params_dict : dictionary
        A dictionary that contains the prior parameters.

    Returns
    -------
    e_z : ndarray
        Array of size (n_obs x n_loci x k_approx x 2)
        specifying the expected belonging of the nth individual
        at the mth loci belonging to the kth population of either
        chromosome.
    stick_beta_params : ndarray
        Array of size (n_obs x k_approx x 2)
        specifying the beta parameters of the individual admixture
        stick-breaking disribution
    pop_beta_params : ndarray
        Array of size (n_loci x k_approx x 2)
        specifying the beta parameters of the population
        allele frequencies
    """


    # get prior parameters
    dp_prior_alpha = prior_params_dict['dp_prior_alpha']
    allele_prior_alpha = prior_params_dict['allele_prior_alpha']
    allele_prior_beta = prior_params_dict['allele_prior_beta']

    kl_old = np.Inf

    t0 = time.time()
    for i in range(max_iter):
        # update z
        e_z = update_z(g_obs, e_log_sticks, e_log_1m_sticks, e_log_pop_freq,
                                e_log_1m_pop_freq)

        # update individual admixtures
        e_log_sticks, e_log_1m_sticks, stick_beta_params = \
            update_stick_beta(g_obs, e_z,
                                e_log_pop_freq, e_log_1m_pop_freq,
                                e_log_sticks, e_log_1m_sticks,
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

    return e_z, stick_beta_params, pop_beta_params
