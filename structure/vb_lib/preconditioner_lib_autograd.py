import numpy as np
import scipy as sp

from scipy import sparse

import paragami

from copy import deepcopy

import warnings

from vb_lib import structure_model_lib

from paragami.optimization_lib import _get_sym_matrix_inv_sqrt_funcs, \
                                            _get_matrix_from_operator

def get_log_beta_covariance(alpha, beta):
    # returns the covariance of the score function
    # of the beta distribution

    digamma_sum = sp.special.polygamma(1, alpha + beta)

    # get Fisher's information matrix
    I11 = sp.special.polygamma(1, alpha) - digamma_sum # var(log x)
    I22 = sp.special.polygamma(1, beta) - digamma_sum # var(log(1 - x))
    I12 = - digamma_sum # cov(log x, log(1 - x))

    # mulitply by alphas and betas because we are using
    # an unconstrained parameterization, where log(alpha) = free_param
    # TODO: better way to do this using autodiff?
    return np.array([[I11 * alpha**2, I12 * alpha * beta], \
                     [I12 * alpha * beta, I22 * beta**2]])

def _get_cov_for_all_beta_params(vb_beta_params, return_info):
    block_cov = ()
    for i in range(len(vb_beta_params) // 2):
        # get covariance
        cov = get_log_beta_covariance(vb_beta_params[2 * i],
                                        vb_beta_params[2 * i + 1])

        if return_info:
            cov = np.linalg.inv(cov)

        block_cov = block_cov + (cov, )

    return block_cov

def get_mfvb_cov(vb_params_dict, vb_params_paragami,
                    use_logitnormal_sticks,
                    return_info = False):
    # compute preconditioner from MFVB covariances

    block_mfvb_cov = ()

    ##############
    # blocks for the population frequency
    vb_params_pop_params = np.array(vb_params_paragami['pop_freq_beta_params'].flatten(\
                        vb_params_dict['pop_freq_beta_params'], free = False))

    block_mfvb_cov = block_mfvb_cov + \
                    _get_cov_for_all_beta_params(vb_params_pop_params, return_info)

    #############
    # blocks for individual admixture
    if use_logitnormal_sticks:
        infos = np.array(vb_params_paragami['ind_mix_stick_propn_info'].flatten(
                        vb_params_dict['ind_mix_stick_propn_info'],
                        free = False))
        if return_info:
            block_mfvb_cov = block_mfvb_cov + (np.diag(1/infos), ) + (np.eye(len(infos)) * 2., )
        else:
            block_mfvb_cov = block_mfvb_cov + (np.diag(infos), ) + (np.eye(len(infos)) * 0.5, )
    else:
        vb_params_admix = np.array(vb_params_paragami['ind_mix_stick_beta_params'].flatten(\
                            vb_params_dict['ind_mix_stick_beta_params'], free = False))

        block_mfvb_cov = block_mfvb_cov + \
                        _get_cov_for_all_beta_params(vb_params_admix, return_info)

    return sparse.block_diag(block_mfvb_cov)
