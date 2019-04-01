import autograd
import autograd.numpy as np
import autograd.scipy as sp

from scipy import sparse

import paragami

from copy import deepcopy

import structure_model_lib

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

def get_mfvb_cov_preconditioner(vb_params_dict, vb_params_paragami,
                        use_logitnormal_sticks):
    # get (constrained parameters):
    vb_params = vb_params_paragami.flatten(vb_params_dict, free = False)

    if use_logitnormal_sticks:
        # TODO:
        raise NotImplementedError()

    block_mfvb_cov = ()
    block_mfvb_info = ()
    for i in range(len(vb_params) // 2):
        # get covariance
        cov = get_log_beta_covariance(vb_params[2 * i], vb_params[2 * i + 1])

        # take matrix square root
        mfvb_cov_fun, mfvb_info_fun = \
            _get_sym_matrix_inv_sqrt_funcs(cov)
        mfvb_cov_mat = _get_matrix_from_operator(mfvb_cov_fun, dim = 2)
        mfvb_info_mat = _get_matrix_from_operator(mfvb_info_fun, dim = 2)

        # append to block
        block_mfvb_cov = block_mfvb_cov + (mfvb_cov_mat, )
        block_mfvb_info = block_mfvb_info + (mfvb_info_mat, )

    return sparse.block_diag(block_mfvb_cov),\
                sparse.block_diag(block_mfvb_info)
