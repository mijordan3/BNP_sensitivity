import autograd
import autograd.numpy as np
import autograd.scipy as sp

from scipy import sparse

import paragami

from copy import deepcopy

import structure_model_lib

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

def get_mfvb_covariance(vb_params_dict, vb_params_paragami,
                        use_logitnormal_sticks):
    # get (constrained parameters):
    vb_params = vb_params_paragami.flatten(vb_params_dict, free = False)

    if use_logitnormal_sticks:
        # TODO:
        raise NotImplementedError()

    block_covs = ()
    for i in range(len(vb_params) // 2):
        cov = get_log_beta_covariance(vb_params[2 * i], vb_params[2 * i + 1])
        block_covs = block_covs + (cov, )

    return sparse.block_diag(block_covs).toarray()
