import autograd

import autograd.numpy as np
import autograd.scipy as sp

import vittles

def check_hessian(vb_sens, which_prior):
    """
    If L(theta) := H^{-1}S, the sensitivity matrix, the computes the
    derivative of L(theta) in the direction of the next Newton step.

    which_prior is a boolean vector specifying the index of the prior parameter
    for which sensitivity will be computed
    """

    vb_opt_free_params = vb_sens._opt0
    prior_free_params = vb_sens._hyper0
    assert len(which_prior) == len(prior_free_params)

    # get hessian inverse
    hess_inverse = vb_sens.hess_solver.solve(np.eye(len(vb_opt_free_params)))

    # gradient wrt to vb free parameters
    vb_opt_grad = vb_sens._obj_fun_grad(vb_opt_free_params, prior_free_params)

    # get the direction of the next Newton step
    step = - vb_sens.hess_solver.solve(vb_opt_grad)

    # This is the second term of the derivative
    def get_term2_objective(vb_sens, vb_free_params,
                                prior_free_params, which_prior):
        cross_hess = vb_sens._hyper_obj_cross_hess(vb_free_params, \
                                prior_free_params)

        cross_hess_prior = np.einsum('nj, j -> n', cross_hess, which_prior)

        return -np.einsum('nj, j -> n', hess_inverse, cross_hess_prior)

    get_term2_derivative = \
                vittles.sensitivity_lib._append_jvp(get_term2_objective,
                                                       num_base_args = 4,
                                                       argnum = 1)

    term2_derivative = get_term2_derivative(vb_sens, vb_opt_free_params,
                                        prior_free_params, which_prior, step)

    # this is the third term of the derivative
    def get_term3_objective(vb_sens, vb_free_params,
                                prior_free_params):
        vb_grad = vb_sens._obj_fun_grad(vb_free_params, prior_free_params)
        return np.dot(hess_inverse, vb_grad)

    _get_term3_derivataive = \
        vittles.sensitivity_lib._append_jvp(get_term3_objective,
                                   num_base_args = 3,
                                   argnum = 1)

    get_term3_derivative = \
        vittles.sensitivity_lib._append_jvp(_get_term3_derivataive,
                                   num_base_args = 4,
                                   argnum = 1)
    sens_mat = np.einsum('nj, j -> n', vb_sens._sens_mat, which_prior)

    term3_derivative = \
        get_term3_derivative(vb_sens, vb_opt_free_params, prior_free_params,
                                        sens_mat, step)

    return term2_derivative + term3_derivative
