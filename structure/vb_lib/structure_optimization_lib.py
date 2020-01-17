import autograd

import autograd.numpy as np
import autograd.scipy as sp

import paragami
import vittles

from copy import deepcopy
import time

from vb_lib import structure_model_lib, cavi_lib

from vb_lib.preconditioner_lib import get_mfvb_cov_preconditioner

import bnpmodeling_runjingdev.optimization_lib as optim_lib


def check_hessian(vb_sens, which_prior):
    """
    If L(theta) := H^{-1}S, the sensitivity matrix, we compute here the
    derivative of L(theta) in the direction of the next Newton step.

    `which_prior` is a boolean vector specifying the index of the prior
    parameter for which sensitivity will be computed
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

    _get_term3_derivative = \
        vittles.sensitivity_lib._append_jvp(get_term3_objective,
                                   num_base_args = 3,
                                   argnum = 1)

    get_term3_derivative = \
        vittles.sensitivity_lib._append_jvp(_get_term3_derivative,
                                   num_base_args = 4,
                                   argnum = 1)
    sens_mat = np.einsum('nj, j -> n', vb_sens._sens_mat, which_prior)

    term3_derivative = \
        get_term3_derivative(vb_sens, vb_opt_free_params, prior_free_params,
                                        -sens_mat, step)

    return term2_derivative + term3_derivative

##################################
# FUNCTION TO OPTIMIZE STRUCTURE
##################################

def optimize_structure(g_obs, vb_params_dict, vb_params_paragami,
                        prior_params_dict,
                        gh_loc, gh_weights,
                        use_logitnormal_sticks = True,
                        run_cavi = True,
                        cavi_max_iter = 50,
                        cavi_tol = 1e-6,
                        netwon_max_iter = 50,
                        max_precondition_iter = 10,
                        gtol=1e-8,
                        ftol=1e-8,
                        xtol=1e-8,
                        approximate_preconditioner = False):
    """
    Optimizes the structure model using a combination of CAVI
    and preconditioned Newton's method.

    Parameters
    ----------
    g_obs : ndarray
        Array of size (n_obs x n_loci x 3), giving a one-hot encoding of
        genotypes
    vb_params_dict : dictionary
        Dictionary of variational parameters.
    prior_params_dict : dictionary
        Dictionary of prior parameters.
    use_logitnormal_sticks : boolean
        Whether to use a logitnormal approximation to infer the sticks.
    gh_loc : vector
        Locations for gauss-hermite quadrature. We need this compute the
        expected prior terms.
    gh_weights : vector
        Weights for gauss-hermite quadrature. We need this compute the
        expected prior terms.
    run_cavi : boolean
        Whether to run CAVI steps before the Newton steps.
    cavi_max_iter : int
        Maximum interations to run CAVI
    cavi_tol : float
        tolerance on difference of objective function to terminate CAVI
    netwon_max_iter : int
        Maximum interations to run Newton
    max_precondition_iter : int
        Maximum times to recompute preconditioner
    approximate_preconditioner : boolean
        whether to approximate the preconditioner using MFVB covariance.
        if False, the Hessian is computed and used as a preconditioner

    Parameters
    ----------
    The optimal free variational parameters

    """

    # get loss as a function of vb free parameters
    get_free_vb_params_loss = paragami.FlattenFunctionInput(
                                original_fun=structure_model_lib.get_kl,
                                patterns = vb_params_paragami,
                                free = True,
                                argnums = 1)

    get_loss = \
        lambda x : get_free_vb_params_loss(g_obs,
                                    x, prior_params_dict,
                                    use_logitnormal_sticks,
                                    gh_loc, gh_weights)

    get_loss_grad = autograd.grad(get_loss)

    if run_cavi:
        # RUN CAVI
        print('running CAVI. ')
        vb_params_dict = cavi_lib.run_cavi(g_obs, vb_params_dict,
                        prior_params_dict, use_logitnormal_sticks,
                        max_iter = cavi_max_iter,
                        f_tol = cavi_tol)

    x = vb_params_paragami.flatten(vb_params_dict, free = True)
    f_val = get_loss(x)

    for i in range(max_precondition_iter):
        print('\n running preconditioned newton; iter = ', i)

        if approximate_preconditioner:
            # get approximate preconditioner using MFVB covariance

            t0 = time.time()

            a_inv, a = \
                get_mfvb_cov_preconditioner( \
                                vb_params_dict, vb_params_paragami,
                                 use_logitnormal_sticks)

            print('approximate preconditioner time: ' +
                    '{:3f} secs'.format(time.time() - t0))

        new_x, ncg_output = optim_lib.precondition_and_optimize(get_loss, x,\
                                    maxiter = netwon_max_iter, gtol = gtol,
                                    preconditioner = (a, a_inv))

        # Check convergence.
        new_f_val = get_loss(new_x)
        grad_val = get_loss_grad(new_x)

        x_diff = np.sum(np.abs(new_x - x))
        f_diff = np.abs(new_f_val - f_val)
        grad_l1 = np.sum(np.abs(grad_val))
        x_conv = x_diff < xtol
        f_conv = f_diff < ftol
        grad_conv = grad_l1 < gtol

        x = new_x
        f_val = new_f_val
        vb_params_dict = vb_params_paragami.fold(x, free = True)

        converged = x_conv or f_conv or grad_conv or ncg_output.success

        print('Iter {}: x_diff = {}, f_diff = {}, grad_l1 = {}'.format(
            i, x_diff, f_diff, grad_l1))

        if converged:
            print('done. ')
            break

    return new_x

def get_e_z_from_vb_params_dict(g_obs, vb_params_dict, use_logitnormal_sticks):
    e_log_sticks, e_log_1m_sticks, \
        e_log_pop_freq, e_log_1m_pop_freq = \
            structure_model_lib.get_moments_from_vb_params_dict(g_obs, \
                                    vb_params_dict, use_logitnormal_sticks)

    return cavi_lib.update_z(g_obs, e_log_sticks, e_log_1m_sticks, e_log_pop_freq,
                                e_log_1m_pop_freq)
