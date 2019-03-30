import autograd

import autograd.numpy as np
import autograd.scipy as sp

import paragami
import vittles

from copy import deepcopy
import time

import structure_model_lib
import cavi_lib

import BNP_modeling.optimization_lib as optim_lib

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

###########################
# THESE FUNCTIONS ARE FOR COMPUTING AN APPROXIMATE HESSIAN
# USING THE MFVB COVARIANCE
###########################
def get_log_beta_density(tau, e_log_beta, e_log_1m_beta):
    # tau should have shape (..., 2). The last dimensions are the
    # beta parameters
    assert tau.shape[-1] == 2
    assert tau[..., 0].shape == e_log_beta.shape
    assert tau[..., 1].shape == e_log_1m_beta.shape

    return (tau[..., 0] - 1) * e_log_beta + (tau[:, :, 1] - 1) * e_log_1m_beta

def _score_objective(vb_params_dict,
                        e_log_pop_freq, e_log_1m_pop_freq,
                        e_logit_sticks, e_logit_sticks2,
                        use_logitnormal_sticks = False,
                        e_log_sticks = None,
                        e_log_1m_sticks = None):

    # A derivative of this with respect to vb_params_dict (flattened)
    # returns the the score function

    if use_logitnormal_sticks:
        raise NoneImplementedError()

    else:
        assert e_log_sticks is not None
        assert e_log_1m_sticks is not None

        ind_mix_stick_beta_params = vb_params_dict['ind_mix_stick_beta_params']
        ind_sticks_log_q = get_log_beta_density(ind_mix_stick_beta_params,
                                                e_log_sticks,
                                                e_log_1m_sticks).sum()

    pop_freq_beta_params = vb_params_dict['pop_freq_beta_params']
    pop_freq_log_q = get_log_beta_density(pop_freq_beta_params,
                                          e_log_pop_freq,
                                          e_log_1m_pop_freq).sum()

    return ind_sticks_log_q + pop_freq_log_q

def _mfvb_covariance_objective(g_obs,
                                vb_params_dict, vb_params_paragami,
                                vb_free_params_opt,
                                use_logitnormal_sticks = True):
    # a derivative with respect to vb_params_dict
    # gives the mfvb covariance

    # we do not take a derivative wrt vb_free_params_opt!

    # Get expected sufficient statistics
    if use_logitnormal_sticks:
        raise NotImplementedError()
    else:
        e_log_sticks, e_log_1m_sticks, \
            e_log_pop_freq, e_log_1m_pop_freq = \
                structure_model_lib.get_moments_from_vb_params_dict(
                                        g_obs, vb_params_dict,
                                        use_logitnormal_sticks = False,
                                        gh_loc = None,
                                        gh_weights = None)
        e_logit_sticks = None
        e_logit_sticks2 = None

    # get score function
    score_objective_flattened = \
        paragami.FlattenFunctionInput(original_fun=_score_objective,
                                    patterns = vb_params_paragami,
                                    free = True,
                                    argnums = 0)

    score_function = autograd.grad(score_objective_flattened, argnum = 0)

    return score_function(vb_free_params_opt,
                            e_log_pop_freq, e_log_1m_pop_freq,
                            e_logit_sticks = e_logit_sticks,
                            e_logit_sticks2 = e_logit_sticks2,
                            use_logitnormal_sticks = use_logitnormal_sticks,
                            e_log_sticks = e_log_sticks,
                            e_log_1m_sticks = e_log_1m_sticks)

def get_mfvb_covariance(g_obs, vb_params_dict, vb_params_paragami,
                                vb_free_params_opt,
                                use_logitnormal_sticks = True):

    mfvb_covariance_objective_flattened = \
        paragami.FlattenFunctionInput(original_fun=_mfvb_covariance_objective,
                                        patterns = vb_params_paragami,
                                        free = True,
                                        argnums = 1)

    get_mfvb_covariance_fun = autograd.jacobian(mfvb_covariance_objective_flattened,
                                                argnum = 1)

    return get_mfvb_covariance_fun(g_obs,
                                deepcopy(vb_free_params_opt),
                                vb_params_paragami,
                                vb_free_params_opt,
                                use_logitnormal_sticks = use_logitnormal_sticks)

##################################
# FUNCTION TO OPTIMIZE STRUCTURE
##################################

def optimize_structure(g_obs, vb_params_dict, vb_params_paragami,
                    prior_params_dict,
                    gh_loc, gh_weights, use_logitnormal_sticks = True,
                    run_cavi = True, cavi_max_iter = 50, cavi_tol = 1e-6,
                    netwon_max_iter = 50,
                    max_precondition_iter = 10,
                    gtol=1e-8, ftol=1e-8, xtol=1e-8,
                    approximate_hessian = False):

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

        # get initial moments
        e_log_sticks, e_log_1m_sticks, \
            e_log_pop_freq, e_log_1m_pop_freq = \
                structure_model_lib.get_moments_from_vb_params_dict(g_obs, \
                                        vb_params_dict, use_logitnormal_sticks)

        # get beta parameters for sticks and populations
        _, stick_beta_params, pop_beta_params = \
            cavi_lib.run_cavi(g_obs, e_log_pop_freq, e_log_1m_pop_freq,
                                e_log_sticks, e_log_1m_sticks,
                                prior_params_dict,
                                max_iter = cavi_max_iter,
                                f_tol = cavi_tol)

        # Set VB parameters
        if use_logitnormal_sticks:
            # convert beta params to logitnormal
            raise NotImplementedError()
        else:
            vb_params_dict['pop_freq_beta_params'] = pop_beta_params
            vb_params_dict['ind_mix_stick_beta_params'] = stick_beta_params

    x = vb_params_paragami.flatten(vb_params_dict, free = True)
    f_val = get_loss(x)

    for i in range(max_precondition_iter):
        print('\n running preconditioned newton; iter = ', i)

        if approximate_hessian:
            t0 = time.time()

            init_hessian = get_mfvb_covariance(g_obs,
                                vb_params_dict, vb_params_paragami, x,
                                use_logitnormal_sticks = use_logitnormal_sticks)

            print('approximate preconditioner time: ' +
                    '{:3f} secs'.format(time.time() - t0))
        else:
            init_hessian = None

        new_x, ncg_output = optim_lib.precondition_and_optimize(get_loss, x,\
                                    maxiter = netwon_max_iter, gtol = gtol,
                                    hessian = init_hessian)

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
