import jax
import jax.numpy as np
import jax.scipy as sp

import numpy as onp
from scipy import optimize

import paragami
from paragami import OptimizationObjective

import time

def construct_and_compile_derivatives(get_loss, init_vb_free_params):
    get_loss_jitted = jax.jit(get_loss)

    # set up objective
    optim_objective = OptimizationObjective(get_loss_jitted)

    # define derivatives
    optim_grad = jax.jit(optim_objective.grad)
    optim_hvp = jax.jit(optim_objective.hessian_vector_product)

    # compile derivatives
    t0 = time.time()
    print('Compiling derivatives ...')
    _ = optim_objective.f(init_vb_free_params)
    _ = optim_grad(init_vb_free_params)
    _ = optim_hvp(init_vb_free_params, init_vb_free_params)
    print('Compile time: {0:3g}secs'.format(time.time() - t0))

    # convert to numpy
    optim_objective_np = lambda x : onp.array(optim_objective.f(x))
    optim_grad_np = lambda x : onp.array(optim_grad(x))
    optim_hvp_np = lambda x, v : onp.array(optim_hvp(x, v))

    optim_objective.reset()

    return optim_objective, optim_objective_np, optim_grad_np, optim_hvp_np

def optimize_full(get_loss, init_vb_free_params):

    optim_objective, optim_objective_np, optim_grad_np, optim_hvp_np = \
        construct_and_compile_derivatives(get_loss, init_vb_free_params)

    # run l-bfgs-b
    t0 = time.time()
    print('\nRunning L-BFGS-B ... ')
    out = optimize.minimize(optim_objective_np,
                        x0 = onp.array(init_vb_free_params),
                        jac = optim_grad_np,
                        method='L-BFGS-B')
    print('done. Elapsed {0:3g}secs'.format(time.time() - t0))

    return out

# def run_bfgs(get_loss, init_vb_free_params,
#                     maxiter = 10, gtol = 1e-8):
#
#     """
#     Runs BFGS to find the optimal variational parameters
#
#     Parameters
#     ----------
#     get_loss : Callable function
#         A callable function that takes in the variational free parameters
#         and returns the negative ELBO.
#     init_vb_free_params : vector
#         Vector of the free variational parameters at which we initialize the
#         optimization.
#     get_loss_grad : Callable function (optional)
#         A callable function that takes in the variational free parameters
#         and returns the gradient of get_loss.
#     maxiter : int
#         Maximum number of iterations to run bfgs.
#     gtol : float
#         The tolerance used to check that the gradient is approximately
#             zero at the optimum.
#
#     Returns
#     -------
#     bfgs_vb_free_params : vec
#         Vector of optimal variational free parameters.
#     bfgs_output :
#         The OptimizeResult class from returned by scipy.optimize.minimize.
#
#     """
#     get_loss_objective = OptimizationObjective(get_loss)
#
#     # optimize
#     bfgs_output = optimize.minimize(
#             get_loss_objective.f,
#             x0=init_vb_free_params,
#             jac=get_loss_objective.grad,
#             method='BFGS',
#             options={'maxiter': maxiter, 'disp': True, 'gtol': gtol})
#
#     bfgs_vb_free_params = bfgs_output.x
#
#     return bfgs_vb_free_params, bfgs_output
#
# def precondition_and_optimize(get_loss, init_vb_free_params,
#                                 maxiter = 10, gtol = 1e-8,
#                                 hessian = None,
#                                 preconditioner = None):
#     """
#     Finds a preconditioner at init_vb_free_params, and then
#     runs trust Newton conjugate gradient to find the optimal
#     variational parameters.
#
#     Parameters
#     ----------
#     get_loss : Callable function
#         A callable function that takes in the variational free parameters
#         and returns the negative ELBO.
#     init_vb_free_params : vector
#         Vector of the free variational parameters at which we initialize the
#         optimization.
#     get_loss_grad : Callable function (optional)
#         A callable function that takes in the variational free parameters
#         and returns the gradient of get_loss.
#     maxiter : int
#         Maximum number of iterations to run Newton
#     gtol : float
#         The tolerance used to check that the gradient is approximately
#             zero at the optimum.
#
#     Returns
#     -------
#     bfgs_vb_free_params : vec
#         Vector of optimal variational free parameters.
#     bfgs_output : class OptimizeResult from scipy.Optimize
#
#     """
#
#     # get preconditioned function
#     precond_fun = paragami.PreconditionedFunction(get_loss)
#     if (hessian is None) and (preconditioner is None):
#         print('Computing Hessian to set preconditioner')
#         t0 = time.time()
#         _ = precond_fun.set_preconditioner_with_hessian(x = init_vb_free_params,
#                                                             ev_min=1e-4)
#         print('preconditioning time: {0:.2f}'.format(time.time() - t0))
#     elif (hessian is not None):
#         assert preconditioner is None, 'can only specify one of hessian or preconditioner'
#         print('setting preconditioner with given Hessian: ')
#         _ = precond_fun.set_preconditioner_with_hessian(hessian = hessian,
#                                                             ev_min=1e-4)
#     elif (preconditioner is not None):
#         assert hessian is None, 'can only specify one of hessian or preconditioner'
#         print('setting with given preconditioner: ')
#         # preconditioner should be a tuple, where the first entry of preconditioner
#         # is the preconditioner; the second entry is its optional inverse
#         # it is (a, a_inv) in the notation of paragami.optimization_lib
#         precond_fun.set_preconditioner_matrix(preconditioner[0], preconditioner[1])
#
#     # optimize
#     get_loss_precond_objective = OptimizationObjective(precond_fun)
#     print('running newton steps')
#     trust_ncg_output = optimize.minimize(
#                             method='trust-ncg',
#                             x0=precond_fun.precondition(init_vb_free_params),
#                             fun=get_loss_precond_objective.f,
#                             jac=get_loss_precond_objective.grad,
#                             hessp=get_loss_precond_objective.hessian_vector_product,
#                             options={'maxiter': maxiter, 'disp': True, 'gtol': gtol})
#
#     # Uncondition
#     trust_ncg_vb_free_pars = precond_fun.unprecondition(trust_ncg_output.x)
#
#     return trust_ncg_vb_free_pars, trust_ncg_output
#
# def optimize_full(get_loss, init_vb_free_params,
#                     bfgs_max_iter = 50, netwon_max_iter = 50,
#                     max_precondition_iter = 10,
#                     gtol=1e-8, ftol=1e-8, xtol=1e-8,
#                     init_hessian = None):
#     """
#     Finds the optimal variational free parameters of using a combination of
#     BFGS and Newton trust region conjugate gradient.
#
#     Runs a few BFGS steps, and computes a preconditioner at the BFGS optimum.
#     After preconditioning, we run Newton trust region conjugate gradient.
#     If the tolerance is not satisfied after Newton steps, we compute another
#     preconditioner and repeat.
#
#     Parameters
#     ----------
#     get_loss : Callable function
#         A callable function that takes in the variational free parameters
#         and returns the negative ELBO.
#     init_vb_free_params : vector
#         Vector of the free variational parameters at which we initialize the
#     bfgs_max_iter : int
#         Maximum number of iterations to run initial BFGS.
#     newton_max_iter : int
#         Maximum number of iterations to run Newton steps.
#     max_precondition_iter : int
#         Maximum number of times to recompute preconditioner.
#     ftol : float
#         The tolerance used to check that the difference in function value
#         is approximately zero at the last step.
#     xtol : float
#         The tolerance used to check that the difference in x values in the L
#         infinity norm is approximately zero at the last step.
#     gtol : float
#         The tolerance used to check that the gradient is approximately
#             zero at the optimum.
#
#     Returns
#     -------
#     vec
#         A vector of optimal variational free parameters.
#
#     """
#
#     get_loss_grad = autograd.grad(get_loss)
#
#     # run a few steps of bfgs
#     if bfgs_max_iter > 0:
#         print('running bfgs ... ')
#         bfgs_vb_free_params, bfgs_ouput = run_bfgs(get_loss,
#                                     init_vb_free_params,
#                                     maxiter = bfgs_max_iter,
#                                     gtol = gtol)
#         x = bfgs_vb_free_params
#         f_val = get_loss(x)
#
#         bfgs_success = bfgs_ouput.success
#     else:
#         bfgs_success = False
#         x = init_vb_free_params
#         f_val = get_loss(x)
#
#     if bfgs_success:
#         print('bfgs converged. Done. ')
#         return x
#     else:
#         # if bfgs did not converge, we precondition and run newton trust region
#         for i in range(max_precondition_iter):
#             print('\n running preconditioned newton; iter = ', i)
#             new_x, ncg_output = precondition_and_optimize(get_loss, x,\
#                                         maxiter = netwon_max_iter, gtol = gtol,
#                                         hessian = init_hessian)
#
#             # Check convergence.
#             new_f_val = get_loss(new_x)
#             grad_val = get_loss_grad(new_x)
#
#             x_diff = np.sum(np.abs(new_x - x))
#             f_diff = np.abs(new_f_val - f_val)
#             grad_l1 = np.sum(np.abs(grad_val))
#             x_conv = x_diff < xtol
#             f_conv = f_diff < ftol
#             grad_conv = grad_l1 < gtol
#
#             x = new_x
#             f_val = new_f_val
#
#             converged = x_conv or f_conv or grad_conv or ncg_output.success
#
#             print('Iter {}: x_diff = {}, f_diff = {}, grad_l1 = {}'.format(
#                 i, x_diff, f_diff, grad_l1))
#
#             if converged:
#                 print('done. ')
#                 break
#
#         return new_x
