import jax
import jax.numpy as np

import numpy as onp
from scipy import optimize 

import paragami
import time 

from structure_vb_lib.structure_model_lib import get_kl
from structure_vb_lib.posterior_quantities_lib import get_optimal_z_from_vb_dict
from structure_vb_lib import preconditioner_lib

from bnpmodeling_runjingdev.bnp_optimization_lib import optimize_kl
from bnpmodeling_runjingdev.sensitivity_lib import get_jac_hvp_fun

class StructureObjective():
    """
    This class contains the structure objective. 
    The method self.f returns the KL objective 
    as a function of vb parameters in flattened space. 
    
    The methods self.grad and self.hvp return the 
    gradient and hessian vector product, respectively. 
    """
    
    def __init__(self,
                 g_obs, 
                 vb_params_paragami,
                 prior_params_dict, 
                 gh_loc, gh_weights, 
                 e_log_phi = None, 
                 jit_functions = True): 
        """
        Parameters
        ----------
        g_obs : ndarray
            The array of one-hot encoded genotypes, of shape (n_obs, n_loci, 3)
        vb_params_paragami : paragami patterned dictionary
            A paragami patterned dictionary that contains the variational parameters.
        prior_params_dict : dictionary
            Dictionary of prior parameters.
            parameters
        gh_loc : vector
            Locations for gauss-hermite quadrature. 
        gh_weights : vector
            Weights for gauss-hermite quadrature. 
        e_log_phi : function
            Function with arguments stick_means and stick_infos 
            and returns the expected log-multiplicative perturbation.
        jit_functions : boolean
            Whether or not to call jax.jit on the function and 
            gradients
        """

        self.g_obs = g_obs
        self.vb_params_paragami = vb_params_paragami 
        self.prior_params_dict = prior_params_dict 
                    
        self.gh_loc = gh_loc
        self.gh_weights = gh_weights 
        self.e_log_phi = e_log_phi 
        
        self._grad = jax.grad(self._f)
        self._hvp = get_jac_hvp_fun(self._f)

        self.jit_functions = jit_functions
        self._define_functions()
        if self.jit_functions: 
            self._compile_functions()
        

    def _f(self, vb_free_params):
        # this is the objective function 
        
        vb_params_dict = self.vb_params_paragami.fold(vb_free_params, free = True)

        return get_kl(self.g_obs,
                      vb_params_dict, 
                      self.prior_params_dict, 
                      self.gh_loc, self.gh_weights, 
                      e_log_phi = self.e_log_phi)
    
        
    def _define_functions(self): 
        if self.jit_functions: 
            self.f = jax.jit(self._f)
            self.grad = jax.jit(self._grad)
            self.hvp = jax.jit(self._hvp)
            
        else: 
            self.f = self._f
            self.grad = self._grad
            self.hvp = self._hvp
    
    def _compile_functions(self): 
        x = self.vb_params_paragami.flatten(self.vb_params_paragami.random(), 
                                            free = True)
        
        print('compiling objective ... ')
        t0 = time.time()
        _ = self.f(x).block_until_ready()
        _ = self.grad(x).block_until_ready()
        _ = self.hvp(x, x).block_until_ready()
        print('done. Elasped: {0:3g}'.format(time.time() - t0))
        

class StructurePrecondObjective(StructureObjective):
    """
    This class contains the structure objective. 
    the method self.f returns the KL objective 
    as a function of vb parameters in free space. 
    
    The methods self.grad and self.hvp return the 
    gradient and hessian vector product, respectively. 
    
    The corresponding methods self.f_precond, 
    self.grad_precond, self.hvp_precond are 
    preconditioned functions, using the MFVB covariance 
    as a preconditioner.
    
    The preconditioned functions take two arguments: the 
    parameters at which to evaluate the function 
    (in preconditioned space), 
    and the vb parameters at which the preconditioner is 
    precomputed. (This is so that we do not need to 
    re-compile everything
    when we want to update the preconditioner).
    """
    
    def __init__(self,
                 g_obs, 
                 vb_params_paragami,
                 prior_params_dict, 
                 gh_loc, gh_weights, 
                 compile_hvp = True,
                 e_log_phi = None): 
        
        """
        Parameters
        ----------
        g_obs : ndarray
            The array of one-hot encoded genotypes, of shape (n_obs, n_loci, 3)
        vb_params_paragami : paragami patterned dictionary
            A paragami patterned dictionary that contains the variational parameters.
        prior_params_dict : dictionary
            Dictionary of prior parameters.
            parameters
        gh_loc : vector
            Locations for gauss-hermite quadrature. 
        gh_weights : vector
            Weights for gauss-hermite quadrature. 
        e_log_phi : function
            Function with arguments stick_means and stick_infos 
            and returns the expected log-multiplicative perturbation.
        """

        super().__init__(g_obs, 
                         vb_params_paragami,
                         prior_params_dict, 
                         gh_loc, gh_weights, 
                         e_log_phi = e_log_phi, 
                         jit_functions = True)
        
        self.compile_preconditioned_objectives(compile_hvp)
        
    def _precondition(self, x, precond_params): 
        print('not actually precoditioned')
        return x

#         vb_params_dict = self.vb_params_paragami.fold(precond_params, free = True)
        
#         return preconditioner_lib.get_mfvb_cov_matmul(x,
#                                                       vb_params_dict,
#                                                       self.vb_params_paragami,
#                                                       return_info = False, 
#                                                       return_sqrt = True)
    
    def _unprecondition(self, x_c, precond_params):
        return x_c
        
#         vb_params_dict = self.vb_params_paragami.fold(precond_params, free = True)
        
#         return preconditioner_lib.get_mfvb_cov_matmul(x_c, vb_params_dict,
#                                                       self.vb_params_paragami,
#                                                       return_info = True, 
#                                                       return_sqrt = True)
        
    def _f_precond(self, x_c, precond_params): 
        return self._f(self._unprecondition(x_c, precond_params))
    
    def _grad_precond(self, x_c, precond_params): 
        x = self._unprecondition(x_c, precond_params)
        return self._unprecondition(self._grad(x), precond_params)
        
    def _hvp_precond(self, x_c, precond_params, v): 
        
        x = self._unprecondition(x_c, precond_params)
        v1 = self._unprecondition(v, precond_params)
        hvp = self._hvp(x, v1)
        hvp = self._unprecondition(hvp, precond_params)

        return hvp
    
    def compile_preconditioned_objectives(self, compile_hvp): 
        self.f_precond = jax.jit(self._f_precond)
        self.precondition = jax.jit(self._precondition)
        self.unprecondition = jax.jit(self._unprecondition)

        self.grad_precond = jax.jit(self._grad_precond)
        
        if compile_hvp: 
            self.hvp_precond = jax.jit(self._hvp_precond)
        else: 
            self.hvp_precond = self._hvp_precond
        
        x = self.vb_params_paragami.flatten(self.vb_params_paragami.random(), 
                                            free = True)
        
        print('compiling preconditioned objective ... ')
        t0 = time.time()
        _ = self.f_precond(x, x).block_until_ready()
        _ = self.precondition(x, x).block_until_ready()
        _ = self.unprecondition(x, x).block_until_ready()
        
        _ = self.grad_precond(x, x).block_until_ready()
        _ = self.hvp_precond(x, x, x).block_until_ready()
        print('done. Elasped: {0:3g}'.format(time.time() - t0))

        
        
def optimize_structure(g_obs, 
                       vb_params_dict, 
                       vb_params_paragami,
                       prior_params_dict,
                       gh_loc, gh_weights, 
                       e_log_phi = None, 
                       precondition_every = 10, 
                       max_lbfgs_iter = 1000,
                       maxiter = 2000, 
                       x_tol = 1e-2, 
                       f_tol = 1e-8): 
    """
    Parameters
    ----------
    g_obs : ndarray
        The array of one-hot encoded genotypes, of shape (n_obs, n_loci, 3)
    vb_params_dict : dictionary
        A dictionary that contains the initial variational parameters.
    vb_params_paragami : paragami patterned dictionary
        A paragami patterned dictionary that contains the variational parameters.
    prior_params_dict : dictionary
        Dictionary of prior parameters.
        parameters
    gh_loc : vector
        Locations for gauss-hermite quadrature. 
    gh_weights : vector
        Weights for gauss-hermite quadrature. 
    e_log_phi : function
        Function with arguments stick_means and stick_infos 
        and returns the expected log-multiplicative perturbation.
    precondition_every : integer
        We re-compute the preconditioner after `precondition_every` steps. 
    """

    # preconditioned objective 
    precon_objective = StructurePrecondObjective(g_obs, 
                                vb_params_paragami,
                                prior_params_dict,
                                gh_loc = gh_loc, 
                                gh_weights = gh_weights,                       
                                e_log_phi = e_log_phi, 
                                compile_hvp = True)
    
    t0 = time.time()
    
    vb_params_free = vb_params_paragami.flatten(vb_params_dict, free = True)
    print('init kl: {:.6f}'.format(precon_objective.f(vb_params_free)))
    
    ################
    # Run L-BFGS
    ################
    print('running L-BFGS')
    out = optimize.minimize(lambda x : onp.array(precon_objective.f(x)),
                            x0 = onp.array(vb_params_free),
                            jac = lambda x : onp.array(precon_objective.grad(x)),
                            method='L-BFGS-B', 
                            options = {'maxiter': max_lbfgs_iter})
    
    vb_params_free = out.x 
    print('LBFGS-message: ', out.message)
    print('Elapsed: {:.3f}'.format(time.time() - t0))
    
    ################
    # Run pre-conditioned newton steps
    ################
    # precondition and run
    iters = 0
    old_kl = 1e16
    
    print('running preconditioned newton-cg')
    
    while (iters < maxiter): 
        t1 = time.time() 
        
        # transform into preconditioned space
        x0 = vb_params_free
        x0_c = precon_objective.precondition(x0, vb_params_free)
        
        # optimize
        out = optimize.minimize(lambda x : onp.array(precon_objective.f_precond(x, vb_params_free)),
                        x0 = onp.array(x0_c),
                        jac = lambda x : onp.array(precon_objective.grad_precond(x, vb_params_free)),
                        hessp = lambda x , v : onp.array(precon_objective.hvp_precond(x, vb_params_free, v)),
                        method = 'trust-ncg', 
                        options = {'maxiter': precondition_every})
        
        iters += out.nit
                
        print('iteration [{}]; kl:{:.6f}; elapsed: {:.3f}secs'.format(iters,
                                                                      out.fun,
                                                                      time.time() - t1))

        # transform to original parameterization
        vb_params_free = precon_objective.unprecondition(out.x, vb_params_free)
        
        print('   ', out.message)

        x_tol_success = np.abs(vb_params_free - x0).max() < x_tol
        if x_tol_success:
            print('x-tolerance reached')
            break
        
        f_tol_success = np.abs(old_kl - out.fun) < np.abs(f_tol * out.fun)
        if f_tol_success: 
            print('f-tolerance reached')
            break

        old_kl = out.fun
            
    vb_opt = vb_params_free
    vb_opt_dict = vb_params_paragami.fold(vb_opt, free = True)
    
    optim_time = time.time() - t0
    print('done. Elapsed {}'.format(round(optim_time, 4)))
    
    return vb_opt_dict, vb_opt, out, precon_objective, optim_time
