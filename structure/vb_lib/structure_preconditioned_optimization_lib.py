import jax
import jax.numpy as np

import numpy as onp
from scipy import optimize 

import vb_lib.structure_model_lib as structure_model_lib
from vb_lib.preconditioner_lib import get_mfvb_cov_matmul

import time 

class StructurePrecondObjective():
    def __init__(self,
                    g_obs, 
                    vb_params_paragami,
                    prior_params_dict, 
                    gh_loc, gh_weights, 
                    log_phi = None,
                    epsilon = 0.): 
        
        self.g_obs = g_obs
        self.vb_params_paragami = vb_params_paragami 
        self.prior_params_dict = prior_params_dict 
        
        self.gh_loc = gh_loc
        self.gh_weights = gh_weights 
        self.log_phi = log_phi 
        self.epsilon = epsilon 
                
        self.compile_objectives()
        self.compile_preconditioned_objectives()
    
    def _f(self, x):
        
        vb_params_dict = self.vb_params_paragami.fold(x, free = True)
        
        return structure_model_lib.get_kl(self.g_obs, vb_params_dict, 
                                  self.prior_params_dict, 
                                  self.gh_loc, self.gh_weights, 
                                  log_phi = self.log_phi, 
                                  epsilon = self.epsilon)
    
    def _precondition(self, x, precond_params): 
        vb_params_dict = self.vb_params_paragami.fold(precond_params, free = True)
        
        return get_mfvb_cov_matmul(x, vb_params_dict,
                                self.vb_params_paragami,
                                return_info = False, 
                                return_sqrt = True)
    
    def _unprecondition(self, x_c, precond_params): 
        
        vb_params_dict = self.vb_params_paragami.fold(precond_params, free = True)
        
        return get_mfvb_cov_matmul(x_c, vb_params_dict,
                                self.vb_params_paragami,
                                return_info = True, 
                                return_sqrt = True)
        
    def _f_precond(self, x_c, precond_params): 
                    
        return self._f(self._unprecondition(x_c, precond_params))
    
    def compile_objectives(self): 
        self.f = jax.jit(self._f)
        self.grad = jax.jit(jax.grad(self._f))
        
        x = self.vb_params_paragami.flatten(self.vb_params_paragami.random(), 
                                            free = True)
        
        print('compiling objectives ... ')
        t0 = time.time()
        _ = self.f(x).block_until_ready()
        _ = self.grad(x).block_until_ready()
        print('done. Elasped: {0:3g}'.format(time.time() - t0))
    
    def compile_preconditioned_objectives(self): 
        self.f_precond = jax.jit(self._f_precond)
        self.precondition = jax.jit(self._precondition)
        self.unprecondition = jax.jit(self._unprecondition)
        
        self.grad_precond = jax.jit(jax.grad(self._f_precond, argnums = 0))
        
        x = self.vb_params_paragami.flatten(self.vb_params_paragami.random(), 
                                            free = True)
        
        print('compiling preconditioned objective ... ')
        t0 = time.time()
        _ = self.f_precond(x, x).block_until_ready()
        _ = self.precondition(x, x).block_until_ready()
        _ = self.unprecondition(x, x).block_until_ready()
        _ = self.grad_precond(x, x).block_until_ready()
        print('done. Elasped: {0:3g}'.format(time.time() - t0))
        
        
def optimize_structure(g_obs, 
                        vb_params_dict, 
                        vb_params_paragami,
                        prior_params_dict,
                        gh_loc, gh_weights, 
                        log_phi = None, epsilon = 0., 
                        precondition_every = 20, 
                        maxiter = 2000): 
    
    # preconditioned objective 
    precon_objective = StructurePrecondObjective(g_obs, 
                                vb_params_paragami,
                                prior_params_dict,
                                gh_loc = gh_loc, gh_weights = gh_weights,                       
                                log_phi = log_phi, epsilon = epsilon)
    
    t0 = time.time()
    
    # run a few iterations without preconditioning 
    print('Run a few iterations without preconditioning ... ')
    init_vb_free = vb_params_paragami.flatten(vb_params_dict, free = True)
    out = optimize.minimize(lambda x : onp.array(precon_objective.f(x)),
                        x0 = onp.array(init_vb_free),
                        jac = lambda x : onp.array(precon_objective.grad(x)),
                        method='L-BFGS-B', 
                        options = {'maxiter': precondition_every})
    iters = out.nit
    success = out.success
    vb_params_free = out.x
    
    print('iteration [{}]; kl:{}; elapsed: {}secs'.format(iters,
                                        np.round(out.fun, 6),
                                        round(time.time() - t0, 4)))
    
    # precondition and run
    while (success == False) and (iters < maxiter): 
        t1 = time.time() 
        
        # transform into preconditioned space
        x0 = precon_objective.precondition(vb_params_free, vb_params_free)
        
        out = optimize.minimize(lambda x : onp.array(precon_objective.f_precond(x, vb_params_free)),
                        x0 = onp.array(x0),
                        jac = lambda x : onp.array(precon_objective.grad_precond(x, vb_params_free)),
                        method='L-BFGS-B', 
                        options = {'maxiter': precondition_every})
        
        iters += out.nit
        success = out.success
        
        # transform to original parameterization
        vb_params_free = precon_objective.unprecondition(out.x, vb_params_free)
        
        print('iteration [{}]; kl:{}; elapsed: {}secs'.format(iters,
                                        np.round(out.fun, 6),
                                        round(time.time() - t1, 4)))
        
    vb_opt = vb_params_free
    vb_opt_dict = vb_params_paragami.fold(vb_opt, free = True)
    
    print('done. Elapsed {}'.format(round(time.time() - t0, 4)))
    
    return vb_opt_dict, vb_opt, out, precon_objective