import jax

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
                    epsilon = 0.,
                    use_preconditioning = True): 
        
        self.g_obs = g_obs
        self.vb_params_paragami = vb_params_paragami 
        self.prior_params_dict = prior_params_dict 
        
        self.gh_loc = gh_loc
        self.gh_weights = gh_weights 
        self.log_phi = log_phi 
        self.epsilon = epsilon 
        
        self.use_preconditioning = use_preconditioning
        
        self.set_and_compile_objectives()
    
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
        
        if self.use_preconditioning: 
            x = self._unprecondition(x_c, precond_params)
        else: 
            x = x_c
            
        return self._f(x)
    
    def set_and_compile_objectives(self): 
        self.f_precond = jax.jit(self._f_precond)
        self.precondition = jax.jit(self._precondition)
        self.unprecondition = jax.jit(self._unprecondition)
        
        self.grad = jax.jit(jax.grad(self._f_precond, argnums = 0))
        
        x = self.vb_params_paragami.flatten(self.vb_params_paragami.random(), 
                                            free = True)
        
        print('compiling preconditioned objective ... ')
        t0 = time.time()
        _ = self.f_precond(x, x).block_until_ready()
        _ = self.precondition(x, x).block_until_ready()
        _ = self.unprecondition(x, x).block_until_ready()
        _ = self.grad(x, x).block_until_ready()
        print('done. Elasped: {0:3g}'.format(time.time() - t0))