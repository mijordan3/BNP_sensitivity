import jax
import jax.numpy as np
import jax.scipy as sp

import numpy as onp
from scipy import optimize

import paragami
from paragami import OptimizationObjective

import time

class OptimizationObjectiveJaxtoNumpy(OptimizationObjective): 
    def __init__(self, get_loss, init_params, compile_hvp = False, 
                        print_every = 1, log_every = 0): 
        super().__init__(get_loss, print_every = print_every, log_every = log_every)
        
        # jit the functions
        self.grad = jax.jit(self.grad)
        self.hessian_vector_product = jax.jit(self.hessian_vector_product)
        self._objective_fun = jax.jit(self._objective_fun)
        
        # compile
        self.compile_derivatives(init_params, compile_hvp)
    
    def compile_derivatives(self, init_params, compile_hvp): 
        # compile derivatives
        t0 = time.time()
        
        print('Compiling objective ...')
        _ = self.f(init_params)
        
        print('Compiling grad ...')
        _ = self.grad(init_params)
        
        if compile_hvp:
            print('Compiling hvp ...')
            _ = self.hessian_vector_product(init_params, init_params)
            
        print('Compile time: {0:3g}secs'.format(time.time() - t0))
        
        self.reset()
    
    def f_np(self, x): 
        return onp.array(self.f(x))
    
    def grad_np(self, x): 
        return onp.array(self.grad(x))
        
    def hvp_np(self, x, v): 
        return onp.array(self.hessian_vector_product(x, v))
            

def run_lbfgs(optim_objective, init_vb_free_params, maxiter = 1000):

    # run l-bfgs-b
    t0 = time.time()
    print('\nRunning L-BFGS-B ... ')
    out = optimize.minimize(optim_objective.f_np,
                        x0 = onp.array(init_vb_free_params),
                        jac = optim_objective.grad_np,
                        method='L-BFGS-B', 
                        options = {'maxiter': maxiter})
    
    print('done. Elapsed {0:3g}secs'.format(time.time() - t0))

    print('objective value: ', optim_objective.f_np(out.x))

    return out

