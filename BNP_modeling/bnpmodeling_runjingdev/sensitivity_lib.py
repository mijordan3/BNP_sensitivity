# this is a bare-bones version of vittles.HyperparameterSensitivityLinearApproximation
# implemented in jax

import jax
import jax.numpy as np

from jax.scipy.sparse.linalg import cg

import time

# wrapper to get hessian vector products in jax
def get_jac_hvp_fun(f):
    def hvp(x, v):
        return jax.jvp(jax.grad(f), (x, ), (v, ))[1]
    return hvp


class HyperparameterSensitivityLinearApproximation(object):
    def __init__(self,
                    objective_fun,
                    opt_par_value,
                    hyper_par_value0,
                    hyper_par_objective_fun = None,
                    cg_precond = None):

        self.objective_fun = objective_fun
        if hyper_par_objective_fun is None:
            hyper_par_objective_fun = objective_fun

        self.cg_precond = cg_precond

        # set up linear system methods
        self._hessian_solver_jitted = jax.jit(self._hessian_solver) 
                
        # set up cross hessian 
        self._set_cross_hess(hyper_par_objective_fun)
        
        # set derivatives
        # this will be slow because of compile time ... 
        # subsequent calls should be fast. 
        print('Compiling ...')
        t0 = time.time()
        self.set_derivatives(opt_par_value, hyper_par_value0)
        print('Compile time: {0:3g}sec\n'.format(time.time() - t0))

    def _set_cross_hess(self, hyper_par_objective_fun):
        # note to myself: 
        # to test a new multiplicative perturbation, 
        # just reset the cross hessian and then rerun 
        # _set_dinput_dhyper(). The linear system 
        # doesnt need to be recompiled. 
        
        dobj_dhyper = jax.jacobian(hyper_par_objective_fun, 1)
        self.dobj_dhyper_dinput = jax.jit(jax.jacobian(dobj_dhyper), 0)
        
    def _set_dinput_dhyper(self):

        cross_hess = self.dobj_dhyper_dinput(self.opt_par_value,
                                                self.hyper_par_value0)

        self.dinput_dhyper = -self.hessian_solver(cross_hess.squeeze()).\
                                    block_until_ready()

        
    def _hessian_solver(self, opt_par_value, hyper_par_value0, b):
        
        # hessian vector product
        obj_fun_hvp = get_jac_hvp_fun(lambda x : self.objective_fun(x, hyper_par_value0))
        
        return cg(A = lambda x : obj_fun_hvp(opt_par_value, x),
                   b = b,
                   M = self.cg_precond)[0]
        
    def _set_hessian_solver(self, opt_par_value, hyper_par_value0): 
        self.hessian_solver = lambda b : \
            self._hessian_solver_jitted(opt_par_value, hyper_par_value0, b)
    
    def set_derivatives(self, 
                           opt_par_value, 
                           hyper_par_value0):
        
        self.opt_par_value = opt_par_value
        self.hyper_par_value0 = hyper_par_value0
        
        self._set_hessian_solver(self.opt_par_value, self.hyper_par_value0)
        self._set_dinput_dhyper()

    def predict_opt_par_from_hyper_par(self, hyper_par_value):
        delta = (hyper_par_value - self.hyper_par_value0)

        if len(self.dinput_dhyper.shape) == 1:
            self.dinput_dhyper = np.expand_dims(self.dinput_dhyper, 1)

        return np.dot(self.dinput_dhyper, delta) + self.opt_par_value
