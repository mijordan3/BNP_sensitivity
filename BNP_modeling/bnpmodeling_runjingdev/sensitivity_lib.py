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
        self.opt_par_value = opt_par_value
        self.hyper_par_value0 = hyper_par_value0
        if hyper_par_objective_fun is None:
            hyper_par_objective_fun = objective_fun

        self.cg_precond = cg_precond

        # set up linear system methods
        self._hessian_solver_jitted = jax.jit(self._hessian_solver) 
        self._set_hessian_solver(self.opt_par_value, self.hyper_par_value0)
        
        # compile linear system
        print('Compiling hessian solver ...')
        t0 = time.time()
        _ = self.hessian_solver(self.opt_par_value).block_until_ready()
        print('Hessian-solver compile time: {0:3g}sec\n'.format(time.time() - t0)) 
        
        # get cross hessian function
        self._set_cross_hess_and_solve(hyper_par_objective_fun)


    def _set_cross_hess_and_solve(self, hyper_par_objective_fun):
        dobj_dhyper = jax.jit(jax.jacobian(hyper_par_objective_fun, 1))
        self.dobj_dhyper_dinput = jax.jit(jax.jacobian(dobj_dhyper), 0)

        print('Compiling cross hessian...')
        t0 = time.time()
        out = self.dobj_dhyper_dinput(self.opt_par_value,
                                        self.hyper_par_value0).squeeze()
        assert (len(out.shape) == 1) and (len(out) == len(self.opt_par_value)), \
                'cross hessian shape: ' + str(out.shape)
        print('Cross-hessian compile time: {0:3g}sec\n'.format(time.time() - t0))

        self._set_dinput_dhyper()

    def _set_dinput_dhyper(self):

        t0 = time.time()
        cross_hess = self.dobj_dhyper_dinput(self.opt_par_value,
                                                self.hyper_par_value0)

        self.dinput_dhyper = -self.hessian_solver(cross_hess.squeeze()).\
                                    block_until_ready()

        print('LR sensitivity time: {0:3g}sec\n'.format(time.time() - t0))
    
    def _hessian_solver(self, opt_par_value, hyper_par_value0, b):
        
        # hessian vector product
        obj_fun_hvp = get_jac_hvp_fun(lambda x : self.objective_fun(x, hyper_par_value0))
        
        return cg(A = lambda x : obj_fun_hvp(opt_par_value, x),
                   b = b,
                   M = self.cg_precond)[0]
        
    def _set_hessian_solver(self, opt_par_value, hyper_par_value0): 
        self.hessian_solver = lambda b : \
            self._hessian_solver_jitted(opt_par_value, hyper_par_value0, b)
    
    def predict_opt_par_from_hyper_par(self, hyper_par_value):
        delta = (hyper_par_value - self.hyper_par_value0)

        if len(self.dinput_dhyper.shape) == 1:
            self.dinput_dhyper = np.expand_dims(self.dinput_dhyper, 1)

        return np.dot(self.dinput_dhyper, delta) + self.opt_par_value
