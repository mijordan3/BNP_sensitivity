# this is a bare-bones version of vittles.HyperparameterSensitivityLinearApproximation
# implemented in jax

import jax
import jax.numpy as np

import numpy as onp
from jax.scipy.sparse.linalg import cg

import scipy.sparse.linalg as sparse_linalg

import time

import inspect

# wrapper to get hessian vector products in jax
def get_jac_hvp_fun(f):
    def hvp(x, v):
        return jax.jvp(jax.grad(f), (x, ), (v, ))[1]
    return hvp

class HyperparameterSensitivityLinearApproximation(object):
    # sensitivity class adapted from 
    # https://github.com/rgiordan/vittles/blob/master/vittles/sensitivity_lib.py
    
    def __init__(self,
                 objective_fun,
                 opt_par_value,
                 hyper_par_value0,
                 obj_fun_hvp = None,
                 hyper_par_objective_fun = None,
                 cg_precond = None, 
                 use_scipy_cgsolve = False):
        """
        Parameters
        ----------
        objective_fun : function
            Objective as function of vb parameters (in flattened space)
            and prior parameter. 
        opt_par_value : array
            the value that optimizes `objective_fun` at `hyper_par_value0`
        hyper_par_value0 : array
            the prior parameter for which 'opt_var_value` optimizes 
            `objective_fun`. 
        obj_fun_hvp : function (optional)
            Function that takes in a vector of same length as `opt_par_value`
            and returns the hessian vector product at `opt_par_value`. 
            If none, this is computed automatically using jax derivatives.
        hyper_par_objective_fun : function, optional
            The part of ``objective_fun`` depending on both ``opt_par`` and
            ``hyper_par``. If not specified,
            ``objective_fun`` is used.
        cg_precond : function (optional)
            Function that takes in a vector `v` of same length as `opt_par_value`
            and returns a preconditioner times `v` for the cg solver
            (this is the argument `M`)
        use_scipy_cgsolve : boolean
            If `True`, we compile HVPs and use the scipy solver (which 
            has richer callback functions we can use for printing values
            and debugging). 
            If `False`, the entire hessian solver is compiled. 
            That is, we compile the mapping from (vb_opt, cross_hess) to dinput/dhyper
            is jitted). Initial compile time will be slow, but the 
            subsequent evaluations will be fast. 
        """
        

        self.objective_fun = objective_fun
        self.opt_par_value = opt_par_value
        self.hyper_par_value0 = hyper_par_value0
        
        if hyper_par_objective_fun is None:
            hyper_par_objective_fun = objective_fun

        self.cg_precond = cg_precond
        self.use_scipy_cgsolve = use_scipy_cgsolve

        # hessian vector products
        if obj_fun_hvp is None: 
            self.obj_fun_hvp = get_jac_hvp_fun(lambda x : 
                                               objective_fun(x, self.hyper_par_value0))
        else: 
            print('NOTE: using custom hvp')
            self.obj_fun_hvp = obj_fun_hvp

        # compile linear system
        self._set_hessian_solver()

        # get cross hessian function
        self._set_cross_hess_and_solve(hyper_par_objective_fun)


    def _set_cross_hess_and_solve(self, hyper_par_objective_fun):
        # note to myself: 
        # this method can be called to reset the functional perturbation 
        # without re-compiling the linear system (which is expensive)

        dobj_dhyper = jax.jacobian(hyper_par_objective_fun, 1)
        self.dobj_dhyper_dinput = jax.jit(jax.jacobian(dobj_dhyper), 0)

        print('Compiling cross hessian...')
        t0 = time.time()
        out = self.dobj_dhyper_dinput(self.opt_par_value,
                                      self.hyper_par_value0).squeeze().\
                                        block_until_ready()
        
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
        
        # save timing result ... 
        self.lr_time = time.time() - t0
        print('LR sensitivity time: {0:3g}sec\n'.format(self.lr_time))

    def _set_hessian_solver(self):
        
        if self.use_scipy_cgsolve: 
            self.cg_solver = ScipyCgSolver(self.obj_fun_hvp, 
                                           self.opt_par_value, 
                                           self.cg_precond)
            
            self.hessian_solver = lambda b : self.cg_solver.hessian_solver(b)
            
        else: 
            self.hessian_solver = \
                jax.jit(lambda b : cg(A = lambda x : self.obj_fun_hvp(self.opt_par_value, x),
                                       b = b,
                                       M = self.cg_precond)[0])
            
            print('Compiling hessian solver ...')
            t0 = time.time()
            _ = self.hessian_solver(self.opt_par_value * 0.).block_until_ready()
            print('Hessian solver compile time: {0:3g}sec\n'.format(time.time() - t0))


    def predict_opt_par_from_hyper_par(self, hyper_par_value):
        delta = (hyper_par_value - self.hyper_par_value0)

        if len(self.dinput_dhyper.shape) == 1:
            self.dinput_dhyper = np.expand_dims(self.dinput_dhyper, 1)

        return np.dot(self.dinput_dhyper, delta) + self.opt_par_value

class ScipyCgSolver(): 
    def __init__(self, obj_fun_hvp, opt_par_value, cg_precond): 
        
        self.vb_dim = len(opt_par_value)
        self.hvp = jax.jit(lambda x : obj_fun_hvp(opt_par_value, x))
        
        print('Compiling hvp ...')
        t0 = time.time()
        _ = self.hvp(opt_par_value).block_until_ready()
        print('hvp compile time: {0:3g}sec\n'.format(time.time() - t0))
        
        if cg_precond is not None: 
            self.cg_precond = jax.jit(cg_precond)
            print('Compiling preconditioner ...')
            t0 = time.time()
            _ = self.cg_precond(opt_par_value).block_until_ready()
            print('preconditioner compile time: {0:3g}sec\n'.format(time.time() - t0))
            
            self.M = sparse_linalg.LinearOperator(matvec = self.cg_precond, 
                                              shape = (self.vb_dim, ) * 2)
        else: 
            self.M = None
            
        self.A = sparse_linalg.LinearOperator(matvec = self.hvp, 
                                              shape = (self.vb_dim, ) * 2)

    def callback(self, xk): 
        
        if self.iter > 0: 
            
            # this gets the residuals. adapted from 
            # https://stackoverflow.com/questions/14243579/ \
            # print-current-residual-from-callback-in-scipy-sparse-linalg-cg
            frame = inspect.currentframe().f_back
            res = frame.f_locals['resid']
            
            elapsed = time.time() - self.t0
            print('Iter [{}]; elapsed {}sec; diff: {}'.format(self.iter,
                                                              np.round(elapsed, 3), 
                                                              res))
            
        self.t0 = time.time()
        self.iter += 1
        
    def hessian_solver(self, b): 
        self.iter = 0
        out = sparse_linalg.cg(A = self.A, 
                                b = b, 
                                M = self.M, 
                                callback = self.callback)
        
        return np.array(out[0])
                                
        