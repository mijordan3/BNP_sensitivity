#!/usr/bin/env python3

import numpy as onp
import scipy as osp

import jax
import jax.numpy as np

from bnpmodeling_runjingdev import modeling_lib
from bnpmodeling_runjingdev import exponential_families as ef
from bnpmodeling_runjingdev.bnp_optimization_lib import update_stick_beta_params

import unittest

class TestBNPOptimization(unittest.TestCase):
    def test_update_stick_beta_params(self): 

        # test our stick updates 
        
        # randomly draw ez's
        onp.random.seed(3452)
        n_obs = 30
        k_approx = 10
        ez = np.array(onp.random.randn(n_obs, k_approx))
        ez = jax.nn.softmax(ez)

        dp_prior_alpha = 6.0
        
        # define loss function wrt to beta parameters
        def fold_beta_params(log_beta_params): 
            return np.exp(log_beta_params).reshape(-1, 2)        

        def _get_loss(ez, log_beta_params, dp_prior_alpha): 
    
            # fold beta params    
            beta_params = fold_beta_params(log_beta_params)

            # prior terms 
            e_log_sticks, e_log_1m_sticks = \
                    modeling_lib.get_e_log_beta(beta_params)

            dp_prior = (dp_prior_alpha - 1) * e_log_1m_sticks.sum()

            # entropy term 
            stick_entropy = ef.beta_entropy(tau = beta_params)

            # log-likelihood term 
            e_log_cluster_probs = \
                modeling_lib.get_e_log_cluster_probabilities_from_e_log_stick(
                                    e_log_sticks, e_log_1m_sticks)

            loglik = (ez.sum(0) * e_log_cluster_probs).sum()

            elbo = loglik + dp_prior + stick_entropy

            return -elbo
        
        # set gradients 
        get_loss = jax.jit(lambda x : _get_loss(ez, x, dp_prior_alpha))
        get_grad = jax.jit(jax.grad(lambda x : _get_loss(ez, x, dp_prior_alpha)))
        get_hess = jax.jit(jax.hessian(lambda x : _get_loss(ez, x, dp_prior_alpha)))

        # randomly initialize
        log_beta_params = np.array(onp.random.randn((k_approx - 1) * 2))
               
        # optimize
        out = osp.optimize.minimize(fun = lambda x : onp.array(get_loss(x)), 
                      x0 = log_beta_params, 
                      jac = lambda x : onp.array(get_grad(x)), 
                      hess = lambda x : onp.array(get_hess(x)), 
                      method = 'trust-exact')
        
        opt_beta = fold_beta_params(out.x)
        
        # my closed form update
        beta_update1, beta_update2 = update_stick_beta_params(ez, dp_prior_alpha)
        
        diff1 = np.abs(opt_beta[:, 0] - beta_update1).max()
        diff2 = np.abs(opt_beta[:, 1] - beta_update2).max()
                
        assert diff1 < 1e-3
        assert diff2 < 1e-3
        
if __name__ == '__main__':
    unittest.main()
