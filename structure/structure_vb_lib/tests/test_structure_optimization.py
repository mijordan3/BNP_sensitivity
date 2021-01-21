import jax
import jax.numpy as np
import numpy as onp

from structure_vb_lib import structure_model_lib, testutils
import structure_vb_lib.structure_optimization_lib as s_optim_lib

import unittest

class TestStructureObjective(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        
        super(TestStructureObjective, self).__init__(*args, **kwargs)
        
        self.g_obs, vb_params_dict, self.vb_params_paragami, self.prior_params_dict, \
            self.gh_loc, self.gh_weights= \
                testutils.draw_data_and_construct_model()
        
        self.vb_params_free = self.vb_params_paragami.flatten(vb_params_dict,
                                                              free = True)
        
        # get structure objective: 
        # this contains our custom gradients and hvps
        self.stru_objective = \
            s_optim_lib.StructureObjective(self.g_obs, 
                                           self.vb_params_paragami,
                                           self.prior_params_dict, 
                                           self.gh_loc, 
                                           self.gh_weights)
        
        # get the autograd gradients and hessians 
        self.get_grad = jax.grad(self.stru_objective.f)
        self.get_hessian = jax.hessian(self.stru_objective.f)

    def test_custom_derivatives(self):
        
        # check gradients 
        # in stru_objective.grad,
        # we had called detach_ez = True, and did not compute 
        # z-entropies for a speed-up. 
        
        # check that it matches with detach_ez = False. 
        
        grad1 = self.stru_objective.grad(self.vb_params_free)
        grad2 = self.get_grad(self.vb_params_free)
        assert np.abs(grad1 - grad2).max() < 1e-8

        
        # check my hessian vector products
        hess = self.get_hessian(self.vb_params_free)
        
        for i in range(hess.shape[0]): 
            
            e_i = onp.zeros(hess.shape[0])
            e_i[i] = 1
    
            hvp1 = self.stru_objective.hvp(self.vb_params_free, e_i)
            hvp2 = np.dot(hess, e_i)
    
            assert np.abs(hvp1 - hvp2).max() < 1e-8


class TestStructurePreconditionedObjective(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        
        super(TestStructurePreconditionedObjective, self).__init__(*args, **kwargs)
        
        # draw model
        self.g_obs, vb_params_dict, self.vb_params_paragami, self.prior_params_dict, \
            self.gh_loc, self.gh_weights= \
                testutils.draw_data_and_construct_model()
        
        # set vb parameters and parameters at which we precondition
        self.vb_params_free = self.vb_params_paragami.flatten(vb_params_dict,
                                                              free = True)
        self.precond_params = self.vb_params_free
        
        # get preconditioned structure objective: 
        # this contains our custom gradients and hvps
        self.stru_precond_objective = \
            s_optim_lib.StructurePrecondObjective(self.g_obs, 
                                           self.vb_params_paragami,
                                           self.prior_params_dict, 
                                           self.gh_loc, 
                                           self.gh_weights)
        
        # get the autograd gradients and hessians 
        precond_kl = lambda x_c : \
            self.stru_precond_objective.f_precond(x_c, 
                                                  self.precond_params)
        
        self.get_grad = jax.grad(precond_kl)
        self.get_hessian = jax.hessian(precond_kl)
        
    def test_custom_derivatives(self):
        
        # check gradients 
        # in stru_objective.grad,
        # we had called detach_ez = True, and did not compute 
        # z-entropies for a speed-up. 
        
        # check that it matches with detach_ez = False. 
        grad1 = self.stru_precond_objective.grad_precond(self.vb_params_free, 
                                                         self.precond_params)
        grad2 = self.get_grad(self.vb_params_free)
        assert np.abs(grad1 - grad2).max() < 1e-8

        
        # check my hessian vector products
        hess = self.get_hessian(self.vb_params_free)
        
        for i in range(hess.shape[0]): 
            
            e_i = onp.zeros(hess.shape[0])
            e_i[i] = 1
    
            hvp1 = self.stru_precond_objective.hvp_precond(self.vb_params_free, 
                                                           self.precond_params,
                                                           e_i)
            hvp2 = np.dot(hess, e_i)
    
            assert np.abs(hvp1 - hvp2).max() < 1e-8


class TestStructureOptimization(unittest.TestCase):
    def test_optimization(self): 
        
        # this just checks that all the optimization 
        # functions run without error 
        _ = testutils.construct_model_and_optimize()
        
if __name__ == '__main__':
    unittest.main()
