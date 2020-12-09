import jax

import jax.numpy as np
import jax.scipy as sp

import numpy as onp

from numpy.polynomial.hermite import hermgauss

from bnpmodeling_runjingdev import modeling_lib

from vb_lib import structure_model_lib, data_utils, cavi_lib
import vb_lib.structure_optimization_lib as s_optim_lib

import unittest

# draw data
n_obs = 10
n_loci = 5
n_pop = 3

g_obs = data_utils.draw_data(n_obs, n_loci, n_pop)[0]

# prior parameters
prior_params_dict, prior_params_paragami = \
    structure_model_lib.get_default_prior_params()

dp_prior_alpha = prior_params_dict['dp_prior_alpha']
allele_prior_alpha = prior_params_dict['allele_prior_alpha']
allele_prior_beta = prior_params_dict['allele_prior_beta']

# vb params
k_approx = 7
gh_deg = 8
gh_loc, gh_weights = hermgauss(gh_deg)
use_logitnormal_sticks = True

vb_params_dict, vb_params_paragami = \
    structure_model_lib.\
        get_vb_params_paragami_object(n_obs, 
                                      n_loci,
                                      k_approx,
                                      use_logitnormal_sticks)
vb_params_free = vb_params_paragami.flatten(vb_params_dict, 
                                           free = True)

# get structure objective: 
# this contains our custom gradients and hvps
stru_objective = s_optim_lib.StructureObjective(g_obs, 
                                                 vb_params_paragami,
                                                 prior_params_dict, 
                                                 gh_loc, gh_weights, 
                                                 jit_functions = True)

# we test against the ordinary gradients and hessians
# from jax
def get_kl(x): 
    vb_params_dict = vb_params_paragami.fold(x, free = True)
        
    return structure_model_lib.get_kl(g_obs, vb_params_dict, 
                                      prior_params_dict, 
                                      gh_loc, gh_weights, 
                                      detach_ez = False)

get_grad = jax.grad(get_kl)
get_hessian = jax.hessian(get_kl)

class TestStructureObjective(unittest.TestCase):
    
    def test_custom_hvp(self):
        hess = get_hessian(vb_params_free)
        
        for i in range(hess.shape[0]): 
            
            e_i = onp.zeros(hess.shape[0])
            e_i[i] = 1
    
            hvp1 = stru_objective.hvp(vb_params_free, e_i)
            hvp2 = np.dot(hess, e_i)
    
            assert np.abs(hvp1 - hvp2).max() < 1e-8
    
    def test_custom_grad(self): 
        # in computing the gradient in stru_objective
        # we detached the ez's. 
        # just check that they match wth ez's not-detached 
        
        grad1 = stru_objective.grad(vb_params_free)
        grad2 = get_grad(vb_params_free)
        
        assert np.abs(grad1 - grad2).max() < 1e-8
        
if __name__ == '__main__':
    unittest.main()
