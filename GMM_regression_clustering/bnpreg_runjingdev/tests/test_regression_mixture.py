import jax
import jax.numpy as np
import jax.scipy as sp

from numpy.polynomial.hermite import hermgauss

import paragami

from bnpreg_runjingdev import regression_mixture_lib, genomics_data_utils
from bnpreg_runjingdev.regression_posterior_quantities import get_optimal_local_params_from_vb_dict
from bnpreg_runjingdev.genomics_utils import spline_bases_lib

import unittest


# get data
# TODO: replace this with simulated data
bnp_data_repo = './../../../genomic_time_series_bnp' 

y, _, regressors, _, _, _ = \
    genomics_data_utils.load_data_and_run_regressions(bnp_data_repo)

n_genes = y.shape[0]

# vb parameters
k_approx = 40
vb_params_dict, vb_params_paragami = \
    regression_mixture_lib.get_vb_params_paragami_object(dim = regressors.shape[1],
                                                         k_approx = k_approx)

# print(vb_params_paragami)

# Gauss-Hermite points for integrating logitnormal stick-breaking prior
gh_deg = 8
gh_loc, gh_weights = hermgauss(gh_deg)

# convert to jax arrays
gh_loc, gh_weights = np.array(gh_loc), np.array(gh_weights)

# prior parameters 
prior_params_dict, prior_params_paragami = \
    regression_mixture_lib.get_default_prior_params()

# paragami object for local parameters: 
def get_local_params_paragami(): 
    
    local_params_paragami = paragami.PatternDict()
    
    local_params_paragami['e_b'] = paragami.NumericArrayPattern(shape=(n_genes, k_approx))
    local_params_paragami['e_b2'] = paragami.NumericArrayPattern(shape=(n_genes, k_approx), lb = 0.)
    local_params_paragami['ez_free'] = paragami.NumericArrayPattern(shape=(n_genes, k_approx))

    local_params_dict = local_params_paragami.random()
    
    return local_params_dict, local_params_paragami

class TestRegressionMixture(unittest.TestCase):
    
    def test_optimal_shifts(self): 
        
        # get paragami object for local parameters
        local_params_dict, local_params_paragami = \
            get_local_params_paragami()
        
        # set optimal shifts
        _, _, e_b, e_b2 = \
            get_optimal_local_params_from_vb_dict(y, regressors, vb_params_dict, prior_params_dict, 
                                                  gh_loc, gh_weights)
        
        
        local_params_dict['e_b'] = e_b
        local_params_dict['e_b2'] = e_b2
        
        # test our shift updates 
        # loss as function of sticks
        def get_shift_loss(local_params_free):
            
            local_params_dict = local_params_paragami.fold(local_params_free, free = True)

            e_b = local_params_dict['e_b']
            e_b2 = local_params_dict['e_b2']

            # loglik
            loglik_obs = regression_mixture_lib.get_loglik_obs_by_nk(y, 
                                                                     regressors, 
                                                                     vb_params_dict,
                                                                     e_b, 
                                                                     e_b2).sum()

            # prior 
            prior_mean = prior_params_dict['prior_shift_mean']
            prior_info = prior_params_dict['prior_shift_info']
            stick_prior = regression_mixture_lib.get_shift_prior(e_b, e_b2, prior_mean, prior_info).sum() 

            # entropy
            stick_entropy = regression_mixture_lib.get_shift_entropy(e_b, e_b2).sum()

            return -(loglik_obs + stick_prior + stick_entropy)
        
        
        # this includes the z's but doesn't matter
        # it doesn't enter the loss
        local_params_free = local_params_paragami.flatten(local_params_dict, free = True)
        shift_grad = jax.grad(get_shift_loss)(local_params_free)
        
        assert np.abs(shift_grad).max() < 1e-6
    
    def test_optimal_local_params(self): 
        
                
        # get paragami object for local parameters
        local_params_dict, local_params_paragami = \
            get_local_params_paragami()
        
        # set optimal shifts and z's
        _, ez_free, e_b, e_b2 = \
            get_optimal_local_params_from_vb_dict(y,
                                                  regressors,
                                                  vb_params_dict,
                                                  prior_params_dict, 
                                                  gh_loc, gh_weights)
        
        
        local_params_dict['e_b'] = e_b
        local_params_dict['e_b2'] = e_b2
        local_params_dict['ez_free'] = ez_free

        # kl as function of local parameters
        def get_local_kl(local_free_params): 
    
            # get local parameters
            local_params_dict = local_params_paragami.fold(local_free_params, free = True)

            e_b = local_params_dict['e_b']
            e_b2 = local_params_dict['e_b2']
            ez_free = local_params_dict['ez_free']

            e_z = jax.nn.softmax(ez_free, axis = 1)

            z_nat_param = \
                regression_mixture_lib.get_optimal_z(y, regressors, 
                                                     vb_params_dict,
                                                     e_b, e_b2, 
                                                     gh_loc, gh_weights, 
                                                     prior_params_dict)[1]

            e_loglik = np.sum(e_z * z_nat_param) 

            # entropy term
            entropy = regression_mixture_lib.get_entropy(vb_params_dict,
                                                         e_z,
                                                         e_b, e_b2, 
                                                         gh_loc, gh_weights)

            # prior term
            e_log_prior = regression_mixture_lib.get_e_log_prior(vb_params_dict,
                                                                 prior_params_dict,
                                                                 gh_loc,
                                                                 gh_weights)

            elbo = e_log_prior + entropy + e_loglik

            return -1 * elbo.squeeze()
    
        
        local_params_free = local_params_paragami.flatten(local_params_dict, free = True)
        local_grad = jax.grad(get_local_kl)(local_params_free)
        
        assert np.abs(local_grad).max() < 1e-6
    
    
if __name__ == '__main__':
    unittest.main()
