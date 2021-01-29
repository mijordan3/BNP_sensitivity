import jax

import jax.numpy as np

from bnpmodeling_runjingdev import cluster_quantities_lib

from structure_vb_lib import posterior_quantities_lib, testutils

import unittest
import numpy.testing as testing


class TestCaviUpdate(unittest.TestCase):
    
    def test_get_e_num_clusters(self):
        # we implemented get_e_num_clusters 
        # using jax.lax.scan, so that we never 
        # need to instantiate the ez's. 
        
        # check against the analytic result when 
        # instantiating the ez's. 
        
        # define model
        k_approx = 10
        g_obs, vb_params_dict, vb_parmas_paragami, prior_params_dict, gh_loc, gh_weights = \
            testutils.draw_data_and_construct_model(use_logitnormal_sticks = False, 
                                                    k_approx = k_approx)
                
        # get ez
        ez = posterior_quantities_lib.get_ez_all(g_obs, vb_params_dict, gh_loc, gh_weights)
        
        e_num_clusters_analytic = cluster_quantities_lib.\
            get_e_num_clusters_from_ez_analytic(ez.reshape(-1, k_approx))
                
        
        # sampled expected number of clusters
        n_samples = 10000
        n_clusters_sampled = \
            posterior_quantities_lib.get_e_num_clusters(g_obs, 
                                                        vb_params_dict,
                                                        gh_loc,
                                                        gh_weights, 
                                                        n_samples = n_samples, 
                                                        return_samples = True)
        
        # monte-carlo estimate
        e_num_clusters_emp = n_clusters_sampled.mean()
        
        tol = 3 * np.sqrt(n_clusters_sampled.var() / n_samples)
        
        testing.assert_allclose(e_num_clusters_emp, 
                                e_num_clusters_analytic, 
                                rtol = 0, 
                                atol = tol)

if __name__ == '__main__':
    unittest.main()
