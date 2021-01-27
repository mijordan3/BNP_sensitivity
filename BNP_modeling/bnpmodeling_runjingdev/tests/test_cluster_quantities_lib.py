#!/usr/bin/env python3

import sys

import numpy as onp
onp.random.seed(453453)

import jax
import jax.numpy as np
import jax.scipy as sp

from bnpmodeling_runjingdev import cluster_quantities_lib

import unittest

import numpy.testing as testing

class TestGetMixtureWeights(unittest.TestCase):
    def test_get_mixture_weights_from_stick_break_propns(self): 
        # tests our vectorized computatation 
        # against a more readable for-loop implementation 
        # of the stick-breaking process
        
        # randomly set some stick-breaking proportions
        n_obs = 3
        k_approx = 10
        stick_breaking_propn = onp.random.rand(n_obs, k_approx - 1) 
        
        mixture_weights = \
            cluster_quantities_lib.\
                get_mixture_weights_from_stick_break_propns(stick_breaking_propn)
        
        assert mixture_weights.shape[0] == stick_breaking_propn.shape[0]
        assert mixture_weights.shape[1] == k_approx
        testing.assert_allclose(mixture_weights.sum(1), 1, rtol = 1e-6)
        
        # use a for-loop to describe the stick-breaking process
        # check against our vectorized implementation
        stick_remain = np.ones(n_obs)
        for k in range(k_approx - 1): 
            
            x = stick_breaking_propn[:, k] * stick_remain
            y = mixture_weights[:, k] 
            
            testing.assert_allclose(x, y, rtol = 0, atol = 1e-8)
            
            stick_remain *= (1 - stick_breaking_propn[:, k])
        
        # check last stick
        testing.assert_allclose(mixture_weights[:, -1], 
                                stick_remain, 
                                rtol = 0, 
                                atol = 1e-8)

class TestEzSamples(unittest.TestCase): 
    def test_sample_ez(self): 
        # check the sampling of e_z from uniform samples
        
        # randomly set some ezs
        n_obs = 5
        k_approx = 3
        e_z = onp.random.rand(n_obs, k_approx)
        e_z /= e_z.sum(-1, keepdims = True)
        
        # now draw e_z conditional on some uniform samples
        n_samples = 100000        
        z_samples = cluster_quantities_lib.sample_ez(e_z, 
                                                     n_samples = n_samples, 
                                                     seed = 342)
        
                
        empirical_propns = z_samples.mean(0)
        
        for n in range(n_obs):
            for k in range(k_approx): 
                
                truth = e_z[n,k]
                emp = empirical_propns[n,k] 
                
                tol = 3 * np.sqrt(truth * (1 - truth) / n_samples)
                
                testing.assert_allclose(truth, emp, rtol = 0, atol = tol)
        
class TestExpectedNumClusters(unittest.TestCase):

    def test_get_e_num_clusters_from_ez(self):
        # check that get_e_num_clusters_from_ez, which computes
        # the expected number of clusters via MCMC matches the
        # analytic expectation

        # construct ez
        n_obs = 5
        k_approx = 3        
        e_z = onp.random.rand(n_obs, k_approx)
        e_z /= e_z.sum(-1, keepdims = True)
        
        # get expected number of clusters via sampling
        n_samples = 10000
        e_num_clusters_sampled, var_num_clusters_sampled = \
            cluster_quantities_lib.\
                get_e_num_large_clusters_from_ez(e_z,
                                                 n_samples = n_samples,
                                                 seed = 1342342,
                                                 threshold = 0)
        
        # get analytic expected nubmer of clusters
        e_num_clusters_analytic = \
            cluster_quantities_lib.get_e_num_clusters_from_ez(e_z)
        
        tol =  np.sqrt(var_num_clusters_sampled / n_samples) * 3

        testing.assert_allclose(e_num_clusters_analytic,
                                e_num_clusters_sampled,
                                atol = tol,
                                rtol = 0)
        
    def test_pred_clusters_from_mixture_weights(self): 
        
        # TODO need to test this
        
        # for now, just make sure it runs
        k_approx = 8
        stick_propn_mean = onp.random.randn(k_approx - 1)
        stick_propn_info = np.exp(onp.random.randn(k_approx - 1))
        _ = cluster_quantities_lib.\
            get_e_num_pred_clusters_from_logit_sticks(stick_propn_mean, 
                                                      stick_propn_info,
                                                      n_obs = 10, 
                                                      threshold = 0,
                                                      n_samples = 100)
        
        
        
#         # this mimicks the computation in 
#         # get_e_num_pred_clusters_from_logit_sticks
        
#         # construct sampled weights 
#         n_samples = 10
#         k_approx = 5
#         weight_samples = onp.random.rand(n_samples, k_approx)
#         weight_samples /= weight_samples.sum(-1, keepdims = True)
        
#         # get predicted number of 
#         for threshold in range(k_approx): 
#             n_clusters_sampled = cluster_quantities_lib.\
#                 get_e_num_pred_clusters_from_mixture_weights(weight_samples, 
#                                                              n_obs = 10, 
#                                                              threshold = threshold)
            
            
        
        
if __name__ == '__main__':
    unittest.main()
