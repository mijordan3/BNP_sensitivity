import unittest

import jax
import jax.numpy as np
import jax.scipy as sp

from numpy.polynomial.hermite import hermgauss
from scipy.optimize import linear_sum_assignment

import paragami

import bnpgmm_runjingdev.gmm_clustering_lib as gmm_lib
import bnpgmm_runjingdev.gmm_optimization_lib as gmm_optim_lib
from bnpgmm_runjingdev.gmm_posterior_quantities_lib import get_e_mixture_weights_from_vb_dict


class TestGMMOptimization(unittest.TestCase):
    
    
    def test_gmm_optimization_on_toy_data(self): 
        
        ###############
        # Draw data
        ###############
        
        # draw cluster belongings
        dim = 2
        true_k = 3
        n_obs = 1000
        true_z = jax.random.categorical(key = jax.random.PRNGKey(45345), 
                                        # decreasing cluster size
                                        logits = np.array([-i * 0.5 for i in range(true_k)]), 
                                        shape = (n_obs, ))
        true_z_onehot = jax.nn.one_hot(true_z, true_k)

        # draw centroids
        true_centroids = np.array([np.ones(dim) * i for i in range(true_k)])

        # draw data
        sd = 0.15
        y = np.dot(true_z_onehot, true_centroids) + \
                    jax.random.normal(key = jax.random.PRNGKey(453453), 
                                      shape = (n_obs, 2)) * sd
        
        ###############
        # set prior
        ###############
        prior_params_dict, prior_params_paragami = gmm_lib.get_default_prior_params(dim)
        
        ###############
        # initialize (randomly) vb parameters
        ###############
        k_approx = 15
        # Gauss-Hermite points for integrating logitnormal stick-breaking prior
        gh_deg = 8
        gh_loc, gh_weights = hermgauss(gh_deg)

        # convert to jax arrays
        gh_loc, gh_weights = np.array(gh_loc), np.array(gh_weights)
        
        vb_params_dict, vb_params_paragami = gmm_lib.get_vb_params_paragami_object(dim, 
                                                                                   k_approx)        
        
        ###############
        # optimize
        ###############
        vb_opt_dict, vb_opt, e_z_opt, out, _ = gmm_optim_lib.optimize_gmm(y,
                                                                 vb_params_dict,
                                                                 vb_params_paragami,
                                                                 prior_params_dict, 
                                                                 gh_loc, gh_weights)
        
        #############
        # Check inferred quantities
        #############
        
        # match estimated w truth
        which_clusters = np.unique(e_z_opt.argmax(1))
        est_ez = e_z_opt[:, which_clusters]
        est_centroids = vb_opt_dict['cluster_params']['centroids'][which_clusters, :]
        
        diffs = ((est_centroids[None, :, :] - \
          true_centroids[:, None, :])**2).sum(-1)
        
        perm = linear_sum_assignment(diffs)[1]
        
        # check centroids
        centroid_diff = est_centroids[perm] - true_centroids
        
        assert np.abs(centroid_diff).max() < 0.05, centroid_diff
        
        # check inferred zs
        assert np.all(est_ez[:, perm].argmax(1) == true_z)
        
        # check mixture weights
        mixture_weights = get_e_mixture_weights_from_vb_dict(vb_opt_dict, 
                                                             gh_loc,
                                                             gh_weights)
                
        mixture_diffs = mixture_weights[which_clusters][perm] - \
                            true_z_onehot.mean(0)
            
        assert np.abs(mixture_diffs).max() < 0.01, mixture_diffs