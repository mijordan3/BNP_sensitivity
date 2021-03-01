import unittest

import jax
import jax.numpy as np
import jax.scipy as sp

from numpy.polynomial.hermite import hermgauss
from scipy.optimize import linear_sum_assignment

import paragami

from bnpreg_runjingdev import regression_mixture_lib
from bnpreg_runjingdev import regression_optimization_lib as reg_optim_lib
from bnpreg_runjingdev import regression_posterior_quantities as reg_posterior_quantities

from bnpreg_runjingdev.genomics_data_utils import spline_bases_lib

class TestRegMixtureOptimization(unittest.TestCase):
    
    def test_reg_mixture(self): 
        
        ###############
        # Draw data
        ###############
        
        # define timepoints
        n_rep = 3 # replcates per time points
        n_unique_timepoints = 14 # number of timepoints

        timepoints = np.array([[i, ] * n_rep for i in range(n_unique_timepoints)]).flatten()

        # get regressors
        regressors = spline_bases_lib.get_genomics_spline_basis(timepoints,
                                                                df=7, 
                                                                degree=3)

        dim_reg = regressors.shape[-1]

        # draw cluster memberships
        true_k = 3
        n_obs = 1000
        true_z = jax.random.categorical(key = jax.random.PRNGKey(26), 
                                        # decreasing cluster size
                                        logits = np.array([-i * 0.5 for i in range(true_k)]), 
                                        shape = (n_obs, ))
        true_z_onehot = jax.nn.one_hot(true_z, true_k)
        
        # draw true regression coefficients
        beta_var = 10.
        true_beta = jax.random.normal(key = jax.random.PRNGKey(901), 
                                      shape = (true_k, dim_reg)) * np.sqrt(beta_var)
        
        # draw random shifts
        shift_var = 10.
        true_shifts = jax.random.normal(key = jax.random.PRNGKey(901), 
                                      shape = (n_obs,)) * np.sqrt(shift_var)

        # sample individuals
        beta_n = np.dot(true_z_onehot, true_beta)

        true_data_info = 5.

        noise = jax.random.normal(key = jax.random.PRNGKey(3453), 
                                  shape = (n_obs, len(timepoints))) * 1 / np.sqrt(true_data_info)

        y = np.dot(beta_n, regressors.transpose()) + noise + true_shifts[:, None]

        ###############
        # set prior
        ###############
        prior_params_dict, vb_params_paragami = regression_mixture_lib.get_default_prior_params()
        prior_params_dict['dp_prior_alpha'] = 6.0
        prior_params_dict['prior_shift_info'] = 1 / shift_var
        prior_params_dict['prior_centroid_info'] = 1 / beta_var
        
        ###############
        # initialize (randomly) vb parameters
        ###############
        k_approx = 15
        vb_params_dict, vb_params_paragami = \
            regression_mixture_lib.get_vb_params_paragami_object(dim_reg, k_approx)
        
        # Gauss-Hermite points for integrating logitnormal stick-breaking prior
        gh_deg = 8
        gh_loc, gh_weights = hermgauss(gh_deg)

        # convert to jax arrays
        gh_loc, gh_weights = np.array(gh_loc), np.array(gh_weights)
                
        ###############
        # optimize
        ###############
        vb_opt_dict, vb_opt, ez_opt, out, optim_time = \
            reg_optim_lib.optimize_regression_mixture(y, regressors, 
                                                      vb_params_dict,
                                                      vb_params_paragami,
                                                      prior_params_dict, 
                                                      gh_loc, 
                                                      gh_weights, 
                                                      run_newton=True)
        
        #############
        # Check inferred quantities
        #############
        # get estimated shifts
        est_shifts = reg_posterior_quantities.get_optimal_local_params_from_vb_dict(y, 
                                                                                    regressors, 
                                                                                    vb_opt_dict, 
                                                                                    prior_params_dict, 
                                                                                    gh_loc, 
                                                                                    gh_weights)[2]
        which_col = ez_opt == ez_opt.max(1)[:, None] # one-hot encoding of estimated memberships
        est_shifts = (est_shifts * which_col).sum(1) # extract shifts 
        assert np.abs(est_shifts - true_shifts).max() < 0.5 #check close
        
        
        # get only the clusters present in the posterior
        which_clusters = np.unique(ez_opt.argmax(1))

        # estimated parameters
        est_beta = vb_opt_dict['centroids'][which_clusters, :]
        est_ez = ez_opt[:, which_clusters]
        
        
        # find minimizing permutation
        # of each est cluster to true cluster

        diffs = ((est_beta[None, :, :] - \
                  true_beta[:, None, :])**2).sum(-1)

        perm = linear_sum_assignment(diffs)[1]
        
        # check centroids
        centroid_diff = est_beta[perm] - true_beta
        
        assert np.abs(centroid_diff).max() < 0.2
        
        # check estimated memberships
        est_z_permed = est_ez[:, perm].argmax(1)
        assert (est_z_permed == true_z).mean() == 1.
        
        # check mixture weights
        est_weights = reg_posterior_quantities.get_e_mixture_weights_from_vb_dict(vb_opt_dict, gh_loc, gh_weights)
        est_weights = est_weights[which_clusters][perm]
        
        assert np.abs(est_weights - true_z_onehot.mean(0)).max() < 0.01