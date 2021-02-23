import unittest

import jax
import jax.numpy as np
import jax.scipy as sp

from numpy.polynomial.hermite import hermgauss

import paragami

import bnpgmm_runjingdev.gmm_clustering_lib as gmm_lib
import bnpgmm_runjingdev.gmm_cavi_lib as cavi_lib
from bnpgmm_runjingdev.utils_lib import load_iris_data, get_param_indices

# loads iris data
y, _ = load_iris_data()
n_obs = y.shape[0]
dim = y.shape[1]

# set up prior
# Get priors
_, prior_params_paragami = gmm_lib.get_default_prior_params(dim)

prior_params_dict = prior_params_paragami.random()

prior_params_dict['prior_wishart_df'] += y.shape[1]

# set up variational distribution
k_approx = 8
gh_deg = 8
gh_loc, gh_weights = hermgauss(gh_deg)
vb_params_dict, vb_params_paragami = \
    gmm_lib.get_vb_params_paragami_object(dim, k_approx)

# kl loss function 
get_vb_params_loss = paragami.FlattenFunctionInput(
                        original_fun=gmm_lib.get_kl,
                        patterns = vb_params_paragami,
                        free = True,
                        argnums = 1)

class TestCaviUpdates(unittest.TestCase):
    
    def test_centroid_updates(self): 
        
        # get optimal e_z's
        e_z = gmm_lib.get_optimal_z_from_vb_dict(y, vb_params_dict,
                                                 gh_loc, gh_weights)
        
        # update centroids 
        centroid_step = cavi_lib.update_centroids(y, e_z, prior_params_dict)
        vb_params_dict['cluster_params']['centroids'] = centroid_step
        
        # get loss and gradients
        get_loss = lambda x : get_vb_params_loss(y, x,
                                                 prior_params_dict,
                                                 gh_loc, gh_weights, 
                                                 e_z = e_z)
        get_grad = jax.grad(get_loss)
    
        grad = get_grad(vb_params_paragami.flatten(vb_params_dict, free = True))
        
        # check centriod gradients
        centroid_indx = [get_param_indices('centroids',
                                           vb_params_dict,
                                           vb_params_paragami)]
        grad_norm = np.abs(grad[centroid_indx]).max()
        assert grad_norm < 1e-5, grad_norm
        
        # update infos
        cluster_info_step = cavi_lib.update_cluster_info(y, e_z, centroid_step, prior_params_dict)
        vb_params_dict['cluster_params']['cluster_info'] = cluster_info_step
        
        # get gradient again
        grad = get_grad(vb_params_paragami.flatten(vb_params_dict, free = True))
        
        info_indx = [get_param_indices('cluster_info',
                                       vb_params_dict,
                                       vb_params_paragami)]
        grad_norm = np.abs(grad[info_indx]).max()
        assert grad_norm < 1e-5, grad_norm

        
    def test_stick_psloss(self):
        
        # randomly initialize vb parameters

        # get e_z
        e_z = gmm_lib.get_optimal_z_from_vb_dict(y, vb_params_dict,
                                                        gh_loc, gh_weights)

        stick_free_param = \
            vb_params_paragami['stick_params'].flatten(vb_params_dict['stick_params'],
                                                                  free = True)

        # we check that the gradient of the loss is that same as the gradient of
        # the stick pseudo-loss

        # grad of stick pseudo-loss
        get_stick_psgrad = jax.grad(cavi_lib._get_sticks_psloss, 0)
        
        stick_grad = get_stick_psgrad(stick_free_param, 
                                        vb_params_paragami['stick_params'], 
                                        e_z,
                                        prior_params_dict,
                                        gh_loc,
                                        gh_weights)
        
        def stick_loss(stick_free_param): 
            stick_params_dict = \
                vb_params_paragami['stick_params'].fold(stick_free_param, 
                                                        free = True)
            
            vb_params_dict['stick_params'] = stick_params_dict
            
            return gmm_lib.get_kl(y, vb_params_dict,
                                  prior_params_dict,
                                  gh_loc, gh_weights, 
                                  e_z = e_z)
        
        get_stick_grad = jax.grad(stick_loss)
        stick_grad_true = get_stick_grad(stick_free_param)
                
        diff = np.abs(stick_grad - stick_grad_true).max()
        assert diff < 1e-10, diff
    
    def test_ez_updates(self): 
        
        def get_kl_from_z_nat_param(z_nat_param):

            log_const = sp.special.logsumexp(z_nat_param, axis=1)
            e_z = np.exp(z_nat_param - np.expand_dims(log_const, axis = 1))

            return gmm_lib.get_kl(y, vb_params_dict, prior_params_dict,
                            gh_loc, gh_weights,
                            e_z = e_z)
        
        stick_propn_mean = vb_params_dict['stick_params']['stick_means']
        stick_propn_info = vb_params_dict['stick_params']['stick_infos']
        centroids = vb_params_dict['cluster_params']['centroids']
        cluster_info = vb_params_dict['cluster_params']['cluster_info']

        z_nat_param = \
            gmm_lib.get_z_nat_params(y, 
                                     stick_propn_mean, stick_propn_info, 
                                     centroids, cluster_info,
                                     gh_loc, gh_weights)

        get_grad = jax.grad(get_kl_from_z_nat_param)
        grad = get_grad(z_nat_param)
        
        grad_norm = np.abs(grad).max()
        assert grad_norm < 1e-8
    
    def test_run_cavi_to_convergence(self):
        
        # run CAVI until convergence, and check the gradient 
                
        vb_opt = cavi_lib.run_cavi(y, vb_params_dict,
                              vb_params_paragami, prior_params_dict,
                              gh_loc, gh_weights,
                              debug = False, 
                              x_tol = 1e-8)[0]
        
        # set up gradient
        get_loss = lambda x : get_vb_params_loss(y, x, prior_params_dict, gh_loc, gh_weights)
        get_grad = jax.grad(get_loss)
        
        # evaluate gradient at optimum
        grad_at_opt = get_grad(vb_params_paragami.flatten(vb_opt, free = True))
        
        # assert gradient is small 
        grad_norm = np.abs(grad_at_opt).max()
        assert grad_norm < 1e-5, grad_norm

    def test_run_cavi_debugger(self):
        
        # with debug = True, checks the KL at every iteraiton. 
        # this is quite slow, so we run only 10 iterations
        
        # I could probably jit the KL function in `run_cavi` ... 
                
        _ = cavi_lib.run_cavi(y, vb_params_dict,
                              vb_params_paragami, prior_params_dict,
                              gh_loc, gh_weights,
                              debug = True, 
                              max_iter = 10)
