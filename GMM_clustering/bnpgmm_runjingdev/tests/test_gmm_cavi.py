import unittest

import autograd
import autograd.numpy as np
import autograd.scipy as sp

from numpy.polynomial.hermite import hermgauss

import paragami

from copy import deepcopy

import bnpgmm_runjingdev.gmm_clustering_lib as gmm_lib
import bnpgmm_runjingdev.gmm_cavi_lib as cavi_lib
import bnpgmm_runjingdev.simulation_lib as simulation_lib

from bnpgmm_runjingdev.utils_lib import get_param_indices

import bnpmodeling_runjingdev.optimization_lib as optimization_lib

np.random.seed(35345)

# set up data
n_obs = 1000
dim = 2
true_k = 5
y = simulation_lib.simulate_data(n_obs, dim,
                    true_k, separation=0.2)[0]

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


class TestCaviUpdates(unittest.TestCase):
    def test_cavi_updates(self):
        # randomly initialize vb parameters
        vb_params_dict = vb_params_paragami.random()

        # get e_z: this will be fixed for this test case
        e_z = gmm_lib.get_optimal_z_from_vb_params_dict(y, vb_params_dict,
                                                        gh_loc, gh_weights)

        # get loss as a function of vb parameters
        get_vb_params_loss = paragami.FlattenFunctionInput(
                                        original_fun=gmm_lib.get_kl,
                                        patterns = vb_params_paragami,
                                        free = True,
                                        argnums = 1)

        get_loss = lambda x : get_vb_params_loss(y, x, prior_params_dict,
                            gh_loc, gh_weights, e_z = e_z)

        grad_loss = autograd.grad(get_loss)

        # fixed e_z: get optimal parameters by running newton
        vb_opt = optimization_lib.optimize_full(get_loss,
                    vb_params_paragami.flatten(vb_params_dict, free = True),
                    bfgs_max_iter = 500, netwon_max_iter = 50,
                    max_precondition_iter = 10,
                    gtol=1e-8, ftol=1e-8, xtol=1e-8)
        vb_opt_dict = vb_params_paragami.fold(vb_opt, free = True)

        # check centroid updates
        centroid_step = cavi_lib.update_centroids(y, e_z, prior_params_dict)
        vb_params_dict['cluster_params']['centroids'] = centroid_step

        # get gradient, assert it is zero
        vb_params_flatten = vb_params_paragami.flatten(vb_params_dict, free = True)
        grad = np.abs(grad_loss(vb_params_flatten)[get_param_indices('centroids',
                                                    vb_params_dict,
                                                    vb_params_paragami)]).max()
        assert grad < 1e-10

        # check centroid info updates
        vb_params_dict['cluster_params']['cluster_info'] = \
            cavi_lib.update_cluster_info(y, e_z,
                                        vb_params_dict['cluster_params']['centroids'],
                                        prior_params_dict)
        vb_params_flatten = vb_params_paragami.flatten(vb_params_dict, free = True)
        # get grad
        grad = np.abs(grad_loss(vb_params_flatten)[get_param_indices('cluster_info',
                                                                     vb_params_dict,
                                                                     vb_params_paragami)]).max()
        assert grad < 1e-10

        # check against newton results
        centroid_diff = np.abs(vb_opt_dict['cluster_params']['centroids'] - \
                        vb_params_dict['cluster_params']['centroids']).max()
        info_diff = np.abs(vb_opt_dict['cluster_params']['cluster_info'] - \
                      vb_params_dict['cluster_params']['cluster_info']).max()

        assert centroid_diff < 1e-5, centroid_diff
        assert info_diff < 1e-5, info_diff

    def test_stick_psloss(self):
        # randomly initialize vb parameters
        vb_params_dict = vb_params_paragami.random()
        # get e_z
        e_z = gmm_lib.get_optimal_z_from_vb_params_dict(y, vb_params_dict,
                                                        gh_loc, gh_weights)

        init_stick_free_param = \
            vb_params_paragami['stick_params'].flatten(vb_params_dict['stick_params'],
                                                                  free = True)

        # we check that the gradient of the loss is that same as the gradient of
        # the stick pseudo-loss

        # grad of full loss
        get_stick_grad = autograd.elementwise_grad(cavi_lib._get_sticks_loss, 1)
        get_stick_hess = autograd.hessian(cavi_lib._get_sticks_loss, 1)

        stick_grad = get_stick_grad(y, init_stick_free_param,
                                    vb_params_paragami['stick_params'],
                                        e_z, vb_params_dict, prior_params_dict,
                                        gh_loc, gh_weights)
        stick_hess = get_stick_hess(y, init_stick_free_param,
                                        vb_params_paragami['stick_params'],
                                        e_z, vb_params_dict,
                                        prior_params_dict, gh_loc, gh_weights)

        # grad of pseudo-loss
        get_stick_grad2 = autograd.elementwise_grad(cavi_lib._get_sticks_psloss, 1)
        get_stick_hess2 = autograd.hessian(cavi_lib._get_sticks_psloss, 1)

        stick_grad2 = get_stick_grad2(y, init_stick_free_param,
                                    vb_params_paragami['stick_params'],
                                        e_z, vb_params_dict, prior_params_dict,
                                        gh_loc, gh_weights)
        stick_hess2 = get_stick_hess2(y, init_stick_free_param,
                                        vb_params_paragami['stick_params'],
                                        e_z, vb_params_dict,
                                        prior_params_dict, gh_loc, gh_weights)

        assert np.abs(np.max(stick_grad - stick_grad2)) < 1e-12
        assert np.abs(np.max(stick_hess - stick_hess2)) < 1e-12



    def test_run_cavi(self):
        vb_params_dict = vb_params_paragami.random()
        _ = cavi_lib.run_cavi(y, vb_params_dict,
                                            vb_params_paragami, prior_params_dict,
                                            gh_loc, gh_weights,
                                           debug = True)
