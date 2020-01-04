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

        # get vb parameters
        stick_propn_mean = vb_params_dict['stick_params']['stick_propn_mean']
        stick_propn_info = vb_params_dict['stick_params']['stick_propn_info']
        centroids = vb_params_dict['cluster_params']['centroids']
        cluster_info = vb_params_dict['cluster_params']['cluster_info']

        # get e_z: this will be fixed for this test case
        e_z, _ = gmm_lib.get_optimal_z(y, stick_propn_mean, stick_propn_info,
                                        centroids, cluster_info,
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

    def test_run_cavi(self):
        vb_params_dict = vb_params_paragami.random()
        _ = cavi_lib.run_cavi(y, vb_params_dict,
                                            vb_params_paragami, prior_params_dict,
                                            gh_loc, gh_weights,
                                           debug = True)

def get_param_indices(param_str, vb_params_dict, vb_params_paragami):
    bool_dict = deepcopy(vb_params_dict)
    for k in vb_params_dict.keys():
        for j in vb_params_dict[k].keys():
            if j == param_str:
                bool_dict[k][j] = (vb_params_dict[k][j] == vb_params_dict[k][j])
            else:
                bool_dict[k][j] = (vb_params_dict[k][j] != vb_params_dict[k][j])

    return vb_params_paragami.flat_indices(bool_dict, free = True)
