import unittest

import autograd
from autograd import numpy as np
from autograd import scipy as sp

import scipy as osp

from numpy.polynomial.hermite import hermgauss

np.random.seed(453453)

import paragami

# BNP sensitivity libraries
import bnpgmm_runjingdev.gmm_clustering_lib as gmm_lib
import bnpgmm_runjingdev.utils_lib as utils_lib
import bnpgmm_runjingdev.gmm_cavi_lib as cavi_lib
import bnpgmm_runjingdev.gmm_preconditioner_lib as preconditioner_lib
import bnpgmm_runjingdev.hessian_lib as hessian_lib

# load iris data
dataset_name = 'iris'
features, iris_species = utils_lib.load_data()
dim = features.shape[1]
n_obs = len(iris_species)

# get prior
prior_params_dict, prior_params_paragami = gmm_lib.get_default_prior_params(dim)
prior_params_dict['alpha'] = np.array([3.0])

prior_params_free = prior_params_paragami.flatten(prior_params_dict, free = True)

# get vb parameters
k_approx = 6
gh_deg = 8
gh_loc, gh_weights = hermgauss(gh_deg)
vb_params_dict, vb_params_paragami = gmm_lib.get_vb_params_paragami_object(dim, k_approx)

# optimize
n_kmeans_init = 10
init_vb_free_params, init_vb_params_dict, init_ez = \
    utils_lib.cluster_and_get_k_means_inits(features, vb_params_paragami,
                                                n_kmeans_init = n_kmeans_init,
                                                seed = 3445)
vb_opt_dict, e_z_opt = cavi_lib.run_cavi(features, vb_params_dict,
                                            vb_params_paragami, prior_params_dict,
                                            gh_loc, gh_weights,
                                            debug = False)
vb_opt = vb_params_paragami.flatten(vb_opt_dict, free = True)

# get objective
get_vb_params_loss = paragami.FlattenFunctionInput(
                                original_fun=gmm_lib.get_kl,
                                patterns = [vb_params_paragami, prior_params_paragami],
                                free = True,
                                argnums = [1, 2])

objective_fun = lambda x, y : get_vb_params_loss(features, x, y, gh_loc, gh_weights)
obj_fun_hessian = autograd.hessian(objective_fun, argnum=0)

class TestBlockHessian(unittest.TestCase):
    def test_get_large_clusters_hessian(self):

        true_hess = obj_fun_hessian(vb_opt, prior_params_free)

        which_k = [0, 3, 5]
        est_hess, indx = \
            hessian_lib.get_large_clusters_hessian(features, which_k,
                                        vb_opt, vb_params_paragami,
                                        prior_params_dict,
                                        gh_loc, gh_weights)

        # check nonzero entires of estimated hessian matches with actual hessian
        assert np.abs(est_hess[indx][:, indx] - \
                        true_hess[indx][:, indx]).max() < 1e-12

        # # check the index
        # which_included = np.full(len(vb_opt), False)
        # which_included[indx] = True
        # bool_dict = vb_params_paragami.fold(which_included,
        #                             free = True,
        #                             validate_value = False)
        #
        # not_which_k = np.full(k_approx, True)
        # not_which_k[which_k] = False
        # assert np.all(bool_dict['cluster_params']['centroids'][:, which_k] == True)
        # print(bool_dict['cluster_params']['centroids'])
        # print(bool_dict['cluster_params']['cluster_info'])
        # assert np.all(bool_dict['cluster_params']['cluster_info'][which_k] == True)
        # assert np.all(bool_dict['cluster_params']['centroids'][:, not_which_k] == False)
        # assert np.all(bool_dict['cluster_params']['cluster_info'][not_which_k] == False)
        #
        # not_which_k = np.full(k_approx, True)
        # not_which_k[which_k] = False
        # assert np.all(bool_dict['cluster_params']['centroids'][:, which_k] == True)
        # assert np.all(bool_dict['cluster_params']['cluster_info'][which_k] == True)
        # assert np.all(bool_dict['cluster_params']['centroids'][:, not_which_k] == False)
        # assert np.all(bool_dict['cluster_params']['cluster_info'][not_which_k] == False)
