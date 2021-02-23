import unittest

import jax
import jax.numpy as np
import jax.scipy as sp

from numpy.polynomial.hermite import hermgauss

import paragami

import bnpgmm_runjingdev.gmm_clustering_lib as gmm_lib
from bnpgmm_runjingdev.utils_lib import load_iris_data

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

class TestGMMClustering(unittest.TestCase):
    
    
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
    
