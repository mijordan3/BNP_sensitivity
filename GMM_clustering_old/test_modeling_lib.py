#!/usr/bin/env python3

import autograd.numpy as np
import autograd.scipy as sp

import modeling_lib
import unittest

np.random.seed(24524)

class TestClusteringSamples(unittest.TestCase):
    def test_clustering_samples(self):
        # cluster belonging probabilities
        n_obs = 5
        n_clusters = 3
        e_z = np.random.random((n_obs, n_clusters))
        e_z = e_z / np.sum(e_z, axis = 1, keepdims = True)
        e_z_cumsum = e_z.cumsum(1)

        # draw uniform samples
        n_samples = 100000
        unif_samples = np.random.random((n_obs, n_samples))

        # get cluster belongings from uniform samples
        z_ind_samples = modeling_lib.get_clusters_from_ez_and_unif_samples(\
                                        e_z_cumsum, unif_samples)

        # sample
        e_z_sampled = np.zeros(e_z.shape)
        for i in range(n_clusters):
            e_z_sampled[:, i] = (z_ind_samples == i).mean(axis = 1)

        tol = 3 * np.sqrt(e_z * (1 - e_z) / n_samples)
        print('e_z diff', np.max(np.abs(e_z_sampled - e_z)))
        assert np.all(np.abs(e_z_sampled - e_z) < tol)

    def test_get_e_num_clusters_from_ez(self):
        n_obs = 5
        n_clusters = 3
        e_z = np.random.random((n_obs, n_clusters))
        e_z = e_z / np.sum(e_z, axis = 1, keepdims = True)

        e_num_clusters_sampled, var_num_clusters_sampled = \
            modeling_lib.get_e_num_large_clusters_from_ez(e_z,
                                                    n_samples = 10000,
                                                    unif_samples = None,
                                                    threshold = 0.0)

        e_num_clusters_analytic = modeling_lib.get_e_number_clusters_from_ez(e_z)

        print('e_num_clusters diff',
                np.abs(e_num_clusters_analytic - e_num_clusters_sampled))
        assert np.abs(e_num_clusters_analytic - e_num_clusters_sampled) < \
                    np.sqrt(var_num_clusters_sampled) * 3






if __name__ == '__main__':
    unittest.main()
