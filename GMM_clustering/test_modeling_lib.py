#!/usr/bin/env python3

import autograd.numpy as np
import autograd.scipy as sp

from autograd import elementwise_grad

import modeling_lib 
import unittest

np.random.seed(24524)

class TestClusteringSamples(unittest.TestCase):
    def test_clustering_samples(self):
        # cluster belonging probabilities
        n = 5
        e_z = np.random.random((n, 3))
        e_z = e_z / np.sum(e_z, axis = 1, keepdims = True)

        # sample
        e_z_mean = np.zeros(e_z.shape)
        n_samples = 10000
        e_z_cumsum = e_z.cumsum(1)
        for i in range(n_samples):
            # draw uniform samples
            unif_samples = np.random.rand(n)
            # get cluster belongings
            e_z_mean += modeling_lib.get_clusters_from_ez_and_unif_sample(\
                                    e_z_cumsum, unif_samples)
        e_z_mean /= n_samples

        tol = 3 * np.sqrt(e_z * (1 - e_z) / n_samples)
        assert np.all(np.abs(e_z_mean - e_z) < tol)


if __name__ == '__main__':
    unittest.main()
