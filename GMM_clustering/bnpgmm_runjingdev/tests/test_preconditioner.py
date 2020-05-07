#!/usr/bin/env python3

import autograd
import autograd.numpy as np
import autograd.scipy as sp

import scipy as osp

from bnpgmm_runjingdev import gmm_preconditioner_lib as preconditioner_lib

import unittest

import numpy.testing as testing


class TestNystromWoodbury(unittest.TestCase):
    def test_nystrom_woodbury(self):

        a = np.random.randn(3, 3)
        A = np.eye(3) + np.dot(a, a.transpose())

        woodb_inv_fun, C, A_block = \
            preconditioner_lib.get_nystrom_woodbury_approx(A, indx = np.array([0, 1]))

        woodb_inv = woodb_inv_fun(np.eye(3))

        W_inv = np.linalg.inv(A_block)
        A_tilde = np.dot(np.dot(C, W_inv), C.transpose())

        assert np.abs(np.linalg.inv(np.eye(3) - A_tilde) -
                        woodb_inv).max() < 1e-12

# TODO
# tests of MFVB preconditioner contained in preconditioner_sandbox.ipynb ... need to move over here

if __name__ == '__main__':
    unittest.main()
