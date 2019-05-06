import autograd

import autograd.numpy as np
import autograd.scipy as sp
from numpy.polynomial.hermite import hermgauss

import scipy as osp

import sys
sys.path.insert(0, '../')

import structure_model_lib
import structure_optimization_lib as str_opt_lib
import preconditioner_lib

import paragami
import vittles

from copy import deepcopy

import argparse
import distutils.util

import os

import data_utils

from BNP_modeling import cluster_quantities_lib, modeling_lib
import BNP_modeling.optimization_lib as opt_lib

import unittest

import numpy.testing as testing

np.random.seed(25465)

# Test my system solver, and make sure it interacts with
# HyperparameterSensitivityLinearApproximation in the way that I expect
class TestSystemSolver(unittest.TestCase):
    def test_system_solver(self):
        # draw data
        n_obs = 10
        n_loci = 5
        n_pop = 3

        g_obs = data_utils.draw_data(n_obs, n_loci, n_pop)[0]

        # prior parameters
        prior_params_dict, prior_params_paragami = \
            structure_model_lib.get_default_prior_params()

        dp_prior_alpha = prior_params_dict['dp_prior_alpha']
        allele_prior_alpha = prior_params_dict['allele_prior_alpha']
        allele_prior_beta = prior_params_dict['allele_prior_beta']

        # vb params
        k_approx = 12
        gh_deg = 8
        gh_loc, gh_weights = hermgauss(gh_deg)

        use_logitnormal_sticks = False

        vb_params_dict, vb_params_paragami = \
            structure_model_lib.get_vb_params_paragami_object(n_obs, n_loci,
                k_approx, use_logitnormal_sticks)

        # initialize
        vb_params_dict = \
            structure_model_lib.set_init_vb_params(g_obs, k_approx,
                                                    vb_params_dict,
                                                    use_logitnormal_sticks)

        # optimize
        vb_opt_free_params = \
            str_opt_lib.optimize_structure(g_obs, vb_params_dict,
                                        vb_params_paragami,
                                        prior_params_dict,
                                        gh_loc, gh_weights,
                                        use_logitnormal_sticks,
                                        run_cavi = True,
                                        cavi_max_iter = 100,
                                        cavi_tol = 1e-2,
                                        netwon_max_iter = 20,
                                        max_precondition_iter = 25,
                                        gtol=1e-8, ftol=1e-8, xtol=1e-8,
                                        approximate_hessian = True)

        vb_params_dict = vb_params_paragami.fold(vb_opt_free_params, free=True)

        # set up objective function
        get_kl_from_vb_free_prior_free = \
            paragami.FlattenFunctionInput(original_fun=structure_model_lib.get_kl,
                        patterns = [vb_params_paragami, prior_params_paragami],
                        free = True,
                        argnums = [1, 2])

        objective_fun = lambda x, y: get_kl_from_vb_free_prior_free(g_obs,
                                x, y, use_logitnormal_sticks,
                                gh_loc, gh_weights)
        prior_free_params = \
            prior_params_paragami.flatten(prior_params_dict, free=True)

        print('\n default hessian computation: ')
        vb_sens = \
            vittles.HyperparameterSensitivityLinearApproximation(
                                        objective_fun = objective_fun,
                                        opt_par_value = vb_opt_free_params,
                                        hyper_par_value = prior_free_params,
                                        validate_optimum=False,
                                        factorize_hessian=True,
                                        hyper_par_objective_fun=None,
                                        grad_tol=1e-8)

        # computation with conjugate-gradient
        hvp = autograd.hessian_vector_product(objective_fun, argnum=0)

        opt0 = deepcopy(vb_opt_free_params)
        hyper0 = deepcopy(prior_params_paragami.flatten(prior_params_dict,
                                                        free=True))
        # my own system solver
        system_solver = preconditioner_lib.SystemSolverFromHVP(hvp, opt0, hyper0)

        print('\n conjugate-gradient computation: ')
        vb_sens2 = \
            vittles.HyperparameterSensitivityLinearApproximation(
                                            objective_fun = objective_fun,
                                            opt_par_value = vb_opt_free_params,
                                            hyper_par_value = prior_free_params,
                                            validate_optimum=False,
                                            factorize_hessian=True,
                                            hyper_par_objective_fun=None,
                                            grad_tol=1e-8,
                                            system_solver=system_solver,
                                            compute_hess=False)

        assert np.max(np.abs(vb_sens._sens_mat - vb_sens2._sens_mat)) < 1e-4

        # now try with preconditioner
        mfvb_cov, mfvb_info = preconditioner_lib.get_mfvb_cov_preconditioner(
                                vb_params_dict, vb_params_paragami,
                                use_logitnormal_sticks)

        system_solver2 = preconditioner_lib.SystemSolverFromHVP(hvp, opt0, hyper0,
                                    cg_opts = {'M': mfvb_info})
        print('\n preconditioned conjugate-gradient computation: ')
        vb_sens3 = \
            vittles.HyperparameterSensitivityLinearApproximation(
                                            objective_fun = objective_fun,
                                            opt_par_value = vb_opt_free_params,
                                            hyper_par_value = prior_free_params,
                                            validate_optimum=False,
                                            factorize_hessian=True,
                                            hyper_par_objective_fun=None,
                                            grad_tol=1e-8,
                                            system_solver=system_solver2,
                                            compute_hess=False)

        assert np.max(np.abs(vb_sens._sens_mat - vb_sens3._sens_mat)) < 1e-4
