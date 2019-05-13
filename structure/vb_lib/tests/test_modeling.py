import autograd

import autograd.numpy as np
import autograd.scipy as sp
from numpy.polynomial.hermite import hermgauss

import scipy as osp

from vb_lib import structure_model_lib, preconditioner_lib, data_utils
import vb_lib.structure_optimization_lib as str_opt_lib

import paragami
import vittles

from copy import deepcopy

import argparse
import distutils.util

import os

from BNP_modeling import cluster_quantities_lib, modeling_lib
import BNP_modeling.optimization_lib as opt_lib

import unittest

import numpy.testing as np_test

np.random.seed(25465)

# Test my system solver, and make sure it interacts with
# HyperparameterSensitivityLinearApproximation in the way that I expect
class TestModeling(unittest.TestCase):
    def test_modeling(self):
        # draw data
        n_obs = 10
        n_loci = 5

        # this is just done randomly at the moment
        # a matrix of integers {0, 1, 2}
        g_obs_int = np.random.choice(3, size = (n_obs, n_loci))

        # one hot encoding
        g_obs = data_utils.get_one_hot(g_obs_int, 3)
        assert g_obs.shape == (n_obs, n_loci, 3)

        # prior parameters
        prior_params_dict, prior_params_paragami = \
            structure_model_lib.get_default_prior_params()

        # vb params
        use_logitnormal_sticks = True
        k_approx = 12
        gh_deg = 8
        gh_loc, gh_weights = hermgauss(gh_deg)

        vb_params_dict, vb_params_paragami = \
            structure_model_lib.get_vb_params_paragami_object(n_obs, n_loci,
                        k_approx, use_logitnormal_sticks)

        ###### check e_log_logitnormal #######
        dp_prior_alpha = prior_params_dict['dp_prior_alpha']
        ind_mix_stick_propn_mean = vb_params_dict['ind_mix_stick_propn_mean']
        ind_mix_stick_propn_info = vb_params_dict['ind_mix_stick_propn_info']

        # get computed moments
        e_log_v, e_log_1mv = \
        structure_model_lib.ef.get_e_log_logitnormal(
                                    lognorm_means = ind_mix_stick_propn_mean,
                                    lognorm_infos = ind_mix_stick_propn_info,
                                    gh_loc = gh_loc,
                                    gh_weights = gh_weights)

        # sample from logit-normal
        num_draws = 10**5
        samples = np.random.normal(ind_mix_stick_propn_mean,
                        1/np.sqrt(ind_mix_stick_propn_info),
                        size = (num_draws, n_obs, k_approx - 1))
        logit_norm_samples = sp.special.expit(samples)

        e_log_samples = np.log(logit_norm_samples)
        e_log_1m_samples = np.log(1 - logit_norm_samples)

        # check difference
        diff1 = np.abs(e_log_v - e_log_samples.mean(axis = 0))
        diff2 = np.abs(e_log_1mv - e_log_1m_samples.mean(axis = 0))

        assert np.all(diff1 < 3 * e_log_samples.std(axis = 0))
        assert np.all(diff2 < 3 * e_log_1m_samples.std(axis = 0))

        ###### check dp prior ######
        e_dp_prior = \
            modeling_lib.get_e_logitnorm_dp_prior(ind_mix_stick_propn_mean,
                                            ind_mix_stick_propn_info,
                                            dp_prior_alpha, gh_loc, gh_weights)

        # sample
        # sample from logit-normal
        num_draws = 10**5
        samples = np.random.normal(ind_mix_stick_propn_mean,
                        1/np.sqrt(ind_mix_stick_propn_info),
                        size = (num_draws, n_obs, k_approx - 1))
        logit_norm_samples = sp.special.expit(samples)
        # samples of the dp_prior
        dp_prior_samples = (np.log(1 - logit_norm_samples) * \
                            (dp_prior_alpha - 1)).sum(axis = 2).sum(axis = 1)

        dp_prior_samples_mean = np.mean(dp_prior_samples)
        dp_prior_samples_std = np.std(dp_prior_samples)

        np_test.assert_allclose(
            dp_prior_samples_mean,
            e_dp_prior,
            atol = 3 * dp_prior_samples_std / np.sqrt(num_draws))

        #### check beta entropy ######
        pop_freq_beta_params = vb_params_dict['pop_freq_beta_params']
        lk = pop_freq_beta_params.shape[0] * pop_freq_beta_params.shape[1]
        beta_entropy = structure_model_lib.ef.beta_entropy( \
                            tau = pop_freq_beta_params.reshape((lk, 2)))

        beta_samples = np.random.beta(a = pop_freq_beta_params[:, :, 0],
                                    b = pop_freq_beta_params[:, :, 1],
                                    size = (num_draws, n_loci, k_approx))

        beta_entropy_samples = -sp.stats.beta.logpdf(beta_samples,
                          pop_freq_beta_params[:, :, 0],
                          pop_freq_beta_params[:, :, 1]).sum(axis = 2).sum(axis = 1)

        np_test.assert_allclose(
            beta_entropy_samples.mean(),
            beta_entropy,
            atol = 3 * beta_entropy_samples.std() / np.sqrt(num_draws))

        ###### check e_log_beta ######
        e_log_beta, e_log_1mbeta = \
            modeling_lib.get_e_log_beta(tau = pop_freq_beta_params)

        e_log_beta_samples = np.log(beta_samples)
        e_log_1mbeta_samples = np.log(1 - beta_samples)

        diff1 = np.abs(e_log_beta - e_log_beta_samples.mean(axis = 0))
        diff2 = np.abs(e_log_1mbeta - e_log_1mbeta_samples.mean(axis = 0))

        assert np.all(diff1 < 3 * e_log_beta_samples.std(axis = 0))
        assert np.all(diff2 < 3 * e_log_1mbeta_samples.std(axis = 0))
