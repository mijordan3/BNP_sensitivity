import autograd

import autograd.numpy as np
import autograd.scipy as sp
from numpy.polynomial.hermite import hermgauss

from vb_lib import structure_model_lib, data_utils

from BNP_modeling import modeling_lib

import unittest

import numpy.testing as testing

np.random.seed(25465)

class TestEzUpdate(unittest.TestCase):
    def test_ez_update(self):
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
        use_logitnormal_sticks = True
        _, vb_params_paragami = \
            structure_model_lib.get_vb_params_paragami_object(n_obs, n_loci,
                k_approx, use_logitnormal_sticks)
        vb_params_dict = vb_params_paragami.random()

        # get moments
        e_log_sticks, e_log_1m_sticks, \
            e_log_pop_freq, e_log_1m_pop_freq = \
                structure_model_lib.get_moments_from_vb_params_dict(g_obs,
                                                    vb_params_dict,
                                                    use_logitnormal_sticks,
                                                    gh_loc,
                                                    gh_weights)

        # function that returns KL as a function of e_z natural parameters
        def get_kl_from_z_nat_param(g_obs, vb_params_dict, prior_params_dict,
                                use_logitnormal_sticks,
                                gh_loc, gh_weights,
                                z_nat_param):

            e_z = structure_model_lib.get_z_opt_from_loglik_cond_z(z_nat_param)

            return structure_model_lib.get_kl(g_obs, vb_params_dict,
                            prior_params_dict,
                            use_logitnormal_sticks,
                            gh_loc, gh_weights, e_z = e_z)

        # if e_z is optimal, this gradient should be zero
        kl_z_nat_param_grad = autograd.grad(get_kl_from_z_nat_param, argnum = 6)

        # get proposed optimal z
        e_log_cluster_probs = \
            modeling_lib.get_e_log_cluster_probabilities_from_e_log_stick(
                                e_log_sticks, e_log_1m_sticks)

        z_opt_nat_param = structure_model_lib.get_loglik_cond_z(g_obs,
                                                    e_log_pop_freq,
                                                    e_log_1m_pop_freq,
                                                    e_log_cluster_probs)

        # compute gradient
        grad = kl_z_nat_param_grad(g_obs, vb_params_dict, prior_params_dict,
                                use_logitnormal_sticks,
                                gh_loc, gh_weights,
                                z_opt_nat_param)

        assert np.max(np.abs(grad)) < 1e-8
