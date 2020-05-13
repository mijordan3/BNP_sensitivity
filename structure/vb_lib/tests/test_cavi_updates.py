import autograd

import autograd.numpy as np
import autograd.scipy as sp
from numpy.polynomial.hermite import hermgauss

from vb_lib import structure_model_lib, data_utils, cavi_lib

from bnpmodeling_runjingdev import modeling_lib

import paragami

import unittest

import numpy.testing as testing

from copy import deepcopy

np.random.seed(25465)


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
_, vb_params_paragami = \
    structure_model_lib.get_vb_params_paragami_object(n_obs, n_loci,
        k_approx, use_logitnormal_sticks)


class TestCaviUpdate(unittest.TestCase):
    def test_ez_update(self):
        vb_params_dict = vb_params_paragami.random()

        # get moments
        e_log_sticks, e_log_1m_sticks, \
            e_log_pop_freq, e_log_1m_pop_freq = \
                structure_model_lib.get_moments_from_vb_params_dict(
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

    def test_admixutre_stick_update(self):
        vb_params_dict = vb_params_paragami.random()

        e_log_sticks, e_log_1m_sticks, \
            e_log_pop_freq, e_log_1m_pop_freq = \
                structure_model_lib.get_moments_from_vb_params_dict( \
                                        vb_params_dict, use_logitnormal_sticks)

        # e_z is fixed
        e_z = cavi_lib.update_z(g_obs, e_log_sticks, e_log_1m_sticks, e_log_pop_freq,
                                e_log_1m_pop_freq)


        # define loss
        get_free_vb_params_loss = paragami.FlattenFunctionInput(
                                    original_fun=structure_model_lib.get_kl,
                                    patterns = vb_params_paragami,
                                    free = True,
                                    argnums = 1)

        get_loss = \
            lambda x : get_free_vb_params_loss(g_obs,
                                        x, prior_params_dict,
                                        use_logitnormal_sticks,
                                        gh_loc, gh_weights, e_z = e_z)

        # define gradient
        get_loss_grad = autograd.grad(get_loss)


        # update individual admixture sticks
        _, _, vb_params_dict['ind_mix_stick_beta_params'] = \
            e_log_sticks, e_log_1m_sticks, stick_beta_params = \
                cavi_lib.update_stick_beta(g_obs, e_z,
                                    e_log_pop_freq, e_log_1m_pop_freq,
                                    dp_prior_alpha, allele_prior_alpha,
                                    allele_prior_beta)

        loss_grad = get_loss_grad(vb_params_paragami.flatten(vb_params_dict, free = True))

        # get grad wrt individual stick parameters
        grad = loss_grad[get_param_indices('ind_mix_stick_beta_params',
                                                 vb_params_dict,
                                                 vb_params_paragami)]

        assert np.abs(grad).max() < 1e-12

    def test_population_stick_update(self):
        vb_params_dict = vb_params_paragami.random()

        e_log_sticks, e_log_1m_sticks, \
            e_log_pop_freq, e_log_1m_pop_freq = \
                structure_model_lib.get_moments_from_vb_params_dict( \
                                        vb_params_dict, use_logitnormal_sticks)

        # e_z is fixed
        e_z = cavi_lib.update_z(g_obs, e_log_sticks, e_log_1m_sticks, e_log_pop_freq,
                                e_log_1m_pop_freq)


        # define loss
        get_free_vb_params_loss = paragami.FlattenFunctionInput(
                                    original_fun=structure_model_lib.get_kl,
                                    patterns = vb_params_paragami,
                                    free = True,
                                    argnums = 1)

        get_loss = \
            lambda x : get_free_vb_params_loss(g_obs,
                                        x, prior_params_dict,
                                        use_logitnormal_sticks,
                                        gh_loc, gh_weights, e_z = e_z)

        # define gradient
        get_loss_grad = autograd.grad(get_loss)


        # update population frequency sticks
        _, _, vb_params_dict['pop_freq_beta_params'] = \
            e_log_sticks, e_log_1m_sticks, stick_beta_params = \
                cavi_lib.update_pop_beta(g_obs, e_z,
                                    e_log_sticks, e_log_1m_sticks,
                                    dp_prior_alpha, allele_prior_alpha,
                                    allele_prior_beta)

        loss_grad = get_loss_grad(vb_params_paragami.flatten(vb_params_dict, free = True))
        # get grad wrt population frequency parameters
        grad = loss_grad[get_param_indices('pop_freq_beta_params',
                                                 vb_params_dict,
                                                 vb_params_paragami)]

        assert np.abs(grad).max() < 1e-12

    def test_cavi(self):
        # run cavi in full, with the debugger on

        # get vb_params
        vb_params_dict = vb_params_paragami.random()

        ez_opt, vb_opt_dict, kl_vec, _ = \
            cavi_lib.run_cavi(g_obs, vb_params_dict,
                                vb_params_paragami,
                                prior_params_dict,
                                use_logitnormal_sticks = False,
                                max_iter = 10,
                                x_tol = 1e-4,
                                debug = True)

    def test_svi(self):
        # run svi in full, with debugger on

        # get vb_params
        vb_params_dict = vb_params_paragami.random()

        # get initial z
        e_log_sticks, e_log_1m_sticks, \
            e_log_pop_freq, e_log_1m_pop_freq = \
                structure_model_lib.get_moments_from_vb_params_dict( \
                                        vb_params_dict, use_logitnormal_sticks)

        e_z_init = cavi_lib.update_z(g_obs, e_log_sticks, e_log_1m_sticks, e_log_pop_freq,
                                        e_log_1m_pop_freq)

        _ = cavi_lib.run_svi(g_obs, vb_params_dict,
                                prior_params_dict,
                                e_z_init,
                                use_logitnormal_sticks = False,
                                batchsize = 2,
                                x_tol = 1e-2,
                                max_iter = 20,
                                print_every = 1,
                                debug_local_updates = True)

    def test_logitnormal_stick_updates(self):
        # get vb_params
        _, vb_params_paragami = \
            structure_model_lib.get_vb_params_paragami_object(n_obs, n_loci,
                k_approx, use_logitnormal_sticks = True)

        vb_params_dict = vb_params_paragami.random()

        # get e_z
        e_z = cavi_lib.get_ez_from_vb_params_dict(g_obs, vb_params_dict,
                            use_logitnormal_sticks = True,
                            gh_loc = gh_loc, gh_weights = gh_weights)

        # get free parameters
        stick_mean_free_params = vb_params_paragami['ind_mix_stick_propn_mean'].\
                            flatten(vb_params_dict['ind_mix_stick_propn_mean'],
                                    free = True)
        stick_info_free_params = vb_params_paragami['ind_mix_stick_propn_info'].\
                            flatten(vb_params_dict['ind_mix_stick_propn_info'],
                                    free = True)

        # test gradients of pseudo-loss
        get_grad_stick_mean = autograd.grad(cavi_lib._get_logitnormal_sticks_psloss, argnum = 2)
        get_grad_stick_info = autograd.grad(cavi_lib._get_logitnormal_sticks_psloss, argnum = 3)

        stick_mean_grad = get_grad_stick_mean(g_obs, e_z, stick_mean_free_params,
                                        stick_info_free_params,
                                        vb_params_paragami,
                                        prior_params_dict,
                                        gh_loc, gh_weights)

        stick_info_grad = get_grad_stick_info(g_obs, e_z, stick_mean_free_params,
                                                stick_info_free_params,
                                                vb_params_paragami,
                                                prior_params_dict,
                                                gh_loc, gh_weights)


        # check against gradients of actual loss
        get_grad_stick_mean2 = autograd.grad(cavi_lib._get_logitnormal_sticks_loss, argnum = 2)
        get_grad_stick_info2 = autograd.grad(cavi_lib._get_logitnormal_sticks_loss, argnum = 3)

        stick_mean_grad2 = get_grad_stick_mean2(g_obs,
                    e_z,
                    stick_mean_free_params,
                    stick_info_free_params,
                    vb_params_dict,
                    vb_params_paragami,
                    prior_params_dict,
                    gh_loc, gh_weights)

        stick_info_grad2 = get_grad_stick_info2(g_obs,
                            e_z,
                            stick_mean_free_params,
                            stick_info_free_params,
                            vb_params_dict,
                            vb_params_paragami,
                            prior_params_dict,
                            gh_loc, gh_weights)

        assert np.all(stick_mean_grad2 == stick_mean_grad)
        assert np.all(stick_info_grad2 == stick_info_grad)

    def test_cavi_with_logitnormal_sticks(self):
        # get vb_params
        _, vb_params_paragami = \
            structure_model_lib.get_vb_params_paragami_object(n_obs, n_loci,
                k_approx, use_logitnormal_sticks = True)

        vb_params_dict = vb_params_paragami.random()

        _ = cavi_lib.run_cavi(g_obs, vb_params_dict,
                                vb_params_paragami,
                                prior_params_dict,
                                use_logitnormal_sticks = True,
                                debug=True,
                                gh_loc = gh_loc,
                                gh_weights = gh_weights)


def get_param_indices(param_str, vb_params_dict, vb_params_paragami):
    bool_dict = deepcopy(vb_params_dict)
    for k in vb_params_dict.keys():
        if k == param_str:
            bool_dict[k] = (vb_params_dict[k] == vb_params_dict[k])
        else:
            bool_dict[k] = (vb_params_dict[k] != vb_params_dict[k])

    return vb_params_paragami.flat_indices(bool_dict, free = True)
