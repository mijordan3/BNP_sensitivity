import jax

import jax.numpy as np
import jax.scipy as sp

from numpy.polynomial.hermite import hermgauss
import scipy as osp

from vb_lib import structure_model_lib, data_utils, cavi_lib
from vb_lib.preconditioner_lib import get_mfvb_cov_matmul
from bnpmodeling_runjingdev.sensitivity_lib import HyperparameterSensitivityLinearApproximation, get_jac_hvp_fun

import paragami

from copy import deepcopy

import time

import matplotlib.pyplot as plt

from bnpmodeling_runjingdev import cluster_quantities_lib, modeling_lib


import numpy as onp
onp.random.seed(53453)

################
# Draw data
################
n_obs = 100
n_loci = 2000
n_pop = 4

g_obs, true_pop_allele_freq, true_ind_admix_propn = \
    data_utils.draw_data(n_obs, n_loci, n_pop)

print(g_obs.shape)

################
# get prior
################
prior_params_dict, prior_params_paragami = \
    structure_model_lib.get_default_prior_params()

dp_prior_alpha = prior_params_dict['dp_prior_alpha']
allele_prior_alpha = prior_params_dict['allele_prior_alpha']
allele_prior_beta = prior_params_dict['allele_prior_beta']

################
# get vb params
################
k_approx = 8

gh_deg = 8
gh_loc, gh_weights = hermgauss(gh_deg)

use_logitnormal_sticks = False

vb_params_dict, vb_params_paragami = \
    structure_model_lib.get_vb_params_paragami_object(n_obs, n_loci, k_approx,
                                    use_logitnormal_sticks = use_logitnormal_sticks)

################
# initialize
################
vb_params_dict = \
        structure_model_lib.set_init_vb_params(g_obs, k_approx, vb_params_dict,
                                                seed = 34221)


get_kl_jitted = jax.jit(lambda g_obs, vb_params_dict, prior_params_dict:
                            structure_model_lib.get_kl(g_obs,
                                                        vb_params_dict,
                                                        prior_params_dict,
                                                        gh_loc = None, gh_weights = None,
                                                        log_phi = None,
                                                        epsilon = 1.,
                                                        detach_ez = False))
t0 = time.time()
_ = get_kl_jitted(g_obs,
                vb_params_dict,
                prior_params_dict)

print('kl compile time: ', time.time() - t0)

for i in range(50):
    t0 = time.time()

    _ = get_kl_jitted(g_obs,
                    vb_params_dict,
                    prior_params_dict).block_until_ready()

    print('kl time: ', time.time() - t0)

print('pause. ')
time.sleep(5)
print('resume. ')
for i in range(50):
    t0 = time.time()

    _ = get_kl_jitted(g_obs,
                    vb_params_dict,
                    prior_params_dict)

    print('kl time2: ', time.time() - t0)

################
# Benchmark CAVI
################
# _ = cavi_lib.run_cavi(g_obs, vb_params_dict,
#                         vb_params_paragami,
#                         prior_params_dict,
#                         gh_loc = gh_loc, gh_weights = gh_weights,
#                         max_iter = 100,
#                         x_tol = 1e-3,
#                         print_every = 1)

# e_log_sticks, e_log_1m_sticks, \
#     e_log_pop_freq, e_log_1m_pop_freq = \
#         structure_model_lib.get_moments_from_vb_params_dict(
#             vb_params_dict, gh_loc, gh_weights)
#
# from vb_lib.cavi_lib import joint_loglik
#
# joint_loglik = jax.jit(joint_loglik)
#
# t0 = time.time()
# _ = joint_loglik(g_obs,
#                     e_log_pop_freq, e_log_1m_pop_freq,
#                     e_log_sticks, e_log_1m_sticks,
#                     dp_prior_alpha, allele_prior_alpha,
#                     allele_prior_beta)
#
# print('joint loglik compile time: ', time.time() - t0)
#
# t0 = time.time()
# for i in range(100):
#     _ = joint_loglik(g_obs,
#                         e_log_pop_freq, e_log_1m_pop_freq,
#                         e_log_sticks, e_log_1m_sticks,
#                         dp_prior_alpha, allele_prior_alpha,
#                         allele_prior_beta)
# print('joint loglik 100 iter time: ', time.time() - t0)

########
# t0 = time.time()
# _ = update_stick_beta(g_obs,
#                     e_log_pop_freq, e_log_1m_pop_freq,
#                     e_log_sticks, e_log_1m_sticks,
#                     dp_prior_alpha, allele_prior_alpha,
#                     allele_prior_beta)
#
# print('stick beta compile time: ', time.time() - t0)
#
# t0 = time.time()
# for i in range(100):
#     _ = update_stick_beta(g_obs,
#                         e_log_pop_freq, e_log_1m_pop_freq,
#                         e_log_sticks, e_log_1m_sticks,
#                         dp_prior_alpha, allele_prior_alpha,
#                         allele_prior_beta)
# print('stick beta 100 iter time: ', time.time() - t0)
#
# ########
# _ = get_pop_beta_update1(g_obs,
#                     e_log_pop_freq, e_log_1m_pop_freq,
#                     e_log_sticks, e_log_1m_sticks,
#                     dp_prior_alpha, allele_prior_alpha,
#                     allele_prior_beta)
# t0 = time.time()
# for i in range(100):
#     _ = get_pop_beta_update1(g_obs,
#                         e_log_pop_freq, e_log_1m_pop_freq,
#                         e_log_sticks, e_log_1m_sticks,
#                         dp_prior_alpha, allele_prior_alpha,
#                         allele_prior_beta)
# print('get_pop_beta_update1 iter: ', time.time() - t0)
#
# ###########
# _ = get_pop_beta_update2(g_obs,
#                     e_log_pop_freq, e_log_1m_pop_freq,
#                     e_log_sticks, e_log_1m_sticks,
#                     dp_prior_alpha, allele_prior_alpha,
#                     allele_prior_beta)
# t0 = time.time()
# for i in range(100):
#     _ = get_pop_beta_update2(g_obs,
#                         e_log_pop_freq, e_log_1m_pop_freq,
#                         e_log_sticks, e_log_1m_sticks,
#                         dp_prior_alpha, allele_prior_alpha,
#                         allele_prior_beta)
# print('get_pop_beta_update2 iter: ', time.time() - t0)
