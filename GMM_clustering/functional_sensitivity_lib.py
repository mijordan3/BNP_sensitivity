# some extra functions to do functional sensitivity
import sys
sys.path.insert(0, './../../LinearResponseVariationalBayes.py')

import autograd.numpy as np
import autograd.scipy as sp

import gmm_clustering_lib as gmm_utils

import LinearResponseVariationalBayes as vb
import LinearResponseVariationalBayes.ExponentialFamilies as ef
import LinearResponseVariationalBayes.SparseObjectives as obj_lib

import scipy as osp


class PriorPerturbation(object):
    def __init__(self, model, log_phi, epsilon=1.0):
        self.logit_v_lb = -4
        self.logit_v_ub = 4

        self.gustafson_style = False

        self.model = model

        self.epsilon_param = vb.ScalarParam('epsilon', lb=0.0)
        self.epsilon_param.set(epsilon)

        self.set_log_phi(log_phi)

        if not model.vb_params.use_logitnormal_sticks:
            raise NotImplementedError(
                'functional sensitivty only computed with logitnormal sticks')
        self.model = model

        self.objective = obj_lib.Objective(
            self.model.global_vb_params, self.get_perturbed_kl)

    #################
    # Functions that are used for optimization and sensitivity.

    def get_e_log_perturbation(self, sum_vector=True):
        if self.gustafson_style:
            perturbation_fun = \
                lambda logit_v: \
                    (1 + self.epsilon_param.get() * np.exp(self.log_phi(logit_v)))
        else:
            perturbation_fun = \
                lambda logit_v: self.log_phi(logit_v) * self.epsilon_param.get()

        e_perturbation_vec = gmm_utils.get_e_func_logit_stick_vec(
            self.model.vb_params, perturbation_fun)

        if sum_vector:
            return -1 * np.sum(e_perturbation_vec)
        else:
            return -1 * e_perturbation_vec

    def get_perturbed_kl(self):
        return self.get_e_log_perturbation() + self.model.set_z_get_kl()

    #################
    # Functions that are used for graphing and the influence function.

    # The log variational density of stick k at logit_v
    # in the logit_stick space.
    def get_log_q_logit_stick(self, logit_v, k):
        mean = self.model.global_vb_params['v_sticks']['mean'].get()[k]
        info = self.model.global_vb_params['v_sticks']['info'].get()[k]
        return -0.5 * (info * (logit_v - mean) ** 2 - np.log(info))

    # Return a vector of log variational densities for all sticks at logit_v
    # in the logit stick space.
    def get_log_q_logit_all_sticks(self, logit_v):
        mean = self.model.global_vb_params['v_sticks']['mean'].get()
        info = self.model.global_vb_params['v_sticks']['info'].get()
        return -0.5 * (info * (logit_v - mean) ** 2 - np.log(info))

    def get_log_p0(self, v):
        alpha = self.model.prior_params['alpha'].get()
        return (alpha - 1) * np.log1p(v) - self.log_norm_p0

    def get_log_p0_logit(self, logit_v):
        alpha = self.model.prior_params['alpha'].get()
        return \
            logit_v - (alpha + 1) * np.log1p(np.exp(logit_v)) - \
            self.log_norm_p0_logit

    def get_log_pc(self, v):
        logit_v = np.log(v) - np.log(1 - v)
        epsilon = self.epsilon_param.get()
        if np.abs(epsilon) < 1e-8:
            return self.get_log_p0(v)

        if self.gustafson_style:
            log_epsilon = np.log(epsilon)
            return \
                self.get_log_p0(v) + \
                self.log_phi(logit_v) + \
                log_epsilon - \
                self.log_norm_pc
        else:
            assert epsilon <= 1
            return \
                self.get_log_p0(v) + \
                epsilon * self.log_phi(logit_v) - \
                self.log_norm_pc

    def get_log_pc_logit(self, logit_v):
        epsilon = self.epsilon_param.get()
        if np.abs(epsilon) < 1e-8:
            return self.get_log_p0_logit(logit_v)

        if self.gustafson_style:
            log_epsilon = np.log(epsilon)
            return \
                self.get_log_p0_logit(logit_v) + \
                self.log_phi(logit_v) + \
                log_epsilon - \
                self.log_norm_pc_logit
        else:
            assert epsilon <= 1
            return \
                self.get_log_p0_logit(logit_v) + \
                epsilon * self.log_phi(logit_v) - \
                self.log_norm_pc_logit

    ###################################
    # Setting functions for initialization

    def set_epsilon(self, epsilon):
        self.epsilon_param.set(epsilon)
        self.set_log_phi(self.log_phi)

    def set_log_phi(self, log_phi):
        # Set attributes derived from phi and epsilon

        # Initial values for the log normalzing constants which will be set below.
        self.log_norm_p0 = 0
        self.log_norm_pc = 0
        self.log_norm_p0_logit = 0
        self.log_norm_pc_logit = 0

        self.log_phi = log_phi

        norm_p0, _ = osp.integrate.quadrature(
            lambda v: np.exp(self.get_log_p0(v)), 0, 1)
        assert norm_p0 > 0
        self.log_norm_p0 = np.log(norm_p0)

        norm_pc, _ = osp.integrate.quadrature(
            lambda v: np.exp(self.get_log_pc(v)),
            0, 1)
        assert norm_pc > 0
        self.log_norm_pc = np.log(norm_pc)

        norm_p0_logit, _ = osp.integrate.quadrature(
            lambda logit_v: np.exp(self.get_log_p0_logit(logit_v)),
            self.logit_v_lb, self.logit_v_ub)
        assert norm_p0_logit > 0
        self.log_norm_p0_logit = np.log(norm_p0_logit)

        norm_pc_logit, _ = osp.integrate.quadrature(
            lambda logit_v: np.exp(self.get_log_pc_logit(logit_v)),
            self.logit_v_lb, self.logit_v_ub)
        assert norm_pc_logit > 0
        self.log_norm_pc_logit = np.log(norm_pc_logit)


# A class to calculate sensitivity to the stick-breaking distribution.
#
# Parameters:
#   model: A DPGaussianMixture object.
#   best_param: A free parameter at a local minimum of the KL divergence.
#   kl_hessian: The Hessian of the KL divergence evaluated at best_param.
#   dgdeta: The derivative of the moments of interest with respect to the
#           variational free parameters.
# class StickSensitivity(object):
#     def __init__(self, model, best_param, kl_hessian, dgdeta):
#         self.model = model
#         self.best_param = best_param
#         self.log_q_logit_stick_obj = obj_lib.Objective(
#             self.model.global_vb_params, self.get_log_q_logit_stick)
#         self.set_lr_matrix(kl_hessian, dgdeta)
#
#     def set_lr_matrix(self, kl_hessian, dgdeta):
#         self.lr_mat = -1 * np.linalg.solve(kl_hessian, dgdeta)
#
#     # The log variational density of stick k at logit_v
#     # in the logit_stick space.
#     def get_log_q_logit_stick(self, logit_v, k):
#         mean = self.model.global_vb_params['v_sticks']['mean'].get()[k]
#         info = self.model.global_vb_params['v_sticks']['info'].get()[k]
#         return -0.5 * (info * (logit_v - mean) ** 2 - np.log(info))
#
#     # Return a vectof of log variational densities for all sticks at logit_v
#     # in the logit stick space.
#     def get_log_q_logit_all_sticks(self, logit_v):
#         mean = self.model.global_vb_params['v_sticks']['mean'].get()
#         info = self.model.global_vb_params['v_sticks']['info'].get()
#         return -0.5 * (info * (logit_v - mean) ** 2 - np.log(info))
#
#     # The base prior (of any stick -- they are all the same) at logit_v in
#     # the logit stick space.
#     def get_log_p0_logit_stick(self, logit_v):
#         alpha = self.model.prior_params['alpha'].get()
#         return logit_v - (alpha + 1) * np.log1p(np.exp(logit_v))
#
#     # The base prior (of any stick -- they are all the same) at v in
#     # the stick space.
#     def get_log_p0_stick(self, v):
#         alpha = self.model.prior_params['alpha'].get()
#         return (alpha - 1) * np.log1p(v)
#
#     # Get the influence function for a perturbation to the prior on stick k.
#     def get_single_stick_influence(self, logit_v, k):
#         log_q = self.get_log_q_logit_stick(logit_v, k)
#         log_p0 = self.get_log_p0_logit_stick(logit_v)
#
#         log_q_grad = self.log_q_logit_stick_obj.fun_free_grad(
#             self.best_param, logit_v=logit_v, k=k)
#         return(self.lr_mat.T @ log_q_grad * np.exp(log_q - log_p0))
#
#     # Get the influence function for a perturbation to all stick priors
#     # simultaneously.
#     def get_all_stick_influence(self, logit_v):
#         log_q_vec = self.get_log_q_logit_all_sticks(logit_v)
#
#         # The prior is the same for every stick.
#         log_p0 = self.get_log_p0_logit_stick(logit_v)
#
#         log_q_grad_vec = np.array([
#             self.log_q_logit_stick_obj.fun_free_grad(
#                 self.best_param, logit_v=logit_v, k=k)
#             for k in range(self.model.k_approx - 1) ])
#
#         dens_ratios = np.exp(log_q_vec - log_p0)
#         return(self.lr_mat.T @ np.einsum('kd,k->d', log_q_grad_vec, dens_ratios))
