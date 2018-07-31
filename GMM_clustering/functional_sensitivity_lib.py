# some extra functions to do functional sensitivity
import sys
sys.path.insert(0, './../../LinearResponseVariationalBayes.py')

import autograd.numpy as np
import autograd.scipy as sp
import LinearResponseVariationalBayes.ExponentialFamilies as ef

# A class to calculate sensitivity to the stick-breaking distribution.
#
# Parameters:
#   model: A DPGaussianMixture object.
#   best_param: A free parameter at a local minimum of the KL divergence.
#   kl_hessian: The Hessian of the KL divergence evaluated at best_param.
#   dgdeta: The derivative of the moments of interest with respect to the
#           variational free parameters.
class StickSensitivity(object):
    def __init__(self, model, best_param, kl_hessian, dgdeta):
        self.model = model
        self.best_param = best_param
        self.log_q_logit_stick_obj = obj_lib.Objective(
            self.model.global_vb_params, self.get_log_q_logit_stick)
        self.set_lr_matrix(kl_hessian, dgdeta)

    def set_lr_matrix(self, kl_hessian, dgdeta):
        self.lr_mat = -1 * np.linalg.solve(kl_hessian, dgdeta)

    # The log variational density of stick k at logit_v
    # in the logit_stick space.
    def get_log_q_logit_stick(self, logit_v, k):
        mean = self.model.global_vb_params['v_sticks']['mean'].get()[k]
        info = self.model.global_vb_params['v_sticks']['info'].get()[k]
        return -0.5 * (info * (logit_v - mean) ** 2 - np.log(info))

    # Return a vectof of log variational densities for all sticks at logit_v
    # in the logit stick space.
    def get_log_q_logit_all_sticks(self, logit_v):
        mean = self.model.global_vb_params['v_sticks']['mean'].get()
        info = self.model.global_vb_params['v_sticks']['info'].get()
        return -0.5 * (info * (logit_v - mean) ** 2 - np.log(info))

    # The base prior (of any stick -- they are all the same) at logit_v in
    # the logit stick space.
    def get_log_p0_logit_stick(self, logit_v):
        alpha = self.model.prior_params['alpha'].get()
        return logit_v - (alpha + 1) * np.log1p(np.exp(logit_v))

    # Get the influence function for a perturbation to the prior on stick k.
    def get_single_stick_influence(self, logit_v, k):
        log_q = self.get_log_q_logit_stick(logit_v, k)
        log_p0 = self.get_log_p0_logit_stick(logit_v)

        log_q_grad = self.log_q_logit_stick_obj.fun_free_grad(
            self.best_param, logit_v=logit_v, k=k)
        return(self.lr_mat.T @ log_q_grad * np.exp(log_q - log_p0))

    # Get the influence function for a perturbation to all stick priors
    # simultaneously.
    def get_all_stick_influence(self, logit_v):
        log_q_vec = self.get_log_q_logit_all_sticks(logit_v)

        # The prior is the same for every stick.
        log_p0 = self.get_log_p0_logit_stick(logit_v)

        log_q_grad_vec = np.array([
            self.log_q_logit_stick_obj.fun_free_grad(
                self.best_param, logit_v=logit_v, k=k)
            for k in range(self.model.k_approx - 1) ])

        dens_ratios = np.exp(log_q_vec - log_p0)
        return(self.lr_mat.T @ np.einsum('kd,k->d', log_q_grad_vec, dens_ratios))
