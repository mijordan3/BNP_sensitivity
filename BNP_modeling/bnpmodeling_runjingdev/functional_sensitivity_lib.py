import jax.numpy as np
import jax.scipy as sp

import bnpmodeling_runjingdev.modeling_lib as modeling_lib

import scipy as osp

from copy import deepcopy


########################
# Functions to evaluate expectations
# given a functional perturbation
########################
class PriorPerturbation(object):
    def __init__(self, alpha0,
                        log_phi,
                        epsilon=1.0,
                        logit_v_ub = 4,
                        logit_v_lb = -4,
                        quad_maxiter = 50):

        self.logit_v_lb = logit_v_lb
        self.logit_v_ub = logit_v_ub

        self.quad_maxiter = quad_maxiter

        self.gustafson_style = False

        self.alpha0 = alpha0

        self.epsilon_param = epsilon

        self.set_log_phi(log_phi)

    #################
    # Functions that are used for graphing and the influence function.

    def get_log_p0(self, v):
        alpha = self.alpha0
        return (alpha - 1) * np.log1p(-v) - self.log_norm_p0

    def get_log_p0_logit(self, logit_v):
        alpha = self.alpha0
        return \
            - alpha * logit_v - (alpha + 1) * np.log1p(np.exp(-logit_v)) - \
            self.log_norm_p0_logit

    def get_log_pc(self, v):
        logit_v = np.log(v) - np.log(1 - v)
        epsilon = self.epsilon_param
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
            # assert epsilon <= 1
            return \
                self.get_log_p0(v) + \
                epsilon * self.log_phi(logit_v) - \
                self.log_norm_pc

    def get_log_pc_logit(self, logit_v):
        epsilon = self.epsilon_param
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
            # assert epsilon <= 1
            return \
                self.get_log_p0_logit(logit_v) + \
                epsilon * self.log_phi(logit_v) - \
                self.log_norm_pc_logit

    ###################################
    # Setting functions for initialization

    def set_epsilon(self, epsilon):
        self.epsilon_param = epsilon
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
            lambda v: np.exp(self.get_log_p0(v)), 0, 1, maxiter = self.quad_maxiter)
        assert norm_p0 > 0
        self.log_norm_p0 = np.log(norm_p0)

        norm_pc, _ = osp.integrate.quadrature(
            lambda v: np.exp(self.get_log_pc(v)),
            0, 1, maxiter = self.quad_maxiter)
        assert norm_pc > 0
        self.log_norm_pc = np.log(norm_pc)

        norm_p0_logit, _ = osp.integrate.quadrature(
            lambda logit_v: np.exp(self.get_log_p0_logit(logit_v)),
            self.logit_v_lb, self.logit_v_ub, maxiter = self.quad_maxiter)
        assert norm_p0_logit > 0
        self.log_norm_p0_logit = np.log(norm_p0_logit)

        norm_pc_logit, _ = osp.integrate.quadrature(
            lambda logit_v: np.exp(self.get_log_pc_logit(logit_v)),
            self.logit_v_lb, self.logit_v_ub, maxiter = self.quad_maxiter)
        assert norm_pc_logit > 0
        self.log_norm_pc_logit = np.log(norm_pc_logit)


def get_e_log_perturbation(log_phi, stick_propn_mean, stick_propn_info,
                           gh_loc, gh_weights, sum_vector=True):

    """
    Computes the expected log multiplicative perturbation

    Parameters
    ----------
    log_phi : Callable function
        The log of the multiplicative perturbation in logit space
    vb_params_dict : dictionary
        A dictionary that contains the variational parameters
    epsilon : float
        The 'epsilon' specififying the multiplicative perturbation
    gh_loc : vector
        Locations for gauss-hermite quadrature. We need this compute the
        expected prior terms.
    gh_weights : vector
        Weights for gauss-hermite quadrature. We need this compute the
        expected prior terms.
    sum_vector : boolean
        whether to sum the expectation over the k sticks

    Returns
    -------
    float
        The expected log perturbation under the variational distribution

    """
    
    e_perturbation_vec = modeling_lib.get_e_func_logit_stick_vec(
                                        stick_propn_mean, stick_propn_info,
                                        gh_loc, gh_weights, log_phi)

    if sum_vector:
        return np.sum(e_perturbation_vec)
    else:
        return e_perturbation_vec


class FunctionalPerturbationObjective(): 
    def __init__(self, 
                 log_phi, 
                 vb_params_paragami, 
                 gh_log, gh_weights,
                 e_log_phi = None, 
                 stick_key = 'stick_params'): 

        # log_phi (or e_log_phi) returns the additve
        # perturbation to the **ELBO** 
        
        # log_phi takes input as logit-stick and returns the 
        # perturbation. 
        # e_log_phi (optional) takes input means and infos 
        # and returns the expectation of log-phi under the 
        # logit-normal variational distribution
        # if e_log_phi is not provided, we compute the expectation 
        # using gauss-hermite quadrature
        
        self.vb_params_paragami = vb_params_paragami
        self.stick_key = stick_key 
        
        self.gh_loc = gh_loc
        self.gh_weights = gh_weights
        
        self.log_phi = log_phi
        
        if self.e_log_phi is None: 
            # set the expected log-perturbation 
            # using gauss-hermite quadrature
            self._set_e_log_phi_with_gh()
        else: 
            # a pre-computed expectation. 
            # this takes 
            self.e_log_phi = e_log_phi
        
    def _set_e_log_phi_with_gh(self): 
        
        self.e_log_phi = lambda means, infos : 
                            get_e_log_perturbation(self.log_phi, 
                                                   means, 
                                                   infos,  
                                                   gh_loc,
                                                   gh_weights, 
                                                   sum_vector=True)

    def e_log_phi_epsilon(self, means, infos, epsilon): 
        # with epsilon fixed this is the input to the optimizer
        # (this is added to the ELBO)
        
        return epsilon * self.e_log_phi(means, infos)
    
    def hyper_par_objective_fun(self,
                                vb_params_free, 
                                epsilon): 
        # NOTE THE NEGATIVE SIGN
        # this is passed into the HyperparameterSensitivity class
        # and is added to the **KL** 

        vb_params_dict = self.vb_params_paragami.fold(vb_params_free, 
                                                free = True)
    
        # get means and infos 
        means = vb_params_dict[self.stick_key]['stick_means']
        infos = vb_params_dict[self.stick_key]['stick_infos']
        
        return - self.e_log_phi_epsilon(means, infos, epsilon)

    