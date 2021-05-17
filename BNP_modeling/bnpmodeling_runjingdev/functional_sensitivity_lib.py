import jax.numpy as np
import jax.scipy as sp

import bnpmodeling_runjingdev.modeling_lib as modeling_lib

from bnpmodeling_runjingdev.stick_integration_lib import get_e_fun_normal

import scipy as osp

import matplotlib.pyplot as plt

from copy import deepcopy


########################
# Functions to evaluate densities
# given a functional perturbation
########################
class PriorPerturbation(object):
    def __init__(self,
                 alpha0,
                 log_phi,
                 epsilon=1.0,
                 logit_v_ub = 4,
                 logit_v_lb = -4,
                 quad_maxiter = 50):
        """
        Class used to compute densities of perturbed priors. 
        Useful for plotting. 

        Parameters
        ----------
        alpha0 : float
            The GEM parameter of the initial prior. 
        log_phi : Callable function
            The log of the multiplicative perturbation in logit space. 
        epsilon : float 
            Factor multiplying log_phi in the perturbation. 
        logit_v_ub : float 
            Upper bound for the integral in computing normalizatoin constants. 
        logit_v_lb : float 
            Upper bound for the integral in computing normalizatoin constants. 
        quad_maxiter : integer 
            maxiters for scipy.integrate.quadrature 
        """

        self.logit_v_lb = logit_v_lb
        self.logit_v_ub = logit_v_ub

        self.quad_maxiter = quad_maxiter

        self.gustafson_style = False

        self.alpha0 = alpha0

        self.epsilon_param = epsilon

        self.set_log_phi(log_phi)

    #############################
    # methods returning the densities 
    #############################

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
    # Setting methods for initialization
    #############################
    
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
    
    #############################
    # methods for plotting
    #############################
    def _plot_log_phi(self, ax): 
        
        # x-axis
        logit_v_grid = np.linspace(self.logit_v_lb, 
                                   self.logit_v_ub,
                                   100)
        
        # plot log-phi
        ax.plot(logit_v_grid, 
                   self.log_phi(logit_v_grid), 
                   color = 'grey', 
                   label = 'log-phi')
        ax.set_title('log phi in logit space')
        ax.set_xlabel(r'$\nu_k$')
        ax.set_ylabel(r'log $\phi$')
        
    def _plot_log_priors(self, ax): 
        # plots log priors in logit space
        
        # x-axis
        logit_v_grid = np.linspace(self.logit_v_lb, 
                                   self.logit_v_ub,
                                   100)
        
        # log prior -- initial
        ax.set_title('log-priors in logit space')
        ax.plot(logit_v_grid, 
                   self.get_log_p0_logit(logit_v_grid), 
                   color = 'lightblue', 
                   label = 'p0')
        
        # log prior -- perturbed
        ax.plot(logit_v_grid, 
                   self.get_log_pc_logit(logit_v_grid), 
                   color = 'blue', 
                   label = 'p1')
        
        ax.set_xlabel(r'logit-$\nu_k$')
        ax.set_ylabel(r'log p')
    
    
    def _plot_priors(self, ax): 
        # plots priors in logit-space
        
        # x-axis
        logit_v_grid = np.linspace(self.logit_v_lb, 
                                   self.logit_v_ub,
                                   100)
        
        # prior -- initial 
        ax.set_title('priors in logit space')
        ax.plot(logit_v_grid,
                   np.exp(self.get_log_p0_logit(logit_v_grid)), 
                   color = 'lightblue', 
                   label = 'p0')
        
        # prior -- perturbed
        ax.plot(logit_v_grid, 
                   np.exp(self.get_log_pc_logit(logit_v_grid)), 
                   color = 'blue', 
                   label = 'p1')
        
        ax.set_xlabel(r'logit-$\nu_k$')
        ax.set_ylabel(r'p')
    
    
    def _plot_priors_constrained(self, ax): 
        # plots priors in constrained space
        
        # x-axis
        logit_v_grid = np.linspace(self.logit_v_lb, 
                                   self.logit_v_ub,
                                   100)

        v_grid = sp.special.expit(logit_v_grid)
        
        # initial prior
        ax.set_title('priors in constrained space')
        ax.plot(v_grid, 
                   np.exp(self.get_log_p0(v_grid)),
                   color = 'lightblue', 
                   label = 'p0')
        
        # perturbed prior
        ax.plot(v_grid, 
                   np.exp(self.get_log_pc(v_grid)), 
                   color = 'blue', 
                   label = 'p1')
        ax.set_xlabel(r'$\nu_k$')
        ax.set_ylabel(r'p')

    
    # convenient wrapper to plot everything
    def plot_perturbation(self, ax = None): 
    
        if ax is None:
            fig, ax = plt.subplots(1, 4, figsize = (14, 3.5)) 

        self._plot_log_phi(ax[0])
        self._plot_log_priors(ax[1])
        self._plot_priors(ax[2])
        self._plot_priors_constrained(ax[3])

        ax[3].legend()
        

def get_e_log_perturbation(log_phi, stick_propn_mean, stick_propn_info,
                           gh_loc, gh_weights, sum_vector=True):

    """
    Computes the expected log multiplicative perturbation

    Parameters
    ----------
    log_phi : Callable function
        The multiplicative perturbation in logit space.
    stick_propn_mean : ndarray
        Mean parameters for the logit of the
        stick-breaking proportions, of length (k_approx - 1)
    stick_propn_info : ndarray
        parameters for the logit of the
        stick-breaking proportions, of length (k_approx - 1)
    gh_loc : vector
        Locations for gauss-hermite quadrature. We need this to compute the
        expectations wrt to stick-breaking proportions.
    gh_weights : vector
        Weights for gauss-hermite quadrature. We need this to compute the
        expectations wrt to stick-breaking proportions.
    sum_vector : boolean
        whether to sum the expectation over the k sticks

    Returns
    -------
    float
        The expected log perturbation under the variational distribution

    """
    
    e_perturbation_vec = get_e_fun_normal(
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
                 gh_loc, gh_weights,
                 e_log_phi = None, 
                 delta = 1.0,
                 stick_key = 'stick_params'): 
        """
        A class with methods for computing the expectation of log-phi. 
        Methods are useful as arguments to optimizers (`optimize_kl`) or the 
        sensitivity class (`HyperparameterSensitivityLinearApproximation`).
        
        Parameters
        ----------
        log_phi : Callable function
            The multiplicative perturbation in logit space. 
            Note the signs. This is an additive perturbation to the ELBO (not the KL). 
        vb_params_paragami : paragami patterned dictionary
            A paragami patterned dictionary that contains the variational parameters.
        gh_loc : vector
            Locations for gauss-hermite quadrature. We need this to compute the
            expectations wrt to stick-breaking proportions.
        gh_weights : vector
            Weights for gauss-hermite quadrature. We need this to compute the
            expectations wrt to stick-breaking proportions.
        e_log_phi : callable, optional
            A function that takes input means and infos 
            and returns the expectation of log-phi under the 
            logit-normal variational distribution. 
            If e_log_phi is not provided, we compute the expectation 
            using Gauss-Hermite quadrature. 
        delta : float 
            Factor multiplying log_phi in the perturbation. 
        stick_key : string 
            Key name of the stick parameters in `vb_params_paragami`
        """
        
        self.vb_params_paragami = vb_params_paragami
        self.stick_key = stick_key 
        
        self.gh_loc = gh_loc
        self.gh_weights = gh_weights
        
        self.delta = delta
        
        self.log_phi = lambda x : log_phi(x) * self.delta
        
        if e_log_phi is None: 
            # set the expected log-perturbation 
            # using gauss-hermite quadrature
            self._set_e_log_phi_with_gh()
        else: 
            # a pre-computed expectation. 
            # this takes 
            self.e_log_phi = \
                lambda means, infos : e_log_phi(means, infos) * \
                                        self.delta
        
    def _set_e_log_phi_with_gh(self): 
        # set the expected log-perturbation 
        # using gauss-hermite quadrature
        
        self.e_log_phi = lambda means, infos : \
                            get_e_log_perturbation(self.log_phi, 
                                                   means, 
                                                   infos,  
                                                   self.gh_loc,
                                                   self.gh_weights, 
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

    