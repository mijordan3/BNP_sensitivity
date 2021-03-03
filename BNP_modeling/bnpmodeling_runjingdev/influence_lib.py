import jax
import jax.numpy as np
import jax.scipy as sp

import numpy as onp

########################
# Functions to evaluate
# the influence function
########################
class InfluenceOperator(object):
    def __init__(self, 
                 vb_opt,
                 vb_params_paragami, 
                 hessian_solver,
                 alpha0, 
                 stick_key = 'stick_params'):
        
        # vb_opt is the vector of optimal vb parameters
        # hessian solver is a function that takes an input vector of len(vb_opt)
        # and returns H^{-1}v
        # alpha0 is the DP prior parameter

        self.vb_opt = vb_opt
        self.vb_params_paragami = vb_params_paragami
        self.hessian_solver = hessian_solver
        self.alpha0 = alpha0

        # stick densities
        # this returns the per stick density
        # (first dimension is k, the stick index)
        self.get_log_qk = lambda logit_stick, vb_free_params : \
                        get_log_qk_from_free_params(logit_stick,
                                                    vb_free_params,
                                                    self.vb_params_paragami, 
                                                    stick_key)

        # this returns the sum over sticks
        self.get_log_q = lambda logit_stick, vb_free_params : \
                            self.get_log_qk(logit_stick, vb_free_params).sum(0)

        self.grad_log_q = jax.jacobian(self.get_log_q, argnums = 1)

        # a len(vb_opt) x (k_approx - 1) binary matrix
        # with 1 if the ith vb free parameter affects the jth stick distribution.
        self.stick_params_mapping = \
                get_stick_params_mapping(self.vb_params_paragami, 
                                         stick_key)

    def get_influence(self,
                      logit_stick,
                      grad_g = None, 
                      normalize_by_prior = True):

        # this is len(vb_opt) x len(logit_stick)
        grad_log_q = self.grad_log_q(logit_stick, self.vb_opt).transpose()

        # this is (k_approx - 1) x len(logit_stick)
        prior_ratio = np.exp(self.get_q_prior_log_ratio(logit_stick, normalize_by_prior))
        
        # map each stick to appropriate vb free param
        # this is len(vb_opt) x len(logit_stick)
        prior_ratio_expanded = np.dot(self.stick_params_mapping, prior_ratio)

        # combine prior ratio and grad log q
        grad_log_q_prior_rat = grad_log_q * prior_ratio_expanded
        
        # solve
        if grad_g is None: 
            print('warning this might be slow ...')
            # somehow jax.lax.map is super slow?
            # jut using python for loop here ...
            influence = \
                np.stack([self.hessian_solver(x) \
                            for x in grad_log_q_prior_rat.transpose()])
            influence = influence.transpose()
            
            return influence
            
        else: 
            assert len(grad_g) == len(self.vb_opt)
            grad_g_hess_inv = self.hessian_solver(grad_g).block_until_ready()
            influence = np.dot(grad_g_hess_inv, grad_log_q_prior_rat)

            return influence, grad_g_hess_inv

    def get_q_prior_log_ratio(self, logit_stick, normalize_by_prior = True):
        # this is log q(logit_stick)  - log p_0(logit_stick)
        # returns a matrix of (k_approx - 1) x length(logit_stick)
        
        if normalize_by_prior: 
            log_beta_prior = get_log_logitstick_prior(logit_stick, self.alpha0)
            log_beta_prior = np.expand_dims(log_beta_prior, 0)
        else: 
            log_beta_prior = 0.
            
        log_ratio = self.get_log_qk(logit_stick, self.vb_opt) - log_beta_prior

        return log_ratio

# TODO implemented for structure model (with 2d sticks)
# needs to be tested for 1d sticks
def get_stick_params_mapping(vb_params_paragami, stick_key): 

    vb_bool_dict = vb_params_paragami.empty_bool(False)    
    
    stick_shape = vb_bool_dict[stick_key]['stick_means'].shape
    n_sticks = len(vb_bool_dict[stick_key]['stick_means'].flatten())
    
    stick_params_mapping = onp.zeros((vb_params_paragami.flat_length(free = True), 
                                     n_sticks), 
                                     dtype = bool)
    
    for k in range(n_sticks): 
        
        stick_bool = onp.zeros(n_sticks, dtype = bool)
        stick_bool[k] = True
        
        vb_bool_dict[stick_key]['stick_means'] = stick_bool.reshape(stick_shape)
        vb_bool_dict[stick_key]['stick_infos'] = stick_bool.reshape(stick_shape)
        
        flat_indices = vb_params_paragami.flat_indices(vb_bool_dict, free = True)
        
        stick_params_mapping[flat_indices, k] = True
        
    return stick_params_mapping

# get explicit density (not expectations) for logit-sticks
# log p_0
def get_log_logitstick_prior(logit_stick, alpha):
    # pi are the stick lengths
    # alpha is the DP parameter

    stick = sp.special.expit(logit_stick)
    return sp.stats.beta.logpdf(stick, a = 1., b = alpha) + \
                np.log(stick) + np.log(1 - stick)


def get_log_qk_from_free_params(logit_stick, vb_free_params, 
                                vb_params_paragami, stick_key):
    
    vb_params_dict = vb_params_paragami.fold(vb_free_params, free = True)

    mean = vb_params_dict[stick_key]['stick_means'].flatten()
    info = vb_params_dict[stick_key]['stick_infos'].flatten()

    logit_stick = np.expand_dims(logit_stick, 0)
    mean = np.expand_dims(mean, 1)
    info = np.expand_dims(info, 1)

    return sp.stats.norm.logpdf(logit_stick, mean, scale = 1 / np.sqrt(info))



class WorstCasePerturbation(object):
    def __init__(self,
                 influence_fun, 
                 logit_v_grid, 
                 delta = 1.,
                 cached_influence_grid = None):
        
        # influence function is a function that takes logit-sticks
        # and returns a scalar value for the influence

        self.logit_v_grid = logit_v_grid
        self.v_grid = sp.special.expit(self.logit_v_grid)

        self.influence_fun = influence_fun
        if cached_influence_grid is None: 
            self.influence_grid = self.influence_fun(self.logit_v_grid)
        else: 
            assert len(cached_influence_grid) == len(logit_v_grid)
            self.influence_grid = cached_influence_grid
            
        self.len_grid = len(self.influence_grid)
        
        self.delta = delta

        self._set_linf_worst_case()

    def _set_linf_worst_case(self):
        # the points at which the influence changes sign

        s_influence1 = np.sign(self.influence_grid)[1:self.len_grid]
        s_influence2 = np.sign(self.influence_grid)[0:(self.len_grid - 1)]
        self._sign_diffs = - s_influence1 + s_influence2

        # the points at which the influence changes sign
        self.change_bool = self._sign_diffs != 0
        self.change_points = self.logit_v_grid[self.change_bool]

        # the signs
        self.signs = s_influence2[self.change_bool]
        self.signs = np.concatenate((self.signs, self.signs[-1][None] * -1))
        self.sign_diffs = self._sign_diffs[self.change_bool]

    def get_e_log_linf_perturbation(self, means, infos):
        
        # in structure, means are 2d. 
        # flatten them (shouldn't matter for iris)
        means = means.flatten()
        infos = infos.flatten()
        
        x = np.expand_dims(self.change_points, axis = 0)
        loc = np.expand_dims(means, axis = 1)
        scale = np.expand_dims(1 / np.sqrt(infos), axis = 1)

        cdf = sp.stats.norm.cdf(x, loc, scale)

        e_log_pert = (cdf * np.expand_dims(self.sign_diffs, 0)).sum()

        # extra term doenst matter, just for unittesting
        # so constants match
        return  (e_log_pert + self.signs[-1] * len(means)) * self.delta
