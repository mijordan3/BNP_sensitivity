import jax
import jax.numpy as np

import time 

# GMM libary
from bnpgmm_runjingdev import gmm_clustering_lib as gmm_lib
from bnpgmm_runjingdev import gmm_posterior_quantities_lib

# BNP libraries
from bnpmodeling_runjingdev.sensitivity_lib import HyperparameterSensitivityLinearApproximation, get_cross_hess
from bnpmodeling_runjingdev import result_loading_utils, influence_lib
import bnpmodeling_runjingdev.functional_sensitivity_lib as func_sens_lib

class PosteriorStatistics(object):
    
    # methods are posterior statistics of interest 
    # as a function of vb-free parameters
    
    def __init__(self, 
                 iris_obs, 
                 vb_params_paragami, 
                 gh_loc, 
                 gh_weights): 
        
        self.iris_obs = iris_obs
        self.vb_params_paragami = vb_params_paragami
        
        self.gh_loc = gh_loc
        self.gh_weights = gh_weights
        
        self.prng_key = jax.random.PRNGKey(25435)
        
    
    def get_mixture_weight_k(self, vb_free, k = 0): 
        
        vb_params_dict = self.vb_params_paragami.fold(vb_free, free = True)

        return gmm_posterior_quantities_lib.get_e_mixture_weights_from_vb_dict(vb_params_dict,
                                                                               self.gh_loc,
                                                                               self.gh_weights)[k]
    
    def get_n_clusters_insample(self, vb_free):

        vb_params_dict = self.vb_params_paragami.fold(vb_free, free = True)

        return gmm_posterior_quantities_lib.get_e_num_clusters_from_vb_dict(self.iris_obs,
                                                                            vb_params_dict,
                                                                            self.gh_loc,
                                                                            self.gh_weights,
                                                                            threshold = 0,
                                                                            n_samples = 10000, 
                                                                            prng_key = self.prng_key)
    
    def get_n_clusters_pred(self, vb_free):

        vb_params_dict = self.vb_params_paragami.fold(vb_free, free = True)

        return gmm_posterior_quantities_lib.get_e_num_pred_clusters_from_vb_dict(vb_params_dict,
                                                                                 n_obs = self.iris_obs.shape[0],
                                                                                 threshold = 0,
                                                                                 n_samples = 10000,
                                                                                 prng_key = self.prng_key)
    
# the prior (for plotting)
def p0(logit_v, alpha0): 
    return np.exp(influence_lib.get_log_logitstick_prior(logit_v, alpha0))



class InfluenceFunctions(): 
    def __init__(self,
                 iris_obs,
                 vb_opt, 
                 vb_params_paragami,
                 prior_params_dict, 
                 gh_loc, 
                 gh_weights): 
        
        self.iris_obs = iris_obs
        
        self.vb_opt = vb_opt
        self.vb_params_paragami = vb_params_paragami
        
        self.prior_params_dict = prior_params_dict
        
        self.gh_loc = gh_loc
        self.gh_weights = gh_weights
        
        # get vb sensitivity 
        self.vb_sens = \
            HyperparameterSensitivityLinearApproximation(self.objective_fun,
                                                         self.vb_opt,
                                                         0.)
        
        # class to get influence functions
        self.influence_operator = influence_lib.InfluenceOperator(self.vb_opt, 
                                                                  self.vb_params_paragami, 
                                                                  self.vb_sens.hessian_solver,
                                                                  self.prior_params_dict['dp_prior_alpha'],
                                                                  stick_key = 'stick_params')
    
    def objective_fun(self, vb_params_free, epsilon): 

        # NOTE! epsilon doesn't actual enter 
        # into this function. 

        # since the initial fit is at epsilon = 0, 
        # we just return the actual KL

        # we really just need the hessian wrt to vb free parameters ... 

        vb_params_dict = self.vb_params_paragami.fold(vb_params_free, 
                                                 free = True)

        return gmm_lib.get_kl(self.iris_obs, 
                              vb_params_dict,
                              self.prior_params_dict,
                              self.gh_loc,
                              self.gh_weights).squeeze()
    
    def get_influence(self, g, logit_v_grid): 
        print('computing gradient ...')
        t0 = time.time()
        get_grad_g = jax.jacobian(g, argnums = 0)
        grad_g = get_grad_g(self.vb_opt).block_until_ready()
        grad_g_time = time.time() - t0  
        print('Elapsed: {:.03f}sec'.format(grad_g_time))

        # get influence function
        print('inverting Hessian (twice) ...')
        t0 = time.time()

        # get influence function as defined
        influence_grid, grad_g_hess_inv = \
            self.influence_operator.get_influence(logit_v_grid, 
                                                  grad_g)


        # this is influence times the prior
        influence_grid_x_prior, _ = \
            self.influence_operator.get_influence(logit_v_grid, 
                                                  grad_g, 
                                                  normalize_by_prior = False)

        hess_inv_time = time.time() - t0
        print('Elapsed: {:.03f}sec'.format(hess_inv_time))

        return influence_grid, influence_grid_x_prior, grad_g_hess_inv
