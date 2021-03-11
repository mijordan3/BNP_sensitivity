# this just contains a suite of possible functional perturbations 

import jax.numpy as np
import jax.scipy as sp

import bnpmodeling_runjingdev.functional_sensitivity_lib as func_sens_lib
from bnpmodeling_runjingdev import influence_lib

def sigmoidal(logit_v): 
        return (sp.special.expit(logit_v) * 2 - 1.) 

def alpha_pert_pos(logit_v, alpha0): 
    v = sp.special.expit(logit_v)

    alpha1 = alpha0 + 5.

    return sp.stats.beta.logpdf(v, a = 1, b = alpha1) - sp.stats.beta.logpdf(v, a = 1, b = alpha0)

def alpha_pert_neg(logit_v, alpha0): 
    v = sp.special.expit(logit_v)

    alpha1 = alpha0 - 5.

    return sp.stats.beta.logpdf(v, a = 1, b = alpha1) - sp.stats.beta.logpdf(v, a = 1, b = alpha0)

def gauss_pert1(logit_v): 
    return sp.stats.norm.pdf(logit_v, loc = 0)  * np.sqrt(2 * np.pi)

def gauss_pert2(logit_v): 
    return sp.stats.norm.pdf(logit_v, loc = -3)  * np.sqrt(2 * np.pi)

class LogPhiPerturbations(): 
    def __init__(self, 
                 vb_params_paragami, 
                 alpha0,
                 gh_loc, 
                 gh_weights,
                 delta=1.0,
                 stick_key = 'stick_params'): 
        
        ##############
        # Sigmoidal perturbations
        ##############
        self.f_obj_sigmoidal = func_sens_lib.FunctionalPerturbationObjective(sigmoidal, 
                                                     vb_params_paragami, 
                                                     gh_loc, 
                                                     gh_weights, 
                                                     delta = delta,
                                                     stick_key = stick_key)
                
        ##############
        # alpha-type perturbations
        ##############
        self.f_obj_alpha_pert_pos = \
            func_sens_lib.FunctionalPerturbationObjective(lambda x : alpha_pert_pos(x, alpha0),
                                                     vb_params_paragami, 
                                                     gh_loc, 
                                                     gh_weights, 
                                                     delta = delta,
                                                     stick_key = stick_key)
        
        self.f_obj_alpha_pert_neg = \
            func_sens_lib.FunctionalPerturbationObjective(lambda x : alpha_pert_neg(x, alpha0),
                                                     vb_params_paragami, 
                                                     gh_loc, 
                                                     gh_weights, 
                                                     delta = delta,
                                                     stick_key = stick_key)
        
        ##############
        # Flip alpha-perturbation along x-axis
        ##############
        self.f_obj_alpha_pert_pos_xflip = \
            func_sens_lib.FunctionalPerturbationObjective(lambda x : alpha_pert_pos(-x, alpha0),
                                                     vb_params_paragami, 
                                                     gh_loc, 
                                                     gh_weights, 
                                                     delta = delta,
                                                     stick_key = stick_key)
        
        self.f_obj_alpha_pert_neg_xflip = \
            func_sens_lib.FunctionalPerturbationObjective(lambda x : alpha_pert_neg(-x, alpha0),
                                                     vb_params_paragami, 
                                                     gh_loc, 
                                                     gh_weights, 
                                                     delta = delta,
                                                     stick_key = stick_key)
        
        
        ##############
        # gaussian perturbations
        ##############
        self.f_obj_gauss_pert1 = \
            func_sens_lib.FunctionalPerturbationObjective(gauss_pert1,
                                                     vb_params_paragami, 
                                                     gh_loc, 
                                                     gh_weights, 
                                                     delta = delta,
                                                     stick_key = stick_key)
        
        self.f_obj_gauss_pert2 = \
            func_sens_lib.FunctionalPerturbationObjective(gauss_pert2,
                                                     vb_params_paragami, 
                                                     gh_loc, 
                                                     gh_weights, 
                                                     delta = delta,
                                                     stick_key = stick_key)

class StepPerturbations(): 
    def __init__(self, mu_vec): 
        # grid of locations for the step functions
        self.mu_vec = mu_vec
        
    def e_step_bump(self, means, infos, mu_indx): 
        loc = means
        scale = 1 / np.sqrt(infos)
        
        cdf1 = sp.stats.norm.cdf(self.mu_vec[mu_indx+1], loc = loc, scale = scale)
        cdf2 = sp.stats.norm.cdf(self.mu_vec[mu_indx], loc = loc, scale = scale)
        
        return (cdf1 - cdf2).sum() 

