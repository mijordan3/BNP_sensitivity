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
                 logit_v_grid = None, 
                 influence_grid = None, 
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
            func_sens_lib.FunctionalPerturbationObjective(lambda x : - gauss_pert1(x),
                                                     vb_params_paragami, 
                                                     gh_loc, 
                                                     gh_weights, 
                                                     delta = delta,
                                                     stick_key = stick_key)
        ##############
        # Worst-case perturbation
        ##############
        if influence_grid is not None: 
            worst_case_pert = \
                influence_lib.WorstCasePerturbation(influence_fun = None, 
                                                    logit_v_grid = logit_v_grid, 
                                                    cached_influence_grid = influence_grid)
            
            # interpolate influence function w step functions
            def influence_fun_interp(logit_v): 
                # find index of logit_v_grid 
                # closest (on the left) to logit_v
                indx = np.searchsorted(worst_case_pert.logit_v_grid, logit_v)

                # return the influence function at those points
                return worst_case_pert.influence_grid[indx]

            # define log phi
            def log_phi(logit_v):
                return np.sign(influence_fun_interp(logit_v))

        
            self.f_obj_worst_case = \
                func_sens_lib.FunctionalPerturbationObjective(log_phi, 
                                                         vb_params_paragami, 
                                                         gh_loc, 
                                                         gh_weights, 
                                                         e_log_phi = worst_case_pert.get_e_log_linf_perturbation, 
                                                         delta = delta,
                                                         stick_key = stick_key)