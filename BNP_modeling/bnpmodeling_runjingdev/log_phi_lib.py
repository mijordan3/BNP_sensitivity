# this just contains a suite of possible functional perturbations 

import jax.numpy as np
import jax.scipy as sp

def sigmoidal(logit_v): 
        return (sp.special.expit(logit_v) * 2 - 1.) 

def sigmoidal_neg(logit_v): 
    return (sp.special.expit(-logit_v) * 2 - 1.) 

def alpha_pert_pos(logit_v, alpha0): 
    v = sp.special.expit(logit_v)

    alpha1 = alpha0 + 5.

    return sp.stats.beta.logpdf(v, a = 1, b = alpha1) - sp.stats.beta.logpdf(v, a = 1, b = alpha0)

def alpha_pert_neg(logit_v, alpha0): 
    v = sp.special.expit(logit_v)

    alpha1 = alpha0 - 5.

    return sp.stats.beta.logpdf(v, a = 1, b = alpha1) - sp.stats.beta.logpdf(v, a = 1, b = alpha0)
