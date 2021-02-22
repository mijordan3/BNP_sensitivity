import jax
import jax.numpy as np

import numpy as onp
from scipy import optimize 

import bnpmodeling_runjingdev.exponential_families as ef
from bnpmodeling_runjingdev import modeling_lib
from bnpmodeling_runjingdev.sensitivity_lib import get_jac_hvp_fun

def update_stick_beta_params(ez, dp_prior_alpha): 
    
    # conditional on ez, returns the optimal beta parameters 
    # for the stick-breaking process. 
    
    # ez are allocations of shape n_obs x k_approx
    
    k_approx = ez.shape[1]
    
    weights = ez.sum(0)
    
    beta_update1 = weights[0:(k_approx-1)] + 1
    
    tmp = weights[1:k_approx]
    beta_update2 = np.cumsum(np.flip(tmp)) + \
                        (dp_prior_alpha - 1) + 1
    
    beta_update2 = np.flip(beta_update2)
    
    return beta_update1, beta_update2


#########################
# Function to convert beta sticks to 
# logitnormal sticks
#########################
def convert_beta_sticks_to_logitnormal(stick_betas, 
                                       logitnorm_stick_params_dict,
                                       logitnorm_stick_params_paragami, 
                                       gh_loc, gh_weights): 
    """
    Given a set of beta parameters for stick-breaking proportions, 
    return the logitnormal stick parameters that have the same
    expected log(stick) and expected log(1 - stick). 
    
    Parameters
    ----------
    stick_betas : array
        array (... x 2) of beta parameters 
        on individual admixture stick-breaking proportions.
    logitnorm_stick_params_dict : dictionary
        parameter dictionary of logitnormal parameters
        (stick_means, stick_infos) for individual admixture
        stick-breaking proportions
    logitnorm_stick_params_paragami : paragami patterned dictionary
        A paragami patterned dictionary that contains the variational
        parameters
    gh_loc : vector
        Locations for gauss-hermite quadrature. 
    gh_weights : vector
        Weights for gauss-hermite quadrature. 
        
    Returns
    -------
    opt_logitnorm_stick_params : dictionary
        A dictionary that contains the variational parameters
        for individual admixture stick-breaking
        proportions. 
    out : scipy.optimize.minimize output
    """

    
    # check shapes
    assert stick_betas.shape[-1] == 2
    
    # the moments from the beta parameters
    target_sticks, target_1m_sticks = modeling_lib.get_e_log_beta(stick_betas)
    
    # square error loss
    def _loss(stick_params_free): 

        logitnorm_stick_params_dict = \
            logitnorm_stick_params_paragami.fold(stick_params_free, 
                                                 free = True)

        stick_means = logitnorm_stick_params_dict['stick_means']
        stick_infos = logitnorm_stick_params_dict['stick_infos']

        e_log_sticks, e_log_1m_sticks = \
            ef.get_e_log_logitnormal(
                lognorm_means = stick_means,
                lognorm_infos = stick_infos,
                gh_loc = gh_loc,
                gh_weights = gh_weights)
    
        loss = (e_log_sticks - target_sticks)**2 +\
                (e_log_1m_sticks - target_1m_sticks)**2
        
        return loss.sum()
    
    # optimize
    loss = jax.jit(_loss)
    loss_grad = jax.jit(jax.grad(_loss))
    loss_hvp = jax.jit(get_jac_hvp_fun(_loss))
    
    stick_params_free = \
        logitnorm_stick_params_paragami.flatten(logitnorm_stick_params_dict, 
                                                free = True)
    
    out = optimize.minimize(fun = lambda x : onp.array(loss(x)), 
                                  x0 = stick_params_free, 
                                  jac = lambda x : onp.array(loss_grad(x)), 
                                  hessp = lambda x,v : onp.array(loss_hvp(x, v)), 
                                  method = 'trust-ncg')
    
    opt_logitnorm_stick_params = \
        logitnorm_stick_params_paragami.fold(out.x, free = True)
    
    return opt_logitnorm_stick_params, out
