import jax 
import jax.numpy as np

import numpy as onp

import time

from bnpmodeling_runjingdev.sensitivity_lib import get_jac_hvp_fun

from bnpreg_runjingdev import regression_mixture_lib 
from bnpreg_runjingdev.regression_posterior_quantities import get_optimal_z_from_vb_dict

from scipy.optimize import minimize
        
def optimize_regression_mixture(gamma, gamma_info, 
                                vb_params_dict, 
                                vb_params_paragami,
                                prior_params_dict, 
                                gh_loc, gh_weights, 
                                e_log_phi = None, 
                                run_newton = True): 
    
    ###################
    # Define loss
    ###################
    def _get_loss(vb_params_free): 
        
        vb_params_dict = vb_params_paragami.fold(vb_params_free, free = True)
    
        return regression_mixture_lib.get_kl(gamma, 
                                             gamma_info,
                                             vb_params_dict,
                                             prior_params_dict,
                                             gh_loc,
                                             gh_weights)
    
    get_loss = jax.jit(_get_loss)
    get_grad = jax.jit(jax.grad(_get_loss))
    get_hvp = jax.jit(get_jac_hvp_fun(_get_loss))
    
    
    ################
    # compile objective functions
    ################
    # intial point
    x0 = vb_params_paragami.flatten(vb_params_dict, free = True)
    
    print('compiling objective and derivatives ... ')
    t0 = time.time()
    _ = get_loss(x0).block_until_ready()
    _ = get_grad(x0).block_until_ready()
    _ = get_hvp(x0, x0).block_until_ready()
    print('done. Compile time: {0:.03f}sec'.format(time.time() - t0))
    
    ################
    # initialize with L-BFGS-B
    ################
    print('Running L-BFGS-B ...')
    t0 = time.time() 
    out = minimize(fun = lambda x : onp.array(get_loss(x)), 
                         x0 = x0, 
                         method = 'L-BFGS-B', 
                         jac = lambda x : onp.array(get_grad(x)))    
    print('L-BFGS-B time: {:.03f}sec'.format(time.time() - t0))
    
    ################
    # run a few more newton steps
    ################
    if run_newton: 
        t1 = time.time() 
        print('Running trust-ncg ... ')
        out = minimize(fun = lambda x : onp.array(get_loss(x)), 
                   x0 = out.x, 
                   method = 'trust-ncg', 
                   jac = lambda x : onp.array(get_grad(x)), 
                   hessp = lambda x,v : onp.array(get_hvp(x, v)))
        print('Newton time: {:.03f}sec'.format(time.time() - t1))

    vb_opt = out.x
    vb_opt_dict = vb_params_paragami.fold(vb_opt, free = True)
    print(out.message)
    
    print('done. ')
    
    optim_time = time.time() - t0
    
    # compute optimal ez
    ez_opt = get_optimal_z_from_vb_dict(gamma, gamma_info,
                                        vb_opt_dict,
                                        gh_loc, gh_weights)
    
    return vb_opt_dict, vb_opt, ez_opt, out, optim_time
