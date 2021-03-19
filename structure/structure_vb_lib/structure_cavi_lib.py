import jax
import jax.numpy as np

import numpy as onp
from scipy.optimize import minimize

import paragami
import time 

from structure_vb_lib import structure_model_lib 

from bnpmodeling_runjingdev.bnp_optimization_lib import optimize_kl
from bnpmodeling_runjingdev.sensitivity_lib import get_jac_hvp_fun


def get_optimal_ec(g_obs, ez, e_log_pop_freq, e_log_pop_cluster_probs): 
    
    g_times_popfreq = np.einsum('nlia, kla -> nlik', g_obs, e_log_pop_freq)
    
    ec_free = np.einsum('nlij, nlik -> njk', ez, g_times_popfreq) + \
                e_log_pop_cluster_probs
    
    ec = jax.nn.softmax(ec_free, axis = -1)
    
    return ec, ec_free



def run_cavi(g_obs, 
             vb_params_dict, 
             vb_params_paragami,
             prior_params_dict,
             gh_loc, gh_weights,
             max_iter = 100,
             e_log_phi = None): 
    
    def get_partial_kl(vb_global_params_free, ez_free, ec_free): 

        # fold parameters 
        vb_params_dict['global_params'] = vb_params_paragami['global_params'].fold(vb_global_params_free,
                                                                  free = True)
        ez = jax.nn.softmax(ez_free, axis = -1)
        ec = jax.nn.softmax(ec_free, axis = -1)

        return structure_model_lib.get_kl(g_obs, 
                                           vb_params_dict, 
                                           prior_params_dict,
                                           gh_loc, 
                                           gh_weights, 
                                           e_log_phi = e_log_phi,
                                           e_z = ez, 
                                           e_c = ec)
    
    # compile functions 
    get_loss = jax.jit(get_partial_kl)
    get_grad = jax.jit(jax.grad(get_partial_kl, 0))
    
    # initialize 
    global_params_free = vb_params_paragami['global_params'].flatten(vb_params_dict['global_params'], free = True)
    ec = vb_params_dict['pop_indx_multinom_params']
    
    for i in range(max_iter): 
        t0 = time.time()
        # get moments
        moments_dict = structure_model_lib.get_global_moments(vb_params_dict['global_params'],
                                                      gh_loc = gh_loc,
                                                      gh_weights = gh_weights)
        
        
        # update ez
        ez, ez_free = structure_model_lib.get_optimal_z(g_obs, 
                                                       moments_dict['e_log_pop_freq'], 
                                                       ec, 
                                                       moments_dict['e_log_ind_cluster_probs'])
        
        # update ec
        ec, ec_free = get_optimal_ec(g_obs, 
                                     ez, 
                                     moments_dict['e_log_pop_freq'], 
                                     moments_dict['e_log_pop_cluster_probs'])
        
        # update global parameters
        out = minimize(fun = lambda x : onp.array(get_loss(x, ez_free, ec_free)), 
                       x0 = global_params_free, 
                       method = 'L-BFGS-B', 
                       jac = lambda x : onp.array(get_grad(x, ez_free, ec_free)))
        
        kl = get_partial_kl(out.x, ez_free, ec_free)
        print('iteration [{}]; kl:{}; elapsed: {}secs'.format(i,
                                        round(kl, 6),
                                        round(time.time() - t0, 4)))
        
        global_params_free = out.x
        vb_params_dict['global_params'] = vb_params_paragami['global_params'].fold(global_params_free, free = True)
        
    vb_params_dict['pop_indx_multinom_params'] = ec
    
    return vb_params_dict