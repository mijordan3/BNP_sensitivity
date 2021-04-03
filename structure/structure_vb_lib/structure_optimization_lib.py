import jax
import jax.numpy as np

import numpy as onp

import paragami

from structure_vb_lib.structure_model_lib import get_kl, get_e_loglik_gene_nlk
from structure_vb_lib.posterior_quantities_lib import get_optimal_z_from_vb_dict

import bnpmodeling_runjingdev.bnp_optimization_lib as bnp_optim_lib
from bnpmodeling_runjingdev.modeling_lib import get_e_log_dirichlet

def initialize_structure(g_obs, 
                         vb_params_dict, 
                         vb_params_paragami,
                         prior_params_dict,
                         gh_loc, 
                         gh_weights, 
                         seed = 1): 
    
    n_obs = g_obs.shape[0]
    k_approx = vb_params_dict['pop_freq_dirichlet_params'].shape[0]
    
    # initialize centroids with kmeans
    init_centroids, km_best = \
        bnp_optim_lib.init_centroids_w_kmeans(g_obs.reshape(n_obs, -1), 
                                k_approx, 
                                seed = seed)
    
    # within each cluster, compute empirical allele frequencies
    dirichlet_params_init = onp.zeros(vb_params_dict['pop_freq_dirichlet_params'].shape)

    for k in range(k_approx): 
        which_g = km_best.labels_ == k

        if which_g.sum() == 0: 
            dirichlet_params_init[k] = np.ones(n_loci, n_allele) 
        else: 
            # add and multiply by some small factors, to prevent the init params from 
            # being too small
            dir_params = g_obs[which_g].mean(2).mean(0) + 1.
            dirichlet_params_init[k] = dir_params / dir_params.sum(-1, keepdims=True) * 10
            
    vb_params_dict['pop_freq_dirichlet_params'] = np.array(dirichlet_params_init)

    # now conditional on dirichlet params, and ignoring the prior, set optimal ezs
    e_log_pop_freq = get_e_log_dirichlet(dirichlet_params_init)
    ez_init = jax.nn.softmax(get_e_loglik_gene_nlk(g_obs, e_log_pop_freq), axis = -1)
    
    # find optimal beta parameters for the stick-breaking distribution, 
    # conditional on the ez's 
    def beta_fun(ez_init_n): 
        beta_update1, beta_update2 = \
            bnp_optim_lib.update_stick_beta_params(ez_init_n.reshape(-1, k_approx), 
                                                          prior_params_dict['dp_prior_alpha']) 
        return np.stack((beta_update1, beta_update2), axis = -1) 

    ind_stick_betas = jax.vmap(beta_fun)(ez_init)
    
    # finally, convert these beta sticks to logitnormal sticks
    vb_params_dict['ind_admix_params'] = \
        bnp_optim_lib.convert_beta_sticks_to_logitnormal(ind_stick_betas,
                                                            vb_params_dict['ind_admix_params'],
                                                            vb_params_paragami['ind_admix_params'],
                                                            gh_loc, 
                                                            gh_weights)[0]
    
    return vb_params_dict

def optimize_structure(g_obs,
                       vb_params_dict,
                       vb_params_paragami,
                       prior_params_dict, 
                       gh_loc, 
                       gh_weights, 
                       e_log_phi = None, 
                       run_lbfgs = True,
                       run_newton = True): 
    
    ###################
    # Define loss
    ###################
    def get_kl_loss(vb_params_free): 
        
        vb_params_dict = vb_params_paragami.fold(vb_params_free, free = True)
    
        return get_kl(g_obs,
                      vb_params_dict,
                      prior_params_dict,
                      gh_loc,
                      gh_weights, 
                      e_log_phi = e_log_phi).squeeze()
    
    ###################
    # optimize
    ###################
    vb_opt_dict, vb_opt, out, optim_time = \
        bnp_optim_lib.optimize_kl(get_kl_loss,
                                   vb_params_dict, 
                                   vb_params_paragami, 
                                   run_lbfgs = run_lbfgs,
                                   run_newton = run_newton)

    ###################
    # get optimal z 
    ###################
    ez_opt = get_optimal_z_from_vb_dict(g_obs, vb_opt_dict, gh_loc, gh_weights)
    
    return vb_opt_dict, vb_opt, ez_opt, out, optim_time
    
