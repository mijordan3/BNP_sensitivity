import jax

from scipy import optimize 

import paragami

from structure_vb_lib.structure_model_lib import get_kl
from structure_vb_lib.posterior_quantities_lib import get_optimal_z_from_vb_dict

from bnpmodeling_runjingdev.bnp_optimization_lib import optimize_kl


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
    vb_opt_dict, vb_opt, out, optim_time = optimize_kl(get_kl_loss,
                                                       vb_params_dict, 
                                                       vb_params_paragami, 
                                                       run_lbfgs = run_lbfgs,
                                                       run_newton = run_newton)

    ###################
    # get optimal z 
    ###################
    ez_opt = get_optimal_z_from_vb_dict(g_obs, vb_opt_dict, gh_loc, gh_weights)
    
    return vb_opt_dict, vb_opt, ez_opt, out, optim_time
    
