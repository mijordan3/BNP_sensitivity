import jax
import jax.numpy as np
import jax.scipy as sp

import paragami

from vb_lib import structure_model_lib

from bnpmodeling_runjingdev.optimization_lib import OptimizationObjectiveJaxtoNumpy

def define_structure_objective(g_obs, vb_params_dict,
                                vb_params_paragami,
                                prior_params_dict,
                                gh_loc = None, gh_weights = None,
                                log_phi = None, epsilon = 0., 
                                use_bnp_prior = True,
                                compile_hvp = False):

    # set up loss
    _kl_fun_free = paragami.FlattenFunctionInput(
                                original_fun=structure_model_lib.get_kl,
                                patterns = vb_params_paragami,
                                free = True,
                                argnums = 1)

    kl_fun_free = lambda x : _kl_fun_free(g_obs, x, prior_params_dict,
                                                     gh_loc, gh_weights,
                                                     log_phi, epsilon, 
                                                     use_bnp_prior = use_bnp_prior)

    # initial free parameters
    init_vb_free = vb_params_paragami.flatten(vb_params_dict, free = True)
    
    # define objective
    optim_objective = OptimizationObjectiveJaxtoNumpy(kl_fun_free, 
                                                     init_vb_free, 
                                                      compile_hvp = compile_hvp, 
                                                      print_every = 1,
                                                      log_every = 0)
    
    return optim_objective, init_vb_free
