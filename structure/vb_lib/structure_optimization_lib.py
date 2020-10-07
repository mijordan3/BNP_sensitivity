import jax
import jax.numpy as np
import jax.scipy as sp

import paragami

from vb_lib import structure_model_lib

from bnpmodeling_runjingdev import optimization_lib

def optimize_structure(g_obs, vb_params_dict,
                        vb_params_paragami,
                        prior_params_dict,
                        gh_loc = None, gh_weights = None,
                        log_phi = None, epsilon = 0.):

    # set up loss
    _kl_fun_free = paragami.FlattenFunctionInput(
                                original_fun=structure_model_lib.get_kl,
                                patterns = vb_params_paragami,
                                free = True,
                                argnums = 1)

    kl_fun_free = lambda x : _kl_fun_free(g_obs, x, prior_params_dict,
                                                     gh_loc, gh_weights,
                                                     log_phi, epsilon)

    # initial free parameters
    init_vb_free = vb_params_paragami.flatten(vb_params_dict, free = True)

    # optimize
    optim_out = optimization_lib.optimize_full(kl_fun_free, init_vb_free)

    # construct optimum
    vb_opt = out.x
    vb_params_dict = vb_params_paragami.fold(vb_opt, free = True)

    return vb_params_dict, vb_out, optim_out
