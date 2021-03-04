import jax 
import jax.numpy as np

import numpy as onp
from numpy.polynomial.hermite import hermgauss

import paragami

# GMM libary
from bnpgmm_runjingdev import gmm_clustering_lib as gmm_lib
from bnpgmm_runjingdev import gmm_posterior_quantities_lib

def load_initial_fit(init_fit_file, iris_obs): 
    
    dim = iris_obs.shape[-1]
    
    vb_opt_dict, vb_params_paragami, init_fit_meta_data = \
            paragami.load_folded(init_fit_file)
        
    vb_opt = vb_params_paragami.flatten(vb_opt_dict, free = True)    
    
    # gauss-hermite parameters
    gh_deg = int(init_fit_meta_data['gh_deg'])
    gh_loc, gh_weights = hermgauss(gh_deg)

    gh_loc = np.array(gh_loc)
    gh_weights = np.array(gh_weights)

    # load prior parameters
    prior_params_dict, prior_params_paragami = gmm_lib.get_default_prior_params(dim)

    # set initial alpha
    alpha0 = init_fit_meta_data['dp_prior_alpha']
    prior_params_dict['dp_prior_alpha'] = alpha0

    
    # check the objective 
    kl = gmm_lib.get_kl(iris_obs, vb_opt_dict, prior_params_dict, gh_loc, gh_weights)
    
    assert kl == init_fit_meta_data['final_kl']
    
    # return ez
    e_z_opt = gmm_posterior_quantities_lib.get_optimal_z_from_vb_dict(iris_obs,
                               vb_opt_dict,
                               gh_loc,
                               gh_weights,
                               use_bnp_prior = True)
    
    return vb_opt, vb_opt_dict, e_z_opt, \
             vb_params_paragami, \
                prior_params_dict, \
                    gh_loc, gh_weights
            