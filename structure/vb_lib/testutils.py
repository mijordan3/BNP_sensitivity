from numpy.polynomial.hermite import hermgauss

from vb_lib import structure_model_lib, data_utils

def draw_data_and_construct_model(n_obs = 5, 
                                  n_loci = 10, 
                                  n_pop = 3, 
                                  k_approx = 4, 
                                  use_logitnormal_sticks = True,
                                  seed = 4525):
    # draw data
    g_obs = data_utils.draw_data(n_obs, n_loci, n_pop)[0]

    # prior parameters
    prior_params_dict, prior_params_paragami = \
        structure_model_lib.get_default_prior_params()

    # vb params
    gh_deg = 8
    gh_loc, gh_weights = hermgauss(gh_deg)

    vb_params_dict, vb_params_paragami = \
        structure_model_lib.\
            get_vb_params_paragami_object(n_obs, 
                                          n_loci,
                                          k_approx,
                                          use_logitnormal_sticks, 
                                          seed = seed)
    
    return g_obs, vb_params_dict, vb_params_paragami, \
            prior_params_dict, gh_loc, gh_weights
