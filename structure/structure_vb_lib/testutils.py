import jax 

from numpy.polynomial.hermite import hermgauss

from structure_vb_lib import structure_model_lib, data_utils
import structure_vb_lib.structure_optimization_lib as s_optim_lib

def draw_data_and_construct_model(n_obs = 5, 
                                  n_loci = 10, 
                                  n_pop = 3, 
                                  k_approx = 4, 
                                  use_logitnormal_sticks = True,
                                  seed = 4525):
    # draw data
    g_obs = data_utils.draw_data(n_obs, n_loci, n_pop)[0]

    # prior parameters
    _, prior_params_paragami = \
        structure_model_lib.get_default_prior_params()
    prior_params_dict = \
        prior_params_paragami.random(key=jax.random.PRNGKey(seed))

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

def construct_model_and_optimize(n_obs = 5, 
                                  n_loci = 10, 
                                  n_pop = 3, 
                                  k_approx = 4, 
                                  seed = 2343, 
                                  debug_cavi = True): 
    
    # draw model
    g_obs, _, vb_params_paragami, \
        prior_params_dict, gh_loc, gh_weights = \
            draw_data_and_construct_model(n_obs, 
                                          n_loci, 
                                          n_pop, 
                                          k_approx, 
                                          use_logitnormal_sticks = True)
    # initialize with cavi
    vb_params_dict, _ = \
            s_optim_lib.initialize_with_cavi(g_obs, 
                                             vb_params_paragami, 
                                             prior_params_dict, 
                                             gh_loc, gh_weights, 
                                             debug_cavi = debug_cavi, 
                                             seed = seed)
    
    # run lbfgs
    vb_opt_dict, vb_opt, out, precond_objective, lbfgs_time = \
        s_optim_lib.run_preconditioned_lbfgs(g_obs, 
                                             vb_params_dict, 
                                             vb_params_paragami,
                                             prior_params_dict,
                                             gh_loc, gh_weights)
    
    return vb_opt, vb_params_paragami, precond_objective