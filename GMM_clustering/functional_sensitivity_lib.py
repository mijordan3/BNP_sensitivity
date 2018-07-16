# some extra functions to do functional sensitivity
import sys
sys.path.insert(0, './../../LinearResponseVariationalBayes.py')

import autograd.numpy as np
import autograd.scipy as sp
import LinearResponseVariationalBayes.ExponentialFamilies as ef

def dp_perturbed_prior(vb_params, prior_params, u = lambda x : 0):
    # expectation of perturbed dp prior

    alpha = prior_params['alpha'].get()
    dp_prior_density = lambda x : 1 / sp.special.beta(1, alpha) \
                            * (1 - x)**(alpha - 1)
    perturbed_log_density = lambda x : np.log(u(x) + dp_prior_density(x))

    integrand = lambda x : perturbed_log_density(sp.special.expit(x))

    lognorm_means = vb_params['global']['v_sticks']['mean'].get()
    lognorm_infos = vb_params['global']['v_sticks']['info'].get()
    gh_loc = vb_params.gh_loc
    gh_weights = vb_params.gh_weights

    return np.sum(ef.get_e_fun_normal(lognorm_means, lognorm_infos, \
                            gh_loc, gh_weights, integrand))

# get explicit densities (not expectations) for computing the influence function
def get_log_logitnormal_density(theta, mean, info):
    return - 0.5 * (np.log(2 * np.pi) - np.log(info)) + \
            -0.5 * info * (sp.special.logit(theta) - mean) ** 2 + \
            -np.log(theta) - np.log(1 - theta)


def get_log_beta_prior(pi, alpha):
    # pi are the stick lengths
    # alpha is the DP parameter

    return (alpha - 1.0) * np.log(1.0 - pi)
