import jax
import jax.numpy as np
import jax.scipy as sp

# copied over from the LinearResponseVariationalBayes.py repo

##########################
# Numeric integration functions

def get_e_fun_normal(means, infos, gh_loc, gh_weights, fun):
    # compute E(fun(X)) where X is an array of normals defined by parameters
    # means and infos, and fun is a function that can evaluate arrays
    # componentwise

    # gh_loc and g_weights are sample points and weights, respectively,
    # chosen such that they will correctly integrate p(x) \exp(-x^2) over
    # (-Inf, Inf), for any polynomial p of degree 2*deg - 1 or less

    assert means.shape == infos.shape
    draws_axis = means.ndim
    change_of_vars = np.sqrt(2) * gh_loc * \
                1/np.sqrt(np.expand_dims(infos, axis = draws_axis)) + \
                np.expand_dims(means, axis = draws_axis)

    integrand = fun(change_of_vars)

    return np.sum(1/np.sqrt(np.pi) * gh_weights * integrand, axis = draws_axis)

def get_e_logitnormal(lognorm_means, lognorm_infos, gh_loc, gh_weights):
    # get the expectation of a logit normal distribution
    identity_fun = lambda x : sp.special.expit(x)

    return get_e_fun_normal(lognorm_means, lognorm_infos, \
                            gh_loc, gh_weights, identity_fun)

def get_e_log_logitnormal(lognorm_means, lognorm_infos, gh_loc, gh_weights):
    # get expectation of Elog(X) and E[1 - log(X)] when X follows a logit normal

    # the function below is log(expit(v))
    log_v = lambda x : np.maximum(-np.log(1 + np.exp(-x)), -1e16) * (x > -1e2)\
                                + x * (x <= -1e2)

    # I believe that the above will avoid the numerical issues. If x is very small,
    # log(1 + e^(-x)) is basically -x, hence the two cases.
    # the maximum in the first term is taken so that when
    # -np.log(1 + np.exp(-x)) = -Inf, it really just returns -1e16;
    # apparently -Inf * 0.0 is NaN in python.

    e_log_v = get_e_fun_normal(lognorm_means, lognorm_infos, \
                            gh_loc, gh_weights, log_v)
    e_log_1mv = - lognorm_means + e_log_v
    return e_log_v, e_log_1mv

