import jax
import jax.numpy as np
import jax.scipy as sp

from bnpmodeling_runjingdev import stick_integration_lib

def assert_positive(x):
    # I happen to use this a lot ...
    # if negative, replace w nan's
    # resulting objective should be nan
    # and errors can be caught
    return np.where(x < 0, np.nan, x)

################
# define entropies
################
def multinom_entropy(e_z):
    # returns the entropy of the cluster belongings
    return -1 * np.sum(e_z * np.log(e_z + 1e-8))

def get_stick_breaking_entropy(stick_propn_mean, stick_propn_info,
                                gh_loc, gh_weights):
    # return the entropy of logitnormal distriibution on the sticks whose
    # logit has mean stick_propn_mean and information stick_propn_info
    # Integration is done on the real line with respect to the Lesbegue measure

    # integration is done numerical with Gauss Hermite quadrature.
    # gh_loc and gh_weights specifiy the location and weights of the
    # quadrature points

    # we seek E[log q(V)], where q is the density of a logit-normal, and
    # V ~ logit-normal. Let W := logit(V), so W ~ Normal. Hence,
    # E[log q(W)]; we can then decompose log q(x) into the terms of a normal
    # distribution and the jacobian term. The expectation of the normal term
    # evaluates to the normal entropy, and we add the jacobian term to it.
    # The jacobian term is 1/(x(1-x)), so we simply add -EV - E(1-V) to the normal
    # entropy.

    gh_weights = assert_positive(gh_weights)

    assert stick_propn_mean.shape == stick_propn_info.shape
    stick_propn_info = assert_positive(stick_propn_info)

    e_log_v, e_log_1mv =\
        stick_integration_lib.get_e_log_logitnormal(
            lognorm_means = stick_propn_mean,
            lognorm_infos = stick_propn_info,
            gh_loc = gh_loc,
            gh_weights = gh_weights)

    return np.sum(univariate_normal_entropy(stick_propn_info)) + \
                    np.sum(e_log_v + e_log_1mv)

def univariate_normal_entropy(info_obs):
    # np.sum(sp.stats.norm.entropy(scale=np.sqrt(var_obs)))
    return 0.5 * np.sum(-1 * np.log(info_obs) + 1 + np.log(2 * np.pi))

def dirichlet_entropy(alpha):
        
    # dimension is (.... x k)
    
    dirichlet_dim = alpha.shape[-1]
    sum_alpha = np.sum(alpha, axis=-1)
    log_beta = np.sum(sp.special.gammaln(alpha), axis=-1) - \
               sp.special.gammaln(sum_alpha)
    
    entropy = \
        log_beta - \
        (dirichlet_dim - sum_alpha) * sp.special.digamma(sum_alpha) - \
        np.sum((alpha - 1) * sp.special.digamma(alpha), axis=-1)
    
    return np.sum(entropy)


##############
# likelihoods
##############
def get_e_log_cluster_probabilities_from_e_log_stick(e_log_v, e_log_1mv):
    zeros_shape = e_log_v.shape[0:-1] + (1,)

    e_log_stick_remain = np.concatenate([np.zeros(zeros_shape), \
                                        np.cumsum(e_log_1mv, axis = -1)], axis = -1)
    e_log_new_stick = np.concatenate((e_log_v, np.zeros(zeros_shape)), axis = -1)

    return e_log_stick_remain + e_log_new_stick

def get_e_log_cluster_probabilities(stick_propn_mean, stick_propn_info,
                                        gh_loc, gh_weights):

    # the expected log mixture weights
    # stick_propn_mean is of shape ... x k_approx

    gh_weights = assert_positive(gh_weights)

    assert stick_propn_mean.shape == stick_propn_info.shape
    if len(stick_propn_mean.shape) == 1:
        stick_propn_mean = stick_propn_mean[None, :]
        stick_propn_info = stick_propn_info[None, :]
        squeeze = True

    stick_propn_info = assert_positive(stick_propn_info)

    e_log_v, e_log_1mv = \
        stick_integration_lib.get_e_log_logitnormal(
            lognorm_means = stick_propn_mean,
            lognorm_infos = stick_propn_info,
            gh_loc = gh_loc,
            gh_weights = gh_weights)

    e_log_cluster_probs = \
        get_e_log_cluster_probabilities_from_e_log_stick(e_log_v, e_log_1mv)
    if squeeze:
        return e_log_cluster_probs.squeeze()
    else:
        return e_log_cluster_probs

def get_e_logitnorm_dp_prior(stick_propn_mean, stick_propn_info, alpha,
                                gh_loc, gh_weights):
    # expected log prior for the stick breaking proportions under the
    # logitnormal variational distribution

    # integration is done numerical with Gauss Hermite quadrature.
    # gh_loc and gh_weights specifiy the location and weights of the
    # quadrature points

    gh_weights = assert_positive(gh_weights)

    assert stick_propn_mean.shape == stick_propn_info.shape
    stick_propn_info = assert_positive(stick_propn_info)

    e_log_v, e_log_1mv = \
        stick_integration_lib.get_e_log_logitnormal(
            lognorm_means = stick_propn_mean,
            lognorm_infos = stick_propn_info,
            gh_loc = gh_loc,
            gh_weights = gh_weights)

    return (alpha - 1) * np.sum(e_log_1mv)

##############
# some useful moments
##############

def get_e_beta(tau):
    # tau should have shape (..., 2). The last dimensions are the
    # beta parameters
    assert tau.shape[-1] == 2

    sum_alpha_beta = np.sum(tau, axis = -1)

    return tau[..., 0] / sum_alpha_beta

def get_e_log_beta(tau):
    # tau should have shape (..., 2). The last dimensions are the
    # beta parameters
    assert tau.shape[-1] == 2

    digamma_alpha = sp.special.digamma(tau[..., 0])
    digamma_beta = sp.special.digamma(tau[..., 1])

    digamma_alpha_beta = sp.special.digamma(np.sum(tau, axis = -1))

    return digamma_alpha - digamma_alpha_beta, digamma_beta - digamma_alpha_beta


def get_e_dirichlet(tau):
    # tau should have shape (..., k). The last dimensions are the 
    # dirichlet parameters
    
    digamma_sum = sp.special.digamma(np.sum(tau, axis = -1, keepdims=True))

    return tau - np.sum(tau, axis = -1, keepdims=True)


def get_e_log_dirichlet(tau):
    # tau should have shape (..., k). The last dimensions are the 
    # dirichlet parameters
    
    digamma_sum = sp.special.digamma(np.sum(tau, axis = -1, keepdims=True))

    return sp.special.digamma(tau) - digamma_sum
