import LinearResponseVariationalBayes as vb
import LinearResponseVariationalBayes.ExponentialFamilies as ef

import autograd
import autograd.numpy as np
import autograd.scipy as sp

import scipy as osp

# for the moment, this was copied over from the sensitivity_to_stick_breaking_in_BNP repo

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

    assert np.all(gh_weights > 0)

    assert stick_propn_mean.shape == stick_propn_info.shape
    assert np.all(stick_propn_info) > 0

    e_log_v, e_log_1mv =\
        ef.get_e_log_logitnormal(
            lognorm_means = stick_propn_mean,
            lognorm_infos = stick_propn_info,
            gh_loc = gh_loc,
            gh_weights = gh_weights)

    return np.sum(ef.univariate_normal_entropy(stick_propn_info)) + \
                    np.sum(e_log_v + e_log_1mv)


def get_e_logitnorm_dp_prior(stick_propn_mean, stick_propn_info, alpha,
                                gh_loc, gh_weights):
    # expected log prior for the stick breaking proportions under the
    # logitnormal variational distribution

    # integration is done numerical with Gauss Hermite quadrature.
    # gh_loc and gh_weights specifiy the location and weights of the
    # quadrature points

    assert np.all(gh_weights > 0)

    assert stick_propn_mean.shape == stick_propn_info.shape
    assert np.all(stick_propn_info) > 0

    e_log_v, e_log_1mv = \
        ef.get_e_log_logitnormal(
            lognorm_means = stick_propn_mean,
            lognorm_infos = stick_propn_info,
            gh_loc = gh_loc,
            gh_weights = gh_weights)

    return (alpha - 1) * np.sum(e_log_1mv)

def get_e_log_cluster_probabilities(stick_propn_mean, stick_propn_info,
                                        gh_loc, gh_weights):

    # the expected log mixture weights

    # TODO: this only works with 2D stick_propn_means at the moment.
    # each row is an observation, with logitnormal
    # stick breaking perameters in each column
    assert len(stick_propn_mean.shape) == 2
    n_obs = stick_propn_mean.shape[0]

    assert np.all(gh_weights > 0)

    assert stick_propn_mean.shape == stick_propn_info.shape
    assert np.all(stick_propn_info) > 0

    e_log_v, e_log_1mv = \
        ef.get_e_log_logitnormal(
            lognorm_means = stick_propn_mean,
            lognorm_infos = stick_propn_info,
            gh_loc = gh_loc,
            gh_weights = gh_weights)

    e_log_stick_remain = np.concatenate([np.zeros((n_obs, 1)),
                                    np.cumsum(e_log_1mv, axis = 1)], axis = 1)
    e_log_new_stick = np.concatenate([e_log_v, np.zeros((n_obs, 1))], axis = 1)

    return e_log_stick_remain + e_log_new_stick

def get_e_log_beta(tau):
    # tau should have shape (..., 2). The last dimensions are the
    # beta parameters
    assert tau.shape[-1] == 2

    digamma_alpha = sp.special.digamma(tau[..., 0])
    digamma_beta = sp.special.digamma(tau[..., 1])

    digamma_alpha_beta = sp.special.digamma(np.sum(tau, axis = -1))

    return digamma_alpha - digamma_alpha_beta, digamma_beta - digamma_alpha_beta
