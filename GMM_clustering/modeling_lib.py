import sys
sys.path.insert(0, './../../LinearResponseVariationalBayes.py')

import LinearResponseVariationalBayes as vb
import LinearResponseVariationalBayes.ExponentialFamilies as ef

import autograd
import autograd.numpy as np
import autograd.scipy as sp

import scipy as osp
################
# define entropies

def multinom_entropy(e_z):
    return -1 * np.sum(e_z * np.log(e_z + 1e-8))

def get_logitnorm_stick_entropy(v_stick_mean, v_stick_info, gh_loc, gh_weights):
    # the entropy of the logitnormal v-sticks
    # we seek E[log q(V)], where q is the density of a logit-normal, and
    # V ~ logit-normal. Let W := logit(V), so W ~ Normal. Hence,
    # E[log q(W)]; we can then decompose log q(x) into the terms of a normal
    # distribution and the jacobian term. The expectation of the normal term
    # evaluates to the normal entropy, and we add the jacobian term to it.
    # The jacobian term is 1/(x(1-x)), so we simply add -EV - E(1-V) to the normal
    # entropy.

    assert np.all(gh_weights > 0)

    assert len(v_stick_mean) == len(v_stick_info)
    assert np.all(v_stick_info) > 0

    e_log_v, e_log_1mv =\
        ef.get_e_log_logitnormal(
            lognorm_means = v_stick_mean,
            lognorm_infos = v_stick_info,
            gh_loc = gh_loc,
            gh_weights = gh_weights)
    return np.sum(ef.univariate_normal_entropy(v_stick_info)) + \
                    np.sum(e_log_v + e_log_1mv)

################
# define priors

def get_e_logitnorm_dp_prior(v_stick_mean, v_stick_info, alpha,
                                gh_loc, gh_weights):

    assert np.all(gh_weights > 0)

    assert len(v_stick_mean) == len(v_stick_info)
    assert np.all(v_stick_info) > 0

    e_log_v, e_log_1mv = \
        ef.get_e_log_logitnormal(
            lognorm_means = v_stick_mean,
            lognorm_infos = v_stick_info,
            gh_loc = gh_loc,
            gh_weights = gh_weights)

    return (alpha - 1) * np.sum(e_log_1mv)


##############
# likelihoods



def get_mixture_weights(stick_lengths):
    # computes mixture weights from stick lengths
    stick_lengths_1m = 1 - stick_lengths
    stick_remain = np.concatenate((np.array([1]),
                                   cumprod_through_log(stick_lengths_1m)))
    stick_add = np.concatenate((stick_lengths, np.array([1])))

    return stick_remain * stick_add


def get_e_log_cluster_probabilities(v_stick_mean, v_stick_info,
                                        gh_loc, gh_weights):
    assert np.all(gh_weights > 0)

    assert len(v_stick_mean) == len(v_stick_info)
    assert np.all(v_stick_info) > 0

    e_log_v, e_log_1mv = \
        ef.get_e_log_logitnormal(
            lognorm_means = v_stick_mean,
            lognorm_infos = v_stick_info,
            gh_loc = gh_loc,
            gh_weights = gh_weights)

    e_log_stick_remain = np.concatenate([np.array([0.]), np.cumsum(e_log_1mv)])
    e_log_new_stick = np.concatenate((e_log_v, np.array([0])))

    return e_log_stick_remain + e_log_new_stick


def loglik_ind(v_stick_mean, v_stick_info, e_z, gh_loc, gh_weights,
                    use_logitnormal_sticks = True):

    assert np.all(gh_weights > 0)

    assert len(v_stick_mean) == len(v_stick_info)
    assert np.all(v_stick_info) > 0


    # expected log likelihood of all indicators for all n observations
    e_log_cluster_probs = \
        get_e_log_cluster_probabilities(v_stick_mean, v_stick_info,
                                        gh_loc, gh_weights)

    return np.sum(e_z * e_log_cluster_probs)


##########################
# Functions to compute the expected number of clusters from vb_params
# the reason its a bit hacky is bc we need to make sure its differentiable
# the way we were computing cluster weights from stick lengths before
# required indexing. It also needs to take in an array of stick lengths,
# since we'll be sampling to compute the expectation.
##########################
