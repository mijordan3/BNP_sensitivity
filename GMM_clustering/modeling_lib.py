import sys
sys.path.insert(0, './../../LinearResponseVariationalBayes.py')

import LinearResponseVariationalBayes as vb
import LinearResponseVariationalBayes.ExponentialFamilies as ef

import autograd
import autograd.numpy as np
import autograd.scipy as sp

################
# define entropies

def multinom_entropy(e_z):
    return -1 * np.sum(e_z * np.log(e_z + 1e-8))

def get_stick_entropy(vb_params):
    # the entropy of the logitnormal v-sticks
    # we seek E[log q(V)], where q is the density of a logit-normal, and
    # V ~ logit-normal. Let W := logit(V), so W ~ Normal. Hence,
    # E[log q(W)]; we can then decompose log q(x) into the terms of a normal
    # distribution and the jacobian term. The expectation of the normal term
    # evaluates to the normal entropy, and we add the jacobian term to it.
    # The jacobian term is 1/(x(1-x)), so we simply add -EV - E(1-V) to the normal
    # entropy.

    if vb_params.use_logitnormal_sticks:
        e_log_v, e_log_1mv =\
            ef.get_e_log_logitnormal(
                lognorm_means = vb_params['global']['v_sticks']['mean'].get(),
                lognorm_infos = vb_params['global']['v_sticks']['info'].get(),
                gh_loc = vb_params.gh_loc,
                gh_weights = vb_params.gh_weights)
        # in this case, the .entropy() returns a UNV entropy
        return np.sum(vb_params['global']['v_sticks'].entropy()) + \
                        np.sum(e_log_v + e_log_1mv)
    else:
        # in this case, the .entropy() returns a beta entropy
        return np.sum(vb_params['global']['v_sticks'].entropy())

################
# define priors

def get_dp_prior(vb_params, prior_params):
    alpha = prior_params['alpha'].get()
    if vb_params.use_logitnormal_sticks:
        e_log_v, e_log_1mv = \
            ef.get_e_log_logitnormal(
                lognorm_means = vb_params['global']['v_sticks']['mean'].get(),
                lognorm_infos = vb_params['global']['v_sticks']['info'].get(),
                gh_loc = vb_params.gh_loc,
                gh_weights = vb_params.gh_weights)
    else:
        e_log_1mv = vb_params['global']['v_sticks'].e_log()[1, :] # E[log 1 - v]

    return (alpha - 1) * np.sum(e_log_1mv)


##############
# likelihoods

# Autograd doesn't work with the original cumprod.
def cumprod_through_log(x, axis = None):
    return np.exp(np.cumsum(np.log(x), axis = axis))


def get_mixture_weights(stick_lengths):
    # computes mixture weights from stick lengths
    stick_lengths_1m = 1 - stick_lengths
    stick_remain = np.concatenate((np.array([1]),
                                   cumprod_through_log(stick_lengths_1m)))
    stick_add = np.concatenate((stick_lengths, np.array([1])))

    return stick_remain * stick_add


def get_e_log_cluster_probabilities(vb_params):
    if vb_params.use_logitnormal_sticks:
        e_log_v, e_log_1mv = \
            ef.get_e_log_logitnormal(
                lognorm_means = vb_params['global']['v_sticks']['mean'].get(),
                lognorm_infos = vb_params['global']['v_sticks']['info'].get(),
                gh_loc = vb_params.gh_loc,
                gh_weights = vb_params.gh_weights)
    else:
        e_log_sticks = vb_params['global']['v_sticks'].e_log()
        e_log_v = e_log_sticks[0, :] # E[log v]
        e_log_1mv = e_log_sticks[1, :] # E[log 1 - v]

    e_log_stick_remain = np.concatenate([np.array([0.]), np.cumsum(e_log_1mv)])
    e_log_new_stick = np.concatenate((e_log_v, np.array([0])))

    return e_log_stick_remain + e_log_new_stick


def loglik_ind(vb_params, e_z):
    # expected log likelihood of all indicators for all n observations
    e_log_cluster_probs = get_e_log_cluster_probabilities(vb_params)
    #return np.sum(vb_params['e_z'].get() * e_log_cluster_probs)
    return np.sum(e_z * e_log_cluster_probs)


##########################
# Functions to compute the expected number of clusters from vb_params
# the reason its a bit hacky is bc we need to make sure its differentiable
# the way we were computing cluster weights from stick lengths before
# required indexing. It also needs to take in an array of stick lengths,
# since we'll be sampling to compute the expectation.
##########################
def get_mixture_weights_array(stick_lengths):
    # computes mixture weights from an array of stick lengths
    k_approx = np.shape(stick_lengths)[1]
    n_sticks = np.shape(stick_lengths)[0]

    stick_lengths_1m = 1 - stick_lengths
    stick_remain = np.hstack((np.ones((n_sticks, 1)),
                              cumprod_through_log(stick_lengths_1m, axis = 1)))
    stick_add = np.hstack((stick_lengths,
                                np.ones((n_sticks, 1))))

    return stick_remain * stick_add


def get_kth_weight_from_sticks(stick_lengths, k):
    assert len(np.shape(stick_lengths)) == 2
    # stick lengths is a matrix of shape (n_samples, k_approx - 1)

    assert k <= (np.shape(stick_lengths)[1] + 1)

    if k == 0:
        stick_remaining = 1.
    else:
        stick_remaining = np.prod(1 - stick_lengths[:, 0:k], axis = 1)

    if k == (np.shape(stick_lengths)[1]):
        stick_length = 1
    else:
        stick_length = stick_lengths[:, k]

    return (stick_remaining * stick_length)


def get_e_number_clusters_from_logit_sticks(mu, sigma, n_obs,
                                            samples = 100000,
                                            unv_norm_samples = None):

    # get logitnormal params
    # mu = model.vb_params['global']['v_sticks']['mean'].get()
    # sigma = model.vb_params['global']['v_sticks']['info'].get()
    k_approx = len(mu)

    # sample from univariate normal
    # TODO: keep these draws fixed to reduce simulation noise --
    # "Rao-Blackwellize" this statistic.
    if unv_norm_samples is None:
        unv_norm_samples = np.random.normal(0, 1, size = (samples, k_approx))

    # sample sticks from variational distribution
    stick_samples = sp.special.expit(unv_norm_samples / np.sqrt(sigma) + mu)

    # get posterior weights
    weight_samples = get_mixture_weights_array(stick_samples)

    return np.mean(np.sum(1 - (1 - weight_samples)**n_obs, axis = 1))


def get_e_number_clusters_from_ez(e_z):
    # computes the expected number of clusters from
    # the e_z in the variational distribution
    k = np.shape(e_z)[1]
    return k - np.sum(np.prod(1 - e_z, axis = 0))

def sample_clusters_from_ez_and_unif_sample(e_z_cumsum, unif_sample):

    n_obs = e_z_cumsum.shape[0]

    assert len(unif_sample) == n_obs

    # get which cluster the sample belongs to
    z_ind = (e_z_cumsum > unif_sample[:, None]).argmax(1)

    # get one hot encoding
    # is there a way to vectorize this?
    z_sample = np.zeros(e_z_cumsum.shape)
    z_sample[np.arange(n_obs), z_ind] = 1

    return z_sample


def get_e_num_large_clusters_from_ez(e_z,
                                    threshold = 0.0,
                                    n_samples = 100000,
                                    unif_samples = None):

    n_obs = e_z.shape[0]

    # draw uniform samples
    if unif_samples is None:
        unif_samples = np.random.random((n_obs, n_samples))

    else:
        assert len(unif_samples.shape[0]) == n_obs

    e_z_cumsum = np.cumsum(e_z, axis = 1)
    num_heavy_clusters_vec = np.zeros(n_samples)
    for i in range(n_samples):
        z_sample = sample_clusters_from_ez_and_unif_sample(e_z_cumsum, unif_samples[:, i])

        num_heavy_clusters_vec[i] = np.sum(np.mean(z_sample, axis = 0) > threshold)

    return np.mean(num_heavy_clusters_vec)






# def get_e_number_clusters_from_logit_sticks_diffble(vb_params, samples = 10000):
#     # get logitnormal params
#     mu = vb_params['global']['v_sticks']['mean'].get()
#     sigma = vb_params['global']['v_sticks']['info'].get()
#     n_sticks = len(mu)
#     #n_obs = vb_params['e_z'].shape()[0]
#
#     # sample from univariate normal
#     unv_norm_samples = np.random.normal(0, 1, size = (samples, n_sticks))
#
#     # sample sticks from variational distribution
#     stick_samples = sp.special.expit(unv_norm_samples / np.sqrt(sigma) + mu)
#
#     e_num_clusters = 0
#     for k in range(n_sticks):
#         weight_samples_k = get_kth_weight_from_sticks(stick_samples, k)
#
#         e_num_clusters = e_num_clusters + np.mean(1 - (1 - weight_samples_k)**n_obs)
#
#     return e_num_clusters
