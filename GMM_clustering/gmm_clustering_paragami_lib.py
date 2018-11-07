import sys
sys.path.insert(0, './../../LinearResponseVariationalBayes.py')

import LinearResponseVariationalBayes as vb
import LinearResponseVariationalBayes.ExponentialFamilies as ef
import LinearResponseVariationalBayes.SparseObjectives as obj_lib
import LinearResponseVariationalBayes.OptimizationUtils as opt_lib

import autograd
import autograd.numpy as np
import autograd.scipy as sp

import scipy as osp

import warnings

from scipy import optimize
from scipy import linalg
from scipy import sparse

from sklearn.cluster import KMeans

from copy import deepcopy

from datetime import datetime
import time

import modeling_lib_paragami as modeling_lib
import functional_sensitivity_lib as fun_sens_lib

import json
import json_tricks

from numpy.polynomial.hermite import hermgauss

sys.path.append('./../../../paragami/')
import paragami

##########################
# Set up vb parameters
##########################

def get_vb_params_paragami_object(dim, k_approx, n_obs,
                                    use_logitnormal_sticks = True):

    vb_params_paragami = paragami.PatternDict()

    # cluster centroids
    vb_params_paragami['centroids'] = \
        paragami.NumericArrayPattern(shape=(dim, k_approx))

    # BNP sticks
    if use_logitnormal_sticks:
        # variational distribution for each stick is logitnormal
        vb_params_paragami['v_stick_mean'] = \
            paragami.NumericArrayPattern(shape = (k_approx - 1,))
        vb_params_paragami['v_stick_info'] = \
            paragami.NumericArrayPattern(shape = (k_approx - 1,), lb = 1e-4)
    else:
        # else its a beta
        vb_params_paragami['v_sticks_beta'] = \
            paragami.NumericArrayPattern(shape=(2, k_approx - 1), lb = 0.)

    # cluster covariances
    vb_params_paragami['gamma'] = \
        paragami.pattern_containers.PatternArray(shape = (k_approx, ), \
                                    base_pattern = paragami.PSDSymmetricMatrixPattern(size=dim))

    vb_params_dict = vb_params_paragami.random()

    return vb_params_dict, vb_params_paragami

def get_vb_params_from_dict(vb_params_dict):
    # VB parameters
    v_stick_mean = vb_params_dict['v_stick_mean']
    v_stick_info = vb_params_dict['v_stick_info']

    centroids = vb_params_dict['centroids']
    gamma = vb_params_dict['gamma']
    assert np.shape(centroids)[1] == np.shape(gamma)[0]
    assert np.shape(centroids)[0] == np.shape(gamma)[1]
    assert np.shape(centroids)[0] == np.shape(gamma)[2]

    return v_stick_mean, v_stick_info, centroids, gamma

# Set the gh_log and gh_weights attributes of the vb_params object.
def set_gauss_hermite_points(vb_params, gh_deg):
    gh_loc, gh_weights = hermgauss(gh_deg)
    vb_params.gh_loc = gh_loc
    vb_params.gh_weights = gh_weights

##########################
# Set up prior parameters
##########################
def get_default_prior_params(dim):
    prior_params_dict = dict()
    prior_params_paragami = paragami.PatternDict()

    # DP prior parameter
    prior_params_dict['alpha'] = np.array([4.0])
    prior_params_paragami['alpha'] = \
        paragami.NumericArrayPattern(shape=(1, ), lb = 0.0)

    # prior on the centroids
    prior_params_dict['prior_centroid_mean'] = np.array([0.0])
    prior_params_paragami['prior_centroid_mean'] = \
        paragami.NumericArrayPattern(shape=(1, ))

    prior_params_dict['prior_centroid_info'] = np.array([0.1])
    prior_params_paragami['prior_centroid_info'] = \
        paragami.NumericArrayPattern(shape=(1, ), lb = 0.0)

    # prior on the variance
    prior_params_dict['prior_gamma_df'] = np.array([8.0])
    prior_params_paragami['prior_gamma_df'] = \
        paragami.NumericArrayPattern(shape=(1, ), lb = 0.0)

    prior_params_dict['prior_gamma_inv_scale'] = 0.62 * np.eye(dim)
    prior_params_paragami['prior_gamma_inv_scale'] = \
        paragami.PSDSymmetricMatrixPattern(size=dim)

    return prior_params_dict, prior_params_paragami

##########################
# Expected prior terms
##########################

def get_e_centroid_prior(centroids, prior_mean, prior_info):

    assert prior_info > 0

    beta_base_prior = ef.uvn_prior(
        prior_mean = prior_mean,
        prior_info = prior_info,
        e_obs = centroids.flatten(),
        var_obs = np.array([0.]))

    return np.sum(beta_base_prior)

def get_e_log_wishart_prior(gamma, df, V_inv):

    dim = V_inv.shape[0]

    assert np.shape(gamma)[1] == dim

    tr_V_inv_gamma = np.einsum('ij, kji -> k', V_inv, gamma)

    s, logdet = np.linalg.slogdet(gamma)
    assert np.all(s > 0)

    return np.sum((df - dim - 1) / 2 * logdet - 0.5 * tr_V_inv_gamma)

# Get a vector of expected functions of the logit sticks.
# You can use this to define proportional functional perturbations to the
# logit stick distributions.
# The function func should take arguments in the logit stick space, i.e.
# logit_stick = log(stick / (1 - stick)).
def get_e_func_logit_stick_vec(vb_params_dict, gh_loc, gh_weights, func):
    v_stick_mean = vb_params_dict['v_stick_mean']
    v_stick_info = vb_params_dict['v_stick_info']

    # print('DEBUG: 0th lognorm mean: ', v_stick_mean[0])
    e_phi = np.array([
        ef.get_e_fun_normal(
            v_stick_mean[k], v_stick_info[k], \
            gh_loc, gh_weights, func)
        for k in range(len(v_stick_mean))
    ])

    return e_phi

def get_e_log_prior(v_stick_mean, v_stick_info, centroids, gamma,
                        prior_params_dict,
                        gh_loc, gh_weights,
                        use_logitnormal_sticks = True):

    # prior parameters
    # dp prior
    alpha = prior_params_dict['alpha']

    # wishart prior
    df = prior_params_dict['prior_gamma_df']
    V_inv = prior_params_dict['prior_gamma_inv_scale']

    # centroid priors
    prior_mean = prior_params_dict['prior_centroid_mean']
    prior_info = prior_params_dict['prior_centroid_info']

    if use_logitnormal_sticks:
        dp_prior = \
            modeling_lib.get_e_logitnorm_dp_prior(v_stick_mean, v_stick_info,
                                                alpha, gh_loc, gh_weights)
    else:
        raise NotImplementedError()

    e_gamma_prior = get_e_log_wishart_prior(gamma, df, V_inv)
    e_centroid_prior = get_e_centroid_prior(centroids, prior_mean, prior_info)

    return np.squeeze(e_gamma_prior + e_centroid_prior + dp_prior)

##########################
# Entropy
##########################
def get_entropy(v_stick_mean, v_stick_info, e_z, gh_loc, gh_weights,
                    use_logitnormal_sticks = True):

    z_entropy = modeling_lib.multinom_entropy(e_z)
    if use_logitnormal_sticks:
        stick_entropy = \
            modeling_lib.get_logitnorm_stick_entropy(v_stick_mean, v_stick_info,
                                    gh_loc, gh_weights)
    else:
        raise NotImplementedError()

    return z_entropy + stick_entropy

##########################
# Likelihood term
##########################
def get_loglik_obs_by_nk(y, centroids, gamma):
    # returns a n x k matrix whose nkth entry is
    # the likelihood for the nth observation
    # belonging to the kth cluster

    dim = np.shape(y)[1]

    assert np.shape(y)[1] == np.shape(centroids)[0]
    assert np.shape(gamma)[0] == np.shape(centroids)[1]
    assert np.shape(gamma)[1] == np.shape(centroids)[0]

    data2_term = np.einsum('ni, kij, nj -> nk', y, gamma, y)
    cross_term = np.einsum('ni, kij, jk -> nk', y, gamma, centroids)
    centroid2_term = np.einsum('ik, kij, jk -> k', centroids, gamma, centroids)

    squared_term = data2_term - 2 * cross_term + \
                    np.expand_dims(centroid2_term, axis = 0)

    return - 0.5 * squared_term + 0.5 * np.linalg.slogdet(gamma)[1][None, :]

##########################
# Optimization over e_z
##########################

def get_z_nat_params(y, v_stick_mean, v_stick_info, centroids, gamma,
                        gh_loc, gh_weights,
                        use_bnp_prior = True,
                        return_loglik_obs_by_nk = False):

    # get likelihood term
    loglik_obs_by_nk = get_loglik_obs_by_nk(y, centroids, gamma)

    # get weight term
    if use_bnp_prior:
        e_log_cluster_probs = \
            modeling_lib.get_e_log_cluster_probabilities(
                            v_stick_mean, v_stick_info,
                            gh_loc, gh_weights)
    else:
        e_log_cluster_probs = 0.

    z_nat_param = loglik_obs_by_nk + e_log_cluster_probs

    if return_loglik_obs_by_nk:
        return z_nat_param, loglik_obs_by_nk
    else:
        return z_nat_param

def get_optimal_z(y, v_stick_mean, v_stick_info, centroids, gamma,
                    gh_loc, gh_weights,
                    use_bnp_prior = True,
                    return_loglik_obs_by_nk = False):

    _z_nat_param = \
        get_z_nat_params(y, v_stick_mean, v_stick_info, centroids, gamma,
                                    gh_loc, gh_weights,
                                    use_bnp_prior,
                                    return_loglik_obs_by_nk)

    if return_loglik_obs_by_nk:
        loglik_obs_by_nk = _z_nat_param[1]
        z_nat_param = _z_nat_param[0]

    else:
        z_nat_param = _z_nat_param

    log_const = sp.misc.logsumexp(z_nat_param, axis=1)
    e_z = np.exp(z_nat_param - log_const[:, None])

    if return_loglik_obs_by_nk:
        return e_z, loglik_obs_by_nk
    else:
        return e_z

def get_kl(y, vb_params_dict, prior_params_dict,
                    gh_loc, gh_weights,
                    data_weights = None,
                    use_bnp_prior = True,
                    use_logitnormal_sticks = True):

    v_stick_mean, v_stick_info, centroids, gamma = \
        get_vb_params_from_dict(vb_params_dict)

    # get optimal cluster belongings
    e_z, loglik_obs_by_nk = \
            get_optimal_z(y, v_stick_mean, v_stick_info, centroids, gamma,
                            gh_loc, gh_weights, 
                            return_loglik_obs_by_nk = True)

    # weight data if necessary, and get likelihood of y
    if data_weights is not None:
        assert np.shape(data_weights)[0] == n_obs, \
                    'data weights need to be n_obs by 1'
        assert np.shape(data_weights)[1] == 1, \
                    'data weights need to be n_obs by 1'
        e_loglik_obs = np.sum(data_weights * e_z * loglik_obs_by_nk)
    else:
        e_loglik_obs = np.sum(e_z * loglik_obs_by_nk)

    # likelihood of z
    if use_bnp_prior:
        e_loglik_ind = modeling_lib.loglik_ind(v_stick_mean, v_stick_info, e_z,
                            gh_loc, gh_weights,
                            use_logitnormal_sticks)
    else:
        e_loglik_ind = 0.

    e_loglik = e_loglik_ind + e_loglik_obs

    if not np.isfinite(e_loglik):
        print('gamma', vb_params_dict['gamma'].get())
        print('det gamma', np.linalg.slogdet(
            vb_params_dict['gamma'])[1])
        print('cluster weights', np.sum(e_z, axis = 0))

    assert(np.isfinite(e_loglik))

    entropy = np.squeeze(get_entropy(v_stick_mean, v_stick_info, e_z,
                                        gh_loc, gh_weights,
                                        use_logitnormal_sticks))
    assert(np.isfinite(entropy))

    e_log_prior = get_e_log_prior(v_stick_mean, v_stick_info, centroids, gamma,
                            prior_params_dict,
                            gh_loc, gh_weights,
                            use_logitnormal_sticks)

    assert(np.isfinite(e_log_prior))

    elbo = e_log_prior + entropy + e_loglik

    return -1 * elbo





def precondition_and_optimize(
    objective, init_x, kl_hessian=None, ev_min=1, ev_max=1e5):

    opt_lib.set_objective_preconditioner(
        objective=objective,
        free_par=init_x,
        hessian=kl_hessian,
        ev_min=ev_min, ev_max=ev_max)

    opt_x, vb_opt = opt_lib.minimize_objective_trust_ncg(
        objective=objective,
        init_x=init_x,
        precondition=True,
        init_logger=False)

    return opt_x, vb_opt

##########################
# the model class
##########################
class DPGaussianMixture(object):
    def __init__(self, y, k_approx, prior_param_dict, gh_deg, \
                    use_bnp_prior = True,
                    use_logitnormal_sticks = False):
        # y is the observed data
        # x is the matrix of 'covariates', in our case, the spline bases
        # k_approx is the number of clusters
        # prior_params is the class in which the prior parameters are stored
        # gh_deg is the parameter used for logitnormal integration

        self.y = y

        self.dim = y.shape[1]
        self.k_approx = k_approx
        self.n_obs = y.shape[0]

        self.gh_deg = gh_deg
        self.gh_loc, self.gh_weights = hermgauss(gh_deg)

        self.use_bnp_prior = use_bnp_prior
        self.use_logitnormal_sticks = use_logitnormal_sticks

        self.e_z = np.full((self.n_obs, self.k_approx), float('nan'))

        self.use_weights = False
        self.weights = vb.VectorParam('w', size=self.n_obs)
        self.weights.set(np.ones((self.n_obs, 1)))

        self.vb_params_dict, self.vb_params_paragami = \
            get_vb_param_paragami_object(self.dim, self.k_approx, self.n_obs,
                                        use_logitnormal_sticks = True)

        self.prior_params_dict = deepcopy(prior_param_dict)

        self.set_optimal_z()

        # Make a set of parameters for optimization.  Note that the
        # parameters are passed by reference, so updating global_vb_params
        # updates vb_params.

        # TODO: There is no need to keep track of "global" VB params.
        # e_z should not be a part of the same data structure.
        # self.global_vb_params = vb.ModelParamsDict()
        # self.global_vb_params.push_param(self.vb_params['global']['centroids'])
        # self.global_vb_params.push_param(self.vb_params['global']['gamma'])
        # self.global_vb_params.push_param(self.vb_params['global']['v_sticks'])

        # The KL divergence:
        # self.objective = \
        #     obj_lib.Objective(self.global_vb_params, self.set_z_get_kl)
        # self.objective.preconditioning = False
        # self.objective.logger.callback = self.display_optimization_status
        #
        # # The prior:
        # self.prior_objective = \
        #     obj_lib.TwoParameterObjective(
        #         self.prior_params, self.global_vb_params, self.set_z_get_kl)
        # self.get_kl_prior_cross_hess = self.prior_objective.fun_free_hessian12
        #
        # # To data:
        # # self.per_gene_kl_obj = obj_lib.Objective(
        # #     self.global_vb_params, self.get_per_gene_kl)
        # # self.get_data_cross_hess = self.per_gene_kl_obj.fun_free_jacobian
        # self.data_objective = \
        #     obj_lib.TwoParameterObjective(
        #         self.global_vb_params, self.weights, self.set_z_get_kl)

    # def __deepcopy__(self, memo):
    #     raise NotImplementedError(
    #         'deepcopy is behaving strangely with the model class')
    #
    # def get_data_cross_hess(self, free_par):
    #     use_weights_cache = self.use_weights
    #     self.use_weights = True
    #     data_cross_hess = self.data_objective.fun_free_hessian12(
    #         free_par, self.weights.get_free())
    #     self.use_weights = use_weights_cache
    #     return data_cross_hess
    #
    # def __str__(self):
    #     b = self.vb_params['global']['b'].e()
    #     return '\n'.join([
    #         'Post sd: {}'.format(1 / np.sqrt(
    #             self.vb_params['global']['gamma'].get())),
    #         str(self.vb_params['global']['beta']),
    #         'b mean = {} sd = {} '.format(np.mean(b), np.std(b)),
    #         'Z totals: {}'.format(np.sum(self.e_z, axis=0)) ])
    #
    # def set_random(self):
    #     self.vb_params['global'].set_free(np.random.random(
    #         self.vb_params['global'].free_size()))
    #     self.set_optimal_z()
    #
    def set_from_global_free_par(self, free_par):
        self.vb_params_dict = self.vb_params_paragami.fold(free_par, free = True)

        return self.set_optimal_z(return_loglik_obs_by_nk = True)

    def get_z_nat_params(self, return_loglik_obs_by_nk = False):
        loglik_obs_by_nk = get_loglik_obs_by_nk(
            self.y, self.vb_params_dict)

        if self.use_bnp_prior:
            e_log_cluster_probs = \
                modeling_lib.get_e_log_cluster_probabilities(self.vb_params_dict,
                                self.gh_loc, self.gh_weights,
                                self.use_logitnormal_sticks)
        else:
            e_log_cluster_probs = 0

        z_nat_param = loglik_obs_by_nk + e_log_cluster_probs

        if return_loglik_obs_by_nk:
            return z_nat_param, loglik_obs_by_nk
        else:
            return z_nat_param

    def set_optimal_z(self, return_loglik_obs_by_nk = False):
        if return_loglik_obs_by_nk:
            z_nat_param, loglik_obs_by_nk = \
                    self.get_z_nat_params(return_loglik_obs_by_nk)
        else:
            z_nat_param = \
                    self.get_z_nat_params(return_loglik_obs_by_nk)

        log_const = sp.misc.logsumexp(z_nat_param, axis=1)
        #self.vb_params['e_z'].set(np.exp(z_nat_param - log_const[:, None]))
        self.e_z = np.exp(z_nat_param - log_const[:, None])

        if return_loglik_obs_by_nk:
            return loglik_obs_by_nk

    # Separate out for debugging.
    def get_e_log_prior(self):
        return np.squeeze(
            get_e_log_prior(self.vb_params_dict, self.prior_params_dict,
                self.gh_loc, self.gh_weights, self.use_logitnormal_sticks))

    def get_kl(self):
        # ...with the current value of z.
        return self.get_kl_utility(set_z=False)

    def set_z_get_kl(self):
        return self.get_kl_utility(set_z=True)

    def get_kl_utility(self, set_z):
        if set_z:
            loglik_obs_by_nk = self.set_optimal_z(
                return_loglik_obs_by_nk = True)
        else:
            loglik_obs_by_nk = get_loglik_obs_by_nk(
                self.y, self.vb_params_dict)

        if self.use_weights:
            #assert np.shape(self.weights) == (self.n_obs, 1)
            weights = np.expand_dims(self.weights.get(), 1)
            e_loglik_obs = np.sum(weights * self.e_z * loglik_obs_by_nk)
        else:
            e_loglik_obs = np.sum(self.e_z * loglik_obs_by_nk)

        if self.use_bnp_prior:
            e_loglik_ind = modeling_lib.loglik_ind(self.vb_params_dict, self.e_z,
                                self.gh_loc,
                                self.gh_weights,
                                self.use_logitnormal_sticks)
        else:
            e_loglik_ind = 0.

        e_loglik = e_loglik_ind + e_loglik_obs

        if not np.isfinite(e_loglik):
            print('gamma', self.vb_params['global']['gamma'].get())
            print('det gamma', np.linalg.slogdet(
                self.vb_params['global']['gamma'].get())[1])
            print('cluster weights', np.sum(self.e_z, axis = 0))

        assert(np.isfinite(e_loglik))

        entropy = np.squeeze(get_entropy(self.vb_params_dict, self.e_z,
                                            self.gh_loc,
                                            self.gh_weights,
                                            self.use_logitnormal_sticks))
        assert(np.isfinite(entropy))

        e_log_prior = self.get_e_log_prior()
        assert(np.isfinite(e_log_prior))

        # print(self.vb_params['global']['gamma'].get())
        elbo = e_log_prior + entropy + e_loglik
        return -1 * elbo

    def get_e_cluster_probabilities(self):
        if self.use_logitnormal_sticks:
            e_v = ef.get_e_logitnormal(
                v_stick_mean = self.vb_params_dict['v_stick_mean'],
                v_stick_info = self.vb_params_dict['v_stick_info'],
                gh_loc = self.gh_loc,
                gh_weights = self.gh_weights)
        else:
            raise NotImplementedError()
            # e_sticks = self.vb_params['global']['v_sticks'].e()
            # e_v = e_sticks[0, :]
            # e_1mv = e_sticks[1, :]

        return modeling_lib.get_mixture_weights(e_v)

    ########################
    # Sensitivity functions.

    def get_per_gene_kl(self):
        loglik_obs_by_nk = self.set_optimal_z(return_loglik_obs_by_nk = True)
        #e_z = self.vb_params['e_z'].get()
        return -1 * np.sum(self.e_z * loglik_obs_by_nk, axis=1)

    ########################
    # Optimization functions follow.

    def get_rmse(self):
        # This can be a callback in optimization as a sanity check.

        #e_z = self.vb_params['e_z'].get()
        centroids = self.global_vb_params['centroids'].get()

        post_means = np.dot(self.e_z, centroids.T)

        resid = self.y - post_means
        return np.sqrt(np.sum(resid ** 2))

    def get_z_allocation(self):
        self.set_optimal_z()
        #return np.sum(self.vb_params['e_z'].get(), axis=0)
        return np.sum(self.e_z, axis=0)

    # A callback for use with SparseObjectives:Objective.
    def display_optimization_status(self, logger):
        # Note that the logger gets passed the x that was given to the
        # function, /not/ the value before multiplication by the preconditioner.
        free_param = logger.x
        self.set_from_global_free_par(free_param)
        print('Iter: {}\t RMSE: {}\t Objective: {}'.format(
                logger.iter,  self.get_rmse(), logger.value))

    # Cluster the individual fits and initialize.  The z matrix will be at
    # least z_init_eps.
    def cluster_and_set_inits(self, n_kmeans_init = 1,
                                    z_init_eps=0.05):
        # K means init.
        for i in range(n_kmeans_init):
            km = KMeans(n_clusters = self.k_approx).fit(self.y)
            enertia = km.inertia_
            if (i == 0):
                enertia_best = enertia
                km_best = deepcopy(km)
            elif (enertia < enertia_best):
                enertia_best = enertia
                km_best = deepcopy(km)

        #e_z_init = np.full(self.vb_params['e_z'].shape(), z_init_eps)
        e_z_init = np.full(self.e_z.shape, z_init_eps)
        for n in range(len(km_best.labels_)):
            e_z_init[n, km_best.labels_[n]] = 1.0 - z_init_eps
        e_z_init /= np.expand_dims(np.sum(e_z_init, axis = 1), axis = 1)

        self.e_z = e_z_init
        #self.vb_params['e_z'].set(e_z_init)

        self.vb_params_dict['centroids'] = km_best.cluster_centers_.T

        if self.use_logitnormal_sticks:
            self.vb_params_dict['v_stick_mean'] = np.ones(self.k_approx - 1)
            self.vb_params_dict['v_stick_info'] = np.ones(self.k_approx - 1)

        else:
            raise NotImplementedError()
            # self.vb_params['global']['v_sticks']['alpha'].set(
            #             np.full((2, self.k_approx - 1), 1.0))

        # Set inital covariances
        dim = self.y.shape[1]
        # TODO: array of PSD matricies for vb_params not implemented yet.
        # gamma_init = np.zeros((self.k_approx, dim, dim))
        # for k in range(self.k_approx):
        #     indx = np.argwhere(km_best.labels_ == k).flatten()
        #     if len(indx == 1):
        #         # if there's only one datapoint in the cluster,
        #         # the covariance is not defined.
        #         gamma_init[k, :, :] = np.eye(dim)
        #     else:
        #         resid_k = self.y[indx, :] - km_best.cluster_centers_[k, :]
        #         gamma_init[k, :, :] = np.linalg.inv(np.cov(resid_k.T) + \
        #                                 np.eye(dim) * 1e-4)
        #
        self.vb_params_dict['gamma'] = np.ones(self.k_approx) * 0.1

        return self.vb_params_paragami.flatten(self.vb_params_dict, free = True)


    def optimize_full(self, init_free_par, bfgs_init=True):

        def bfgs_fun(x):
            return opt_lib.minimize_objective_bfgs(
                self.objective,
                init_x=x,
                precondition=False)

        if bfgs_init:
            initial_optimization_fun = bfgs_fun
        else:
            initial_optimization_fun = None

        def local_precondition_and_optimize(x):
            return precondition_and_optimize(self.objective, x)

        best_param, converged, x_conv, f_conv, grad_conv, obj_opt, opt_results = \
         opt_lib.repeatedly_optimize(
            objective=self.objective,
            optimization_fun=local_precondition_and_optimize,
            init_x=init_free_par,
            initial_optimization_fun=initial_optimization_fun,
            max_iter=100,
            gtol=1e-8, ftol=1e-8, xtol=1e-8, disp=False,
            keep_intermediate_optimizations=True)

        self.global_vb_params.set_free(best_param)
        self.set_optimal_z()

        class_weights = np.sum(self.e_z, axis=0) / self.n_obs
        if class_weights[-1] > 1/self.k_approx:
            warnings.warn('last cluster may not be un-occupied')

        return best_param, converged, x_conv, f_conv, grad_conv, obj_opt, opt_results


########################
# Sensitivity functions
#######################
class InterestingMoments(object):
    def __init__(self, model):
        self.model = model
        self.moment_params = vb.ModelParamsDict('Moment parameters')
        self.moment_params.push_param(
            vb.ArrayParam('centroids', shape=(model.dim, model.k_approx)))
        # self.moment_params.push_param(
        #     vb.ArrayParam('e_z', shape=(model.n_obs, model.k_approx)))
        self.moment_params.push_param(
            vb.VectorParam('cluster_weights', size=model.k_approx))
        self.moment_params.push_param(
            vb.VectorParam('v_sticks', size=model.k_approx - 1))

        self.moment_converter = obj_lib.ParameterConverter(
            par_in=self.model.global_vb_params,
            par_out=self.moment_params,
            converter=self.set_moments)
        self.get_moment_jacobian = self.moment_converter.free_to_vec_jacobian

    def set_moments(self):
        self.set_moments_from_free_par(self.model.global_vb_params.get_free())

    def set_moments_from_free_par(self, free_par):
        self.model.set_from_global_free_par(free_par)
        self.moment_params['centroids'].set(
            self.model.vb_params['global']['centroids'].get())

        # e_z = self.model.vb_params['e_z'].get()
        # self.moment_params['e_z'].set(e_z)
        self.moment_params['cluster_weights'].set(\
            self.model.get_e_cluster_probabilities())

        if self.model.vb_params.use_logitnormal_sticks:
            self.moment_params['v_sticks'].set(
                self.model.vb_params['global']['v_sticks']['mean'].get())
        else:
            self.moment_params['v_sticks'].set(
                self.model.vb_params['global']['v_sticks'].e())

    def set_and_get_moments_from_free_par(self, free_par):
        self.set_moments_from_free_par(free_par)
        return self.moment_params.get_vector()

# Get the expected posterior predictive number of distinct clusters.
def get_e_num_pred_clusters_from_free_par(free_par, model, n_samples = 100000):
    model.global_vb_params.set_free(free_par)
    mu = model.global_vb_params['v_sticks']['mean'].get()
    sigma = 1 / np.sqrt(model.global_vb_params['v_sticks']['info'].get())
    n_obs = model.n_obs
    return modeling_lib.get_e_number_clusters_from_logit_sticks(mu, sigma, n_obs, \
                                                        n_samples = n_samples)

# Get the expected posterior number of distinct clusters.
def get_e_num_clusters_from_free_par(free_par, model):
    model.global_vb_params.set_free(free_par)
    model.set_optimal_z()
    return modeling_lib.get_e_number_clusters_from_ez(model.e_z)

class ExpectedPredNumClusters(object):
    # Get the expected posterior predictive number of distinct clusters above
    # some given threshold.
    # Note that we cache the normal samples that we use to sample
    # cluster belongings.

    def __init__(self, model):
        self.model = model
        self.n_obs = self.model.y.shape[0]
        self.k_approx = self.model.k_approx

        self.set_normal_samples()

    def set_normal_samples(self, n_samples = 10000):
        self.unv_norm_samples = np.random.normal(0, 1, \
                                    size = (n_samples, self.k_approx - 1))

    def get_e_num_pred_heavy_clusters_from_free_par(self, free_par, threshold = 0):
        self.model.global_vb_params.set_free(free_par)
        self.model.set_optimal_z()

        mu = self.model.global_vb_params['v_sticks']['mean'].get()
        sigma = 1 / np.sqrt(self.model.global_vb_params['v_sticks']['info'].get())

        return modeling_lib.get_e_number_clusters_from_logit_sticks(
                                    mu, sigma, self.n_obs,
                                    threshold = threshold,
                                    unv_norm_samples = self.unv_norm_samples)

class ExpectedNumClustersFromZ(object):
    # Get the expected posterior number of distinct clusters above
    # some given threshold.
    # Note that we cache the uniform samples that we use to sample
    # cluster belongings.

    def __init__(self, model):
        self.model = model
        self.n_obs = self.model.y.shape[0]

        self.set_uniform_samples()

    def set_uniform_samples(self, n_samples = 10000):
        self.unif_samples = np.random.random((self.n_obs, n_samples))

    def get_e_num_heavy_clusters_from_free_par(self, free_par, threshold = 0):
        self.model.global_vb_params.set_free(free_par)
        self.model.set_optimal_z()

        return modeling_lib.get_e_num_large_clusters_from_ez(self.model.e_z,
                                                threshold = threshold,
                                                unif_samples = self.unif_samples)


#################################
# Functions to reload the model
#################################

# TODO: All this was copied over in a hurry from
# https://github.com/NelleV/genomic_time_series_bnp/bin/checkpoints/checkpoints/checkpoints_lib.py#L71
# It could probably be tidied up.

sp_string = '_sp_packed'
np_string = '_np_packed'

def get_timestamp():
    return datetime.today().timestamp()

# Pack a sparse csr_matrix in a json-seralizable format.
def pack_csr_matrix(sp_mat):
    assert sparse.isspmatrix_csr(sp_mat)
    sp_mat = sparse.csr_matrix(sp_mat)
    return { 'data': json_tricks.dumps(sp_mat.data),
             'indices': json_tricks.dumps(sp_mat.indices),
             'indptr': json_tricks.dumps(sp_mat.indptr),
             'shape': sp_mat.shape,
             'type': 'csr_matrix' }


# Convert the output of pack_csr_matrix back into a csr_matrix.
def unpack_csr_matrix(sp_mat_dict):
    assert sp_mat_dict['type'] == 'csr_matrix'
    data = json_tricks.loads(sp_mat_dict['data'])
    indices = json_tricks.loads(sp_mat_dict['indices'])
    indptr = json_tricks.loads(sp_mat_dict['indptr'])
    return sparse.csr_matrix(
        ( data, indices, indptr), shape = sp_mat_dict['shape'])

# Populate a dictionary with the minimum necessary fields to describe
# a pre-processing step.
def get_preprocessing_dict(method, p_values=None, log_fold_change=None):
    """
    Returns a pre-processing dict, population with defaults.
    Parameters
    ----------
    method : string
    p_values : ndarray, optional, default: None
    log_fold_change : ndarray, optional, default: None
    Returns
    -------
    dictionary
    """

    if p_values is None:
        p_values = np.array([])

    if log_fold_change is None:
        log_fold_change = np.array([])

    return {
        'timestamp': get_timestamp(),
        'method': method,
        'p_values' + np_string: json_tricks.dumps(p_values),
        'log_fold_change' + sp_string: json_tricks.dumps(log_fold_change)}


def get_fit_dict(method, initialization_method, seed, centroids,
                 labels=None,
                 cluster_assignments=None,
                 preprocessing_dict=None, basis_mat=None,
                 cluster_weights=None):
    """
    returns a "fit" dictionary
    Parameters
    ----------
    method : string
    initialization_method : string
    seed : int
        random seed
    centroids : ndarray
    labels : ndarray (n, ), optional, default: None
        1D-array containing the cluster labels.
        Either labels or cluster_assignments needs to be provided.
    cluster_assignments : {ndarray, csr_matrix}, optional, default: None
        n by k sparse or dense matrix contraining the cluster assignments or
        probability.
        Either labels or cluster_assignments needs to be provided.
    preprocessing_dict : dictionary
        dictionary containing a `timestamp` and `method` key
    basis_mat : ndarray, optional, default: None
    cluster_weights : ndarray, optional, default: None
    """
    if labels is None and cluster_assignments is None:
        raise ValueError(
            "In order to save the results, either provide labels or cluster"
            " assignment")

    if labels is not None and len(labels.shape) > 1:
        raise ValueError(
            "Labels should be a 1D array. "
            "Provided a %d-d array" % len(labels.shape))

    if cluster_assignments is not None and not sparse.issparse(cluster_assignments):
        cluster_assignments = sparse.csr_matrix(cluster_assignments)

    if labels is not None:
        if labels.max() >= centroids.shape[1]:
            raise ValueError(
                "There are %d centroids, but the labels contain up to %d "
                "element" % (centroids.shape[1], labels.max()))

        # The user provided labels and not a sparse matrix
        if cluster_assignments is not None:
            if np.any(labels != cluster_assignments.argmax(axis=1).A.flatten()):
                raise ValueError(
                    "Incoherence between the labels provided and the cluster "
                    "assignments.")
        else:
            cluster_assignments = sparse.csr_matrix(
                (np.ones(len(labels)), (np.arange(len(labels)), labels)),
                shape=(len(labels), centroids.shape[1]))

    if preprocessing_dict is None:
        preprocessing_dict = get_preprocessing_dict("NoPreprocessing")

    if basis_mat is None:
        basis_mat = np.array([])

    if cluster_weights is None:
        cluster_weights = cluster_assignments.sum(axis=0).A.flatten()
        cluster_weights /= cluster_weights.sum()

    return {
        'timestamp': get_timestamp(),
        'method': method,
        'initialization_method': initialization_method,
        'seed': seed,
        'preprocessing_method': preprocessing_dict['method'],
        'preprocessing_timestamp': preprocessing_dict['timestamp'],
        'centroids' + np_string: json_tricks.dumps(centroids),
        'basis_mat' + np_string: json_tricks.dumps(basis_mat),
        'cluster_weights' + np_string: json_tricks.dumps(cluster_weights),
        'cluster_assignments' + sp_string: pack_csr_matrix(cluster_assignments)
         }


def get_checkpoint_dictionary(model, kl_hessian=None, seed=None, compact=False):
    # Set the optimal z to remove arrayboxes if necessary.
    model.set_optimal_z()
    #e_z = model.vb_params['e_z'].get()
    labels = np.squeeze(np.argmax(model.e_z, axis=1))
    centroids = model.vb_params['global']['centroids'].get()
    fit_dict = get_fit_dict(
        method = 'gmm',
        initialization_method = 'kmeans',
        seed = seed,
        centroids = centroids,
        labels = labels,
        cluster_assignments = osp.sparse.csr_matrix(model.e_z),
        preprocessing_dict = None,
        basis_mat = None,
        cluster_weights = model.get_e_cluster_probabilities())
    fit_dict['k_approx'] = model.k_approx
    fit_dict['use_bnp_prior'] = model.vb_params.use_bnp_prior
    fit_dict['vb_global_free_par' + np_string] = \
        json_tricks.dumps(model.global_vb_params.get_free())
    fit_dict['vb_global_vec_par' + np_string] = \
        json_tricks.dumps(model.global_vb_params.get_vector())
    fit_dict['prior_params_vec' + np_string] = \
        json_tricks.dumps(model.prior_params.get_vector())
    fit_dict['gh_deg'] = model.gh_deg
    fit_dict['y' + np_string] = json_tricks.dumps(model.y)
    fit_dict['use_weights'] = model.use_weights
    fit_dict['sample_weights' + np_string] = json_tricks.dumps(model.weights.get())
    fit_dict['use_logitnormal_sticks'] = \
        model.vb_params.use_logitnormal_sticks

    if not kl_hessian is None:
        fit_dict['kl_hessian' + np_string] = json_tricks.dumps(kl_hessian)

    # Optionally do not save the large objects.  Zero them out here
    # rather than before calling get_fit_dict() because get_fit_dict()
    # does validation on e_z.
    if compact:
        empty_array = json_tricks.dumps(np.array([]))
        fit_dict['y'] = empty_array
        fit_dict['e_z'] = empty_array
        fit_dict['labels'] = empty_array

    return fit_dict


def get_model_from_checkpoint(fit_dict):
    # load model from fit dictionary

    # the data
    y = json_tricks.loads(fit_dict['y' + np_string])
    dim = np.shape(y)[1]

    # some model parameters
    k_approx = fit_dict['k_approx']

    # set prior parameters
    prior_params = get_default_prior_params(dim)
    prior_params_vec = json_tricks.loads(
        fit_dict['prior_params_vec' + np_string])
    prior_params.set_vector(prior_params_vec)

    # define gmm
    model = DPGaussianMixture(y, \
                k_approx, \
                prior_params, \
                gh_deg = fit_dict['gh_deg'], \
                use_bnp_prior = fit_dict['use_bnp_prior'], \
                use_logitnormal_sticks=fit_dict['use_logitnormal_sticks'])

    global_free_par = json_tricks.loads(
        fit_dict['vb_global_free_par' + np_string])
    global_vec_par = json_tricks.loads(
        fit_dict['vb_global_vec_par' + np_string])
    model.global_vb_params.set_free(global_free_par)
    assert np.linalg.norm(model.global_vb_params.get_vector() -
                          global_vec_par) < 1e-6

    # set weights if necessary
    model.use_weights = fit_dict['use_weights']
    weights = json_tricks.loads(fit_dict['sample_weights' + np_string])
    model.weights.set(np.reshape(weights, (model.n_obs, 1)))

    model.set_optimal_z()

    return model


def get_kl_hessian_from_checkpoint(fit_dict):
    return json_tricks.loads(fit_dict['kl_hessian' + np_string])
