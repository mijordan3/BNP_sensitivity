import sys
sys.path.insert(0, './../../LinearResponseVariationalBayes.py')

import LinearResponseVariationalBayes as vb
import LinearResponseVariationalBayes.ExponentialFamilies as ef
import LinearResponseVariationalBayes.SparseObjectives as obj_lib

import autograd
import autograd.numpy as np
import autograd.scipy as sp

import scipy as osp

from scipy import optimize
from scipy import linalg

from sklearn.cluster import KMeans

from copy import deepcopy

import time

# import individual_fits_lib as individ_lib

import modeling_lib as model_lib
import functional_sensitivity_lib as fun_sens_lib

# Should we put this somewhere? We only need get_hess_inv_sqrt
#sys.path.insert(0, './../../genomic_time_series_bnp/src/vb_modeling/')
#sys.path.insert(0, '/home/rgiordan/Documents/git_repos/genomic_time_series_bnp/src/vb_modeling/')
#import sparse_hessians_lib as sp_hess_lib


import checkpoints
import json
import json_tricks

import checkpoints

from numpy.polynomial.hermite import hermgauss

import matplotlib.pyplot as plt

from copy import deepcopy
##########################
# Set up vb parameters
##########################

def push_global_params(global_params, dim, k_approx, use_logitnormal_sticks):
    # The variational parameters for cluster centroids,
    # the stick lengths, and cluster variances

    # the kth column is the centroid for the kth cluster
    # variational distribution for the centroids are a point mass
    global_params.push_param(vb.ArrayParam('centroids', shape=(dim, k_approx)))

    # note the shape k_approx - 1 ...
    # the last stick is always 1 in our approximation
    if use_logitnormal_sticks:
        # variational distribution for each stick is logitnormal
        global_params.push_param(
            vb.UVNParamVector(name = 'v_sticks', length = (k_approx - 1),
                              min_info = 1e-4))
    else:
        # else its a beta
        global_params.push_param(
            vb.DirichletParamArray(name='v_sticks', shape=(2, k_approx - 1)))

    # variational gamma parameters to estimate data variance
    # variational distribution for the variance is a point mass
    global_params.push_param(vb.PosDefMatrixParamVector('gamma', \
                                length = k_approx, matrix_size = dim))

def push_local_params(vb_params, n_obs, k_approx):
    vb_params.push_param(vb.SimplexParam(name='e_z', shape=(n_obs, k_approx)))

def get_vb_params(dim, k_approx, n_obs, gh_deg, \
                    use_bnp_prior = True,
                    use_logitnormal_sticks = True):

    vb_params = vb.ModelParamsDict('vb_params')

    vb_params.use_logitnormal_sticks = use_logitnormal_sticks

    # global parameters: the sticks, the centoids,
    # and the variance
    global_params = vb.ModelParamsDict('global')
    push_global_params(global_params, dim, k_approx, \
                        vb_params.use_logitnormal_sticks)
    vb_params.push_param(global_params)

    # local parameters:
    # the cluster belongings
    push_local_params(vb_params, n_obs, k_approx)

    # not really vb parameters: set the weights and locations for
    # integrating the logitnormal
    gh_loc, gh_weights = hermgauss(gh_deg)
    vb_params.gh_loc = gh_loc
    vb_params.gh_weights = gh_weights

    # also save this flag here
    vb_params.use_bnp_prior = use_bnp_prior

    return vb_params

##########################
# Set up prior parameters
##########################
def get_default_prior_params(dim):
    prior_params = vb.ModelParamsDict('prior_params')

    # DP prior parameter
    prior_params.push_param(
        vb.ScalarParam(name = 'alpha', lb = 0.0, val = 4.0))

    # prior on the centroids
    prior_params.push_param(
        vb.ScalarParam(name = 'prior_centroid_mean', val=0.0))
    prior_params.push_param(
        vb.ScalarParam(name = 'prior_centroid_info', val=0.1, lb = 0.0))

    # prior on the variance
    prior_params.push_param(
        vb.ScalarParam(name = 'prior_gamma_df', val=dim * 2, lb = dim - 1))
    prior_params.push_param(
        vb.PosDefMatrixParam(name = 'prior_gamma_inv_scale', size = dim))

    prior_params['prior_gamma_inv_scale'].set(np.eye(dim))

    return prior_params

##########################
# Expected prior terms
##########################

def get_e_centroid_prior(vb_params, prior_params):
    beta = vb_params['global']['centroids'].get()

    beta_base_prior = ef.uvn_prior(
        prior_mean = prior_params['prior_centroid_mean'].get(),
        prior_info = prior_params['prior_centroid_info'].get(),
        e_obs = beta.flatten(),
        var_obs = np.array([0.]))

    return np.sum(beta_base_prior)

def get_e_log_wishart_prior(vb_params, prior_params):

    df = prior_params['prior_gamma_df'].get()
    V_inv = prior_params['prior_gamma_inv_scale'].get()

    gamma = vb_params['global']['gamma'].get()
    dim = np.shape(gamma)[1]

    # New way:
    tr_V_inv_gamma = np.einsum('ij, kji -> k', V_inv, gamma)

    # Old way for the record:
    # V_inv_gamma = np.einsum('ij, kjl -> kil', V_inv, gamma)
    # eye = np.eye(dim)
    # tr_V_inv_gamma = np.einsum('kij, ji -> k', V_inv_gamma, eye)

    return np.sum((df - dim - 1) / 2 * np.linalg.slogdet(gamma)[1] -
                    0.5 * tr_V_inv_gamma)

def get_e_log_prior(vb_params, prior_params, u = None):

    if u == None:
        # no perturbation
        dp_prior = model_lib.get_dp_prior(vb_params, prior_params)
    else:
        dp_prior = np.sum(fun_sens_lib.dp_perturbed_prior(vb_params, prior_params, \
                        u = u))
    e_gamma_prior = get_e_log_wishart_prior(vb_params, prior_params)

    e_centroid_prior = get_e_centroid_prior(vb_params, prior_params)

    return e_gamma_prior + e_centroid_prior + dp_prior

##########################
# Entropy
##########################
def get_entropy(vb_params):
    e_z = vb_params['e_z'].get()
    return model_lib.multinom_entropy(e_z) + \
        model_lib.get_stick_entropy(vb_params)

##########################
# Likelihood term
##########################
def get_loglik_obs_by_nk(y, vb_params):
    # returns a n x k matrix whose nkth entry is
    # the likelihood for the nth observation
    # belonging to the kth cluster

    centroid = vb_params['global']['centroids'].get()
    gamma = vb_params['global']['gamma'].get()
    assert np.shape(y)[1] == np.shape(centroid)[0]
    assert np.shape(gamma)[0] == np.shape(centroid)[1]
    assert np.shape(gamma)[1] == np.shape(centroid)[0]

    dim = np.shape(y)[1]

    data2_term = np.einsum('ni, kij, nj -> nk', y, gamma, y)
    cross_term = np.einsum('ni, kij, jk -> nk', y, gamma, centroid)
    centroid2_term = np.einsum('ik, kij, jk -> k', centroid, gamma, centroid)

    squared_term = data2_term - 2 * cross_term + \
                    np.expand_dims(centroid2_term, axis = 0)

    # squared_term = (np.expand_dims(data2_term, axis = 1) -
    #                 2 * cross_term +
    #                 np.expand_dims(centroid2_term, axis = 0))

    return - 0.5 * squared_term + 0.5 * np.linalg.slogdet(gamma)[1][None, :]

##########################
# the model class
##########################
class DPGaussianMixture(object):
    def __init__(self, y, k_approx, prior_params, gh_deg, \
                    use_bnp_prior = True, use_logitnormal_sticks = False,
                    u = None):
        # y is the observed data
        # x is the matrix of 'covariates', in our case, the spline bases
        # k_approx is the number of clusters
        # prior_params is the class in which the prior parameters are stored
        # gh_deg is the parameter used for logitnormal integration

        # u is the functional perturbation

        self.y = y

        self.dim = y.shape[1]
        self.k_approx = k_approx
        self.n_obs = y.shape[0]
        self.gh_deg = gh_deg

        self.use_weights = False
        self.weights = np.ones((self.n_obs, 1))

        self.vb_params = get_vb_params(
            self.dim, k_approx, self.n_obs, gh_deg,
            use_bnp_prior = use_bnp_prior,
            use_logitnormal_sticks = use_logitnormal_sticks)

        self.prior_params = deepcopy(prior_params)

        # functional perturbation
        self.u = u
        if u != None:
            assert use_logitnormal_sticks , \
                'functonal sensitivty only computed with logitnormal sticks'

        # Make a set of parameters for optimization.  Note that the
        # parameters are passed by reference, so updating global_vb_params
        # updates vb_params.
        self.global_vb_params = vb.ModelParamsDict()
        self.global_vb_params.push_param(self.vb_params['global']['centroids'])
        self.global_vb_params.push_param(self.vb_params['global']['gamma'])
        self.global_vb_params.push_param(self.vb_params['global']['v_sticks'])

        self.objective = \
            obj_lib.Objective(self.global_vb_params, self.set_z_get_kl)
        self.objective.preconditioning = False
        self.objective.logger.callback = self.display_optimization_status

        # Sensitivity terms.

        # To priors:
        # self.prior_indices = obj_lib.make_index_param(self.prior_params)
        # self.kl_param_and_prior_obj = obj_lib.TwoParameterObjective(
        #     self.prior_params, self.global_vb_params, self.set_z_get_kl)
        ## self.__get_kl_prior_free_grad = autograd.grad(
        ##     self.get_kl_from_prior_and_free_par, argnum=1)
        ## self.get_kl_prior_cross_hess = autograd.jacobian(
        ##     self.__get_kl_prior_free_grad, argnum=0)
        #self.get_kl_prior_cross_hess = kl_param_and_prior_obj.fun_free_hessian12

        # To data:
        self.per_gene_kl_obj = obj_lib.Objective(
            self.global_vb_params, self.get_per_gene_kl)
        #self.get_data_cross_hess = autograd.jacobian(self.get_per_gene_kl)
        self.get_data_cross_hess = self.per_gene_kl_obj.fun_free_jacobian

    def __str__(self):
        b = self.vb_params['global']['b'].e()
        return '\n'.join([
            'Post sd: {}'.format(1 / np.sqrt(
                self.vb_params['global']['gamma'].get())),
            str(self.vb_params['global']['beta']),
            'b mean = {} sd = {} '.format(np.mean(b), np.std(b)),
            'Z totals: {}'.format(np.sum(self.vb_params['e_z'].get(), axis=0)) ])

    def set_random(self):
        self.vb_params['global'].set_free(np.random.random(
            self.vb_params['global'].free_size()))
        self.set_optimal_z()

    def set_from_global_free_par(self, free_par):
        self.global_vb_params.set_free(free_par)
        return self.set_optimal_z(return_loglik_obs_by_nk = True)

    def get_z_nat_params(self, return_loglik_obs_by_nk = False):
        loglik_obs_by_nk = get_loglik_obs_by_nk(
            self.y, self.vb_params)

        if self.vb_params.use_bnp_prior:
            e_log_cluster_probs = \
                model_lib.get_e_log_cluster_probabilities(self.vb_params)
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
        self.vb_params['e_z'].set(np.exp(z_nat_param - log_const[:, None]))

        if return_loglik_obs_by_nk:
            return loglik_obs_by_nk

    def get_kl(self):
        # ...with the current value of z.
        e_z = self.vb_params['e_z'].get()
        loglik_obs_by_nk = get_loglik_obs_by_nk(
            self.y, self.vb_params)

        if self.use_weights:
            assert np.shape(self.weights) == (self.n_obs, 1)
            e_loglik_obs = np.sum(self.weights * e_z * loglik_obs_by_nk)
        else:
            e_loglik_obs = np.sum(e_z * loglik_obs_by_nk)

        if self.vb_params.use_bnp_prior:
            e_loglik_ind = model_lib.loglik_ind(self.vb_params)
        else:
            e_loglik_ind = 0.

        e_loglik = e_loglik_ind + e_loglik_obs

        assert(np.isfinite(e_loglik))

        entropy = np.squeeze(get_entropy(self.vb_params))
        assert(np.isfinite(entropy))

        e_log_prior = np.squeeze(
            get_e_log_prior(self.vb_params, self.prior_params, u = self.u))
        assert(np.isfinite(e_log_prior))

        elbo = e_log_prior + entropy + e_loglik
        return -1 * elbo

    def set_z_get_kl(self):
        # Update z and evaluate likelihood.
        loglik_obs_by_nk = self.set_optimal_z(return_loglik_obs_by_nk = True)
        e_z = self.vb_params['e_z'].get()

        if self.use_weights:
            assert np.shape(self.weights) == (self.n_obs, 1)
            e_loglik_obs = np.sum(self.weights * e_z * loglik_obs_by_nk)
        else:
            e_loglik_obs = np.sum(e_z * loglik_obs_by_nk)

        if self.vb_params.use_bnp_prior:
            e_loglik_ind = model_lib.loglik_ind(self.vb_params)
        else:
            e_loglik_ind = 0.

        e_loglik = e_loglik_ind + e_loglik_obs

        if not np.isfinite(e_loglik):
            print('gamma', self.vb_params['global']['gamma'].get())
            print('det gamma', np.linalg.slogdet(self.vb_params['global']['gamma'].get())[1])
            print('cluster weights', np.sum(e_z, axis = 0))

            # print('loglik_obs_by_nk', np.all(np.isfinite(loglik_obs_by_nk)))
            # print('e_z', np.all(np.isfinite(e_z)))
            # print('e_loglik_ind', e_loglik_ind)

        assert(np.isfinite(e_loglik))

        entropy = np.squeeze(get_entropy(self.vb_params))
        assert(np.isfinite(entropy))

        e_log_prior = np.squeeze(
            get_e_log_prior(self.vb_params, self.prior_params, u = self.u))
        assert(np.isfinite(e_log_prior))

        # print(self.vb_params['global']['gamma'].get())
        elbo = e_log_prior + entropy + e_loglik
        return -1 * elbo


    def get_e_cluster_probabilities(self):
        if self.vb_params.use_logitnormal_sticks:
            e_v = ef.get_e_logitnormal(
                lognorm_means = self.vb_params['global']['v_sticks']['mean'].get(),
                lognorm_infos = self.vb_params['global']['v_sticks']['info'].get(),
                gh_loc = self.vb_params.gh_loc,
                gh_weights = self.vb_params.gh_weights)
        else:
            e_sticks = self.vb_params['global']['v_sticks'].e()
            e_v = e_sticks[0, :]
            e_1mv = e_sticks[1, :]

        return model_lib.get_mixture_weights(e_v)


    ########################
    # Saving and loading.

    def get_checkpoint_dictionary(self, seed=None, compact=False):
        e_z = self.vb_params['e_z'].get()
        labels = np.squeeze(np.argmax(e_z, axis=1))
        centroids = self.vb_params['global']['centroids'].get()
        fit_dict = checkpoints.get_fit_dict(
            method = 'gmm',
            initialization_method = 'kmeans',
            seed = seed,
            centroids = centroids,
            labels = labels,
            cluster_assignments = osp.sparse.csr_matrix(e_z),
            preprocessing_dict = None,
            basis_mat = None,
            cluster_weights = self.get_e_cluster_probabilities())
        np_string = checkpoints.np_string
        fit_dict['k_approx'] = self.k_approx
        fit_dict['use_bnp_prior'] = self.vb_params.use_bnp_prior
        fit_dict['vb_global_free_par' + np_string] = \
            json_tricks.dumps(self.global_vb_params.get_free())
        fit_dict['vb_global_vec_par' + np_string] = \
            json_tricks.dumps(self.global_vb_params.get_vector())
        fit_dict['prior_params_vec' + np_string] = \
            json_tricks.dumps(self.prior_params.get_vector())
        fit_dict['gh_deg'] = self.gh_deg
        fit_dict['y' + np_string] = json_tricks.dumps(self.y)
        fit_dict['use_weights'] = self.use_weights
        fit_dict['sample_weights' + np_string] = json_tricks.dumps(self.weights)
        fit_dict['use_logitnormal_sticks'] = \
            self.vb_params.use_logitnormal_sticks

        # Optionally do not save the large objects.  Zero them out here
        # rather than before calling get_fit_dict() because get_fit_dict()
        # does validation on e_z.
        if compact:
            empty_array = json_tricks.dumps(np.array([]))
            fit_dict['y'] = empty_array
            fit_dict['e_z'] = empty_array
            fit_dict['labels'] = empty_array

        return fit_dict


    ########################
    # Sensitivity functions.

    # def get_kl_from_prior_and_free_par(self, prior_free, free_par):
    #     self.prior_params.set_free(prior_free)
    #     self.global_vb_params.set_free(free_par)
    #     return self.set_z_get_kl()

    # Get the likelihood for each gene.  This is equivalent to the derivative
    # of the KL divergence with respect to the weight vector.
    # def get_per_gene_kl(self, free_par):
    #     self.global_vb_params.set_free(free_par)
    #     loglik_obs_by_nk = self.set_optimal_z(return_loglik_obs_by_nk = True)
    #     e_z = self.vb_params['e_z'].get()
    #     return -1 * np.sum(e_z * loglik_obs_by_nk, axis=1)

    def get_per_gene_kl(self):
        loglik_obs_by_nk = self.set_optimal_z(return_loglik_obs_by_nk = True)
        e_z = self.vb_params['e_z'].get()
        return -1 * np.sum(e_z * loglik_obs_by_nk, axis=1)

    ########################
    # Optimization functions follow.

    def get_rmse(self):
        # This can be a callback in optimization as a sanity check.

        e_z = self.vb_params['e_z'].get()
        centroids = self.global_vb_params['centroids'].get()

        post_means = np.dot(e_z, centroids.T)

        resid = self.y - post_means
        return np.sqrt(np.sum(resid ** 2))

    def get_z_allocation(self):
        self.set_optimal_z()
        return np.sum(self.vb_params['e_z'].get(), axis=0)

    # A callback for use with SparseObjectives:Objective.
    def display_optimization_status(self, logger):
        # Note that the logger gets passed the x that was given to the
        # function, /not/ the value before multiplication by the preconditioner.
        free_param = logger.x
        self.set_from_global_free_par(free_param)
        print('Iter: {}\t RMSE: {}\t Objective: {}'.format(
                logger.iter,  self.get_rmse(), logger.value))

    # def get_per_observation_fits(self):
    #     # Initialize all the data points.
    #     beta_obs = np.full((self.n_obs, self.beta_dim), float('nan'))
    #     shift_obs = np.full(self.n_obs, float('nan'))
    #
    #     individual_fits = individ_lib.IndividiualFits(self)
    #
    #     xtx = np.matmul(self.x.T, self.x)
    #     xtxinv_x = np.linalg.solve(xtx, self.x.T)
    #
    #     for obs in range(self.n_obs):
    #         shift_obs[obs] = np.mean(self.y[obs, :])
    #         beta_obs[obs] = model_lib.get_scale_shift_regression(
    #             xtxinv_x, self.y[obs, :], 1.0, shift_obs[obs])
    #
    #     return beta_obs, shift_obs

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

        e_z_init = np.full(self.vb_params['e_z'].shape(), z_init_eps)
        for n in range(len(km_best.labels_)):
            e_z_init[n, km_best.labels_[n]] = 1.0 - z_init_eps
        e_z_init /= np.expand_dims(np.sum(e_z_init, axis = 1), axis = 1)

        self.vb_params['e_z'].set(e_z_init)

        centroids_init = km_best.cluster_centers_.T
        # beta_init / np.linalg.norm(beta_init, axis=0, keepdims=True)
        self.vb_params['global']['centroids'].set(centroids_init)

        if self.vb_params.use_logitnormal_sticks:
            self.vb_params['global']['v_sticks']['mean'].set(
                np.full(self.k_approx - 1, 1.0))
            self.vb_params['global']['v_sticks']['info'].set(
                np.full(self.k_approx - 1, 1.0))
        else:
            self.vb_params['global']['v_sticks']['alpha'].set(
                        np.full((2, self.k_approx - 1), 1.0))

        # Set inital covariances
        dim = self.y.shape[1]
        gamma_init = np.zeros((self.k_approx, dim, dim))
        for k in range(self.k_approx):
            indx = np.argwhere(km_best.labels_ == k).flatten()
            resid_k = self.y[indx, :] - km_best.cluster_centers_[k, :]
            gamma_init[k, :, :] = np.linalg.inv(np.cov(resid_k.T) + \
                                    np.eye(dim) * 1e-4)

        self.vb_params['global']['gamma'].set(gamma_init)

        return self.global_vb_params.get_free()


    def minimize_kl_newton(self, init_x, precondition,
                           maxiter = 50, gtol = 1e-6, disp = True,
                           print_every = 1, init_logger = True):
        opt_time = time.time()
        if init_logger:
            self.objective.logger.initialize()
        self.objective.logger.print_every = print_every
        self.objective.preconditioning = precondition
        if precondition:
            init_x_cond = np.linalg.solve(self.objective.preconditioner, init_x)
            vb_opt = optimize.minimize(
                lambda par: self.objective.fun_free_cond(par, verbose=disp),
                x0=init_x_cond,
                jac=self.objective.fun_free_grad_cond,
                hessp=self.objective.fun_free_hvp_cond,
                method='trust-ncg',
                options={'maxiter': maxiter, 'gtol': gtol, 'disp': disp})
        else:
            vb_opt = optimize.minimize(
                lambda par: self.objective.fun_free(par, verbose=disp),
                x0=init_x,
                jac=self.objective.fun_free_grad,
                hessp=self.objective.fun_free_hvp,
                method='trust-ncg',
                options={'maxiter': maxiter, 'gtol': gtol, 'disp': disp})
        opt_time = time.time() - opt_time
        return vb_opt, opt_time

    def minimize_kl_bfgs(self, init_x, precondition,
                         maxiter = 500, disp = True, print_every = 50,
                         init_logger = True):
        opt_time = time.time()
        if init_logger:
            self.objective.logger.initialize()

        self.objective.logger.print_every = print_every
        self.objective.preconditioning = precondition
        if precondition:
            init_x_cond = np.linalg.solve(self.objective.preconditioner, init_x)
            vb_opt = optimize.minimize(
                lambda par: self.objective.fun_free_cond(par, verbose=disp),
                x0=init_x_cond,
                jac=self.objective.fun_free_grad_cond,
                method='BFGS',
                options={'maxiter': maxiter, 'disp': disp})
        else:
            vb_opt = optimize.minimize(
                lambda par: self.objective.fun_free(par, verbose=disp),
                x0=init_x,
                jac=self.objective.fun_free_grad,
                method='BFGS',
                options={'maxiter': maxiter, 'disp': disp})
        opt_time = time.time() - opt_time
        return vb_opt, opt_time


    def precondition_and_optimize(
        self, init_x, gtol=1e-8, maxiter=100, disp=True, print_every_n=10):

        inv_hess_sqrt, kl_hessian, kl_hessian_corrected = \
            self.get_preconditioner(init_x)
        self.objective.preconditioner = inv_hess_sqrt
        vb_opt, opt_time = self.minimize_kl_newton(
            precondition = True,
            init_x = init_x,
            maxiter = maxiter, gtol = gtol, disp = disp,
            print_every=print_every_n, init_logger = False)
        return self.objective.uncondition_x(vb_opt.x), vb_opt, opt_time, \
               kl_hessian, kl_hessian_corrected


    def optimize_full(self, init_free_par, do_second_bfgs=True,
                      init_max_iter=300, final_max_iter=300,
                      gtol=1e-8, ftol=1e-8, xtol=1e-8, max_condition_iter=10,
                      disp=True):

        if disp:
            print('BGFS')
        init_opt, init_opt_time = self.minimize_kl_bfgs(
            precondition=False, init_x=init_free_par,
            maxiter = init_max_iter, disp = disp,
            print_every=10, init_logger = True)
        init_x = init_opt.x

        newton_time = 0.0

        x_diff = float('inf')
        f_diff = float('inf')
        x_conv = False
        f_conv = False
        f_val = self.objective.fun_free(init_x)
        i = 0

        # Converge if either the x or the f converge.
        if disp:
            print('Conditioned Newton:')
        while i < max_condition_iter and (not x_conv) and (not f_conv):
            if disp:
                print('i = ', i)
            i += 1
            new_init_x, vb_opt, opt_time, kl_hessian, kl_hessian_corrected = \
                self.precondition_and_optimize(
                    init_x, gtol=gtol, maxiter=final_max_iter, disp=disp)
            new_f_val = self.objective.fun_free(new_init_x)

            newton_time += opt_time
            x_diff = np.sum(np.abs(init_x - new_init_x))
            f_diff = np.abs(new_f_val - f_val)
            x_conv = x_diff < xtol
            f_conv = f_diff < ftol
            init_x = new_init_x
            f_val = new_f_val
            if disp:
                print('Iter {}: x_diff = {}, f_diff = {}'.format(i, x_diff, f_diff))

        best_param = init_x

        # print('done')

        return best_param, kl_hessian, kl_hessian_corrected, \
               init_opt_time, newton_time, x_conv, f_conv, vb_opt

    def get_preconditioner(self, free_par):
        obj_hessian = self.objective.fun_free_hessian(free_par)
        inv_hess_sqrt, hessian_corrected = \
            obj_lib.get_sym_matrix_inv_sqrt(obj_hessian, ev_min=1, ev_max=1e5)
            #sp_hess_lib.get_hess_inv_sqrt(obj_hessian)

        return inv_hess_sqrt, obj_hessian, hessian_corrected

########################
# Sensitivity functions
#######################
class InterestingMoments(object):
    def __init__(self, model):
        self.model = deepcopy(model)
        self.moment_params = vb.ModelParamsDict('Moment parameters')
        self.moment_params.push_param(
            vb.ArrayParam('centroids', shape=(model.dim, model.k_approx)))
        self.moment_params.push_param(
            vb.ArrayParam('e_z', shape=(model.n_obs, model.k_approx)))
        self.moment_params.push_param(
            vb.VectorParam('cluster_weights', size=model.k_approx))
        self.moment_params.push_param(
            vb.VectorParam('v_sticks', size=model.k_approx - 1))

    def set_moments_from_free_par(self, free_par):
        self.model.set_from_global_free_par(free_par)
        self.moment_params['centroids'].set(
            self.model.vb_params['global']['centroids'].get())

        e_z = self.model.vb_params['e_z'].get()
        self.moment_params['e_z'].set(e_z)
        self.moment_params['cluster_weights'].set(\
            self.model.get_e_cluster_probabilities())

        if self.model.vb_params.use_logitnormal_sticks:
            self.moment_params['v_sticks'].set(
                self.model.vb_params['global']['v_sticks']['mean'].get())
        else:
            self.moment_params['v_sticks'].set(
                self.model.vb_params['global']['v_sticks'].e())


class StickSensitivity(object):
    def __init__(self, model, best_param, kl_hessian, dgdeta):
        self.model = model
        self.best_param = best_param
        self.log_q_logit_stick_obj = obj_lib.Objective(
            self.model.global_vb_params, self.get_log_q_logit_stick)
        self.log_q_logit_sticks_obj = obj_lib.Objective(
            self.model.global_vb_params, self.get_log_q_logit_sticks)
        self.set_lr_matrix(kl_hessian, dgdeta)

    def set_lr_matrix(self, kl_hessian, dgdeta):
        self.lr_mat = -1 * np.linalg.solve(kl_hessian, dgdeta)

    # The log variational density of stick k at logit_v
    # in the logit_stick space.
    def get_log_q_logit_stick(self, logit_v, k):
        mean = self.model.global_vb_params['v_sticks']['mean'].get()[k]
        info = self.model.global_vb_params['v_sticks']['info'].get()[k]
        return -0.5 * (info * (logit_v - mean) ** 2 - np.log(info))

    # The base prior (of any stick) at logit_v in the logit stick space.
    def get_log_p0_logit_stick(self, logit_v):
        alpha = self.model.prior_params['alpha'].get()
        return logit_v - (alpha + 1) * np.log1p(np.exp(logit_v))

    def get_single_stick_influence(self, logit_v, k):
        log_q = self.get_log_q_logit_stick(logit_v, k)
        log_p0 = self.get_log_p0_logit_stick(logit_v)

        log_q_grad = self.log_q_logit_stick_obj.fun_free_grad(
            self.best_param, logit_v=logit_v, k=k)
        return(self.lr_mat.T @ log_q_grad * np.exp(log_q - log_p0))


# A class to calculate sensitivity to the stick-breaking distribution.
#
# Parameters:
#   model: A DPGaussianMixture object.
#   best_param: A free parameter at a local minimum of the KL divergence.
#   kl_hessian: The Hessian of the KL divergence evaluated at best_param.
#   dgdeta: The derivative of the moments of interest with respect to the
#           variational free parameters.
class StickSensitivity(object):
    def __init__(self, model, best_param, kl_hessian, dgdeta):
        self.model = model
        self.best_param = best_param
        self.log_q_logit_stick_obj = obj_lib.Objective(
            self.model.global_vb_params, self.get_log_q_logit_stick)
        self.set_lr_matrix(kl_hessian, dgdeta)

    def set_lr_matrix(self, kl_hessian, dgdeta):
        self.lr_mat = -1 * np.linalg.solve(kl_hessian, dgdeta)

    # The log variational density of stick k at logit_v
    # in the logit_stick space.
    def get_log_q_logit_stick(self, logit_v, k):
        mean = self.model.global_vb_params['v_sticks']['mean'].get()[k]
        info = self.model.global_vb_params['v_sticks']['info'].get()[k]
        return -0.5 * (info * (logit_v - mean) ** 2 - np.log(info))

    # Return a vectof of log variational densities for all sticks at logit_v
    # in the logit stick space.
    def get_log_q_logit_all_sticks(self, logit_v):
        mean = self.model.global_vb_params['v_sticks']['mean'].get()
        info = self.model.global_vb_params['v_sticks']['info'].get()
        return -0.5 * (info * (logit_v - mean) ** 2 - np.log(info))

    # The base prior (of any stick -- they are all the same) at logit_v in
    # the logit stick space.
    def get_log_p0_logit_stick(self, logit_v):
        alpha = self.model.prior_params['alpha'].get()
        return logit_v - (alpha + 1) * np.log1p(np.exp(logit_v))

    # Get the influence function for a perturbation to the prior on stick k.
    def get_single_stick_influence(self, logit_v, k):
        log_q = self.get_log_q_logit_stick(logit_v, k)
        log_p0 = self.get_log_p0_logit_stick(logit_v)

        log_q_grad = self.log_q_logit_stick_obj.fun_free_grad(
            self.best_param, logit_v=logit_v, k=k)
        return(self.lr_mat.T @ log_q_grad * np.exp(log_q - log_p0))

    # Get the influence function for a perturbation to all stick priors
    # simultaneously.
    def get_all_stick_influence(self, logit_v):
        log_q_vec = self.get_log_q_logit_all_sticks(logit_v)

        # The prior is the same for every stick.
        log_p0 = self.get_log_p0_logit_stick(logit_v)

        log_q_grad_vec = np.array([
            self.log_q_logit_stick_obj.fun_free_grad(
                self.best_param, logit_v=logit_v, k=k)
            for k in range(self.model.k_approx - 1) ])

        dens_ratios = np.exp(log_q_vec - log_p0)
        return(self.lr_mat.T @ np.einsum('kd,k->d', log_q_grad_vec, dens_ratios))


class LinearSensitivity(object):
    def __init__(self, model, moment_model):
        self.model = deepcopy(model)
        self.moment_model = moment_model

        self.optimal_global_free_params = self.model.global_vb_params.get_free()
        self.set_sensitivities(self.optimal_global_free_params)

        # d/d_eta q_eta(\theta)
        self.get_log_q_pi_obj = obj_lib.Objective(self.get_log_q_pi)
        self.get_log_q_pi_jac_autodiff = autograd.jacobian(self.get_log_q_pi, 0)


    def set_sensitivities(self, free_par):
        # Save the parameter
        self.free_par = deepcopy(free_par)

        print('KL Hessian:')
        self.kl_hessian = self.model.objective.fun_free_hessian(free_par)

        print('Prior Hessian...')
        self.prior_cross_hess = self.model.get_kl_prior_cross_hess(
             self.model.prior_params.get_free(), free_par)

        print('Data Hessian...')
        self.data_cross_hess = self.model.get_data_cross_hess(free_par)

        print('Linear systems...')
        self.kl_hessian_chol = osp.linalg.cho_factor(self.kl_hessian)
        self.kl_hessian_inv = np.linalg.inv(self.kl_hessian)

        self.prior_sens_mat = -1 * osp.linalg.cho_solve(
            self.kl_hessian_chol, self.prior_cross_hess)
        self.data_sens_mat = -1 * osp.linalg.cho_solve(
            self.kl_hessian_chol, self.data_cross_hess.T)

        print('Done.')

    def get_functional_sensitivity(self, u, k,
                                        imp_propn = 4,
                                        n_samples = 10000):
        # returns the KL hessian inverse * d/d\eta E_q[u(\theta) / p_0(\theta)]

        # returns a vector of length len(global free params)
        # giving the sensitivity of the global free parameters
        # with respect to perturbing the prior of the k stick by a function u

        # we compute the expectation with importance sampling. imp_propn
        # gives the factor of the variance that we multiply by;
        # n_samples is the number of samples we draw to compute the expectation

        assert self.model.vb_params.use_logitnormal_sticks , \
            'functional sensitivty only computed with logitnormal sticks. '

        # importance sample
        self.model.global_vb_params.set_free(self.optimal_global_free_params)
        mean_sample = self.model.vb_params['global']['v_sticks']['mean'].get()[k]
        info_sample = (1 / imp_propn) * \
                self.model.vb_params['global']['v_sticks']['info'].get()[k]

        imp_sample_integrand = lambda x : np.exp(\
            self.get_q_prior_log_ratio(x, k) + \
            np.log(u(x) + 1e-16) -
            fun_sens_lib.get_log_logitnormal_density(x, mean_sample, info_sample)) * \
            self.get_influence_operator(x, k, diffble = False)

        samples = sp.special.expit(np.random.randn(n_samples) * 1 / np.sqrt(info_sample) + \
                        mean_sample)

        return np.mean(imp_sample_integrand(samples), axis = 1)

    def influence_function(self, theta, k, diffble = True):
        # returns the influence function on the global free parameters
        # with respect to the the kth stick

        assert self.model.vb_params.use_logitnormal_sticks == True, \
            'influence function is computed for logitnormal sticks only'

        influence_operator = self.get_influence_operator(theta, k, \
                                    diffble = diffble)
        ratio = np.exp(self.get_q_prior_log_ratio(theta, k))

        if not isinstance(theta, np.ndarray):
            ratio = np.array([ratio])

        return influence_operator * ratio[None, :]

    def get_influence_operator(self, theta, k, diffble = True):
        # this is H^(-1) * grad(log q_k), the influence on the global free params
        # from the kth stick
        # the dimensions of which are n_free_params x len(theta)

        if diffble:
            log_q_pi_jac = \
                self.get_log_q_pi_jac_autodiff(\
                    self.optimal_global_free_params, theta, k)
            log_q_pi_jacT = np.rollaxis(log_q_pi_jac, \
                                        len(np.shape(log_q_pi_jac)) - 1)

            return np.einsum('ij, j...->i...', \
                                self.kl_hessian_inv, log_q_pi_jacT)

        else:
            # diffbl = False is a lot faster
            # but it gives autograd arrayboxes. So we only use it
            # for integration: i.e. when we need to compute for many samples

            # in this case, theta needs to be a vector
            # in the other case, theta could be a matrix
            assert len(np.shape(theta)) < 2
            log_q_pi_jac = self.get_log_q_pi_jac_manual(\
                                self.optimal_global_free_params, theta, k)

            return osp.linalg.cho_solve(self.kl_hessian_chol, log_q_pi_jac.T)

    def get_q_prior_log_ratio(self, theta, k):
        # this is q(theta) / p_0(theta)
        # returns a vector of length(theta)

        alpha = self.model.prior_params['alpha'].get()

        log_prior_density = lambda x : \
                fun_sens_lib.get_log_beta_prior(x, alpha) - \
                (sp.special.gammaln(alpha) - sp.special.gammaln(alpha + 1))

        log_ratio = self.get_log_q_pi(self.optimal_global_free_params, theta, k) \
                        - log_prior_density(theta)
        return log_ratio

    # def get_log_q_pi(self, global_free_params, theta, k):
    #     self.model.global_vb_params.set_free(global_free_params)
    #     # self.model.set_optimal_z()
    #
    #     mean = self.model.vb_params['global']['v_sticks']['mean'].get()[k]
    #     info = self.model.vb_params['global']['v_sticks']['info'].get()[k]
    #
    #     return fun_sens_lib.get_log_logitnormal_density(theta, mean, info)

    # ## Needs autograd version
    def get_log_q_pi(self, theta, k):
        # TODO: you need to deal with passing these extra arguments somehow.
        mean = self.model.vb_params['global']['v_sticks']['mean'].get()[k]
        info = self.model.vb_params['global']['v_sticks']['info'].get()[k]
        return fun_sens_lib.get_log_logitnormal_density(theta, mean, info)

    # functions to get the jacobian of log_q_pi faster
    # This should be tested ..
    # ## Needs autograd version
    def _get_logit_k_mean(self, global_free_params, k):
        self.model.global_vb_params.set_free(global_free_params)
        return self.model.vb_params['global']['v_sticks']['mean'].get()[k]

    def _get_logit_k_info(self, global_free_params, k):
    # ## Needs autograd version
        self.model.global_vb_params.set_free(global_free_params)
        return self.model.vb_params['global']['v_sticks']['info'].get()[k]

    def get_log_q_pi_jac_manual(self, global_free_params, theta, k):
        mean = self._get_logit_k_mean(global_free_params, k)
        info = self._get_logit_k_info(global_free_params, k)

        d_log_q_pi_mean = info * (sp.special.logit(theta) - mean)
        d_log_q_pi_info = -0.5 * (sp.special.logit(theta) - mean)**2 + \
                0.5 / info

        get_d_mean_global_params = \
            autograd.grad(self._get_logit_k_mean, argnum = 0)
        get_d_info_global_params = \
            autograd.grad(self._get_logit_k_info, argnum = 0)

        d_mean_global_params = \
            get_d_mean_global_params(global_free_params, k)
        d_info_global_params = \
            get_d_info_global_params(global_free_params, k)

        if len(np.shape(theta)) > 0:
            jac = np.zeros((len(theta), len(d_info_global_params)))
        else:
            jac = np.zeros((1, len(d_info_global_params)))

        mean_indx = np.argwhere(d_mean_global_params > 0).squeeze()
        info_indx = np.argwhere(d_info_global_params > 0).squeeze()

        jac[:, mean_indx] = \
            (d_log_q_pi_mean * d_mean_global_params[mean_indx]).squeeze()
        jac[:, info_indx] = \
            (d_log_q_pi_info * d_info_global_params[info_indx]).squeeze()
        return jac

    #########################
    # Functions to get worst case influence
    def total_influence_function(self, x, diffble = True):
        global_free_size = len(self.optimal_global_free_params)
        x_shape = np.shape(x)

        if not len(x_shape) == 0:
            jac_size = np.concatenate((np.array([global_free_size]), x_shape))
        else:
            # if theta is just a scalar
            jac_size = [global_free_size, 1]

        total_influence = np.zeros(jac_size)
        k_approx = self.model.k_approx
        for k in range(k_approx - 1):
            total_influence += self.influence_function(x, k, diffble = diffble)

        return total_influence

    def get_worst_case_perturbation_unscaled(self, g_eta, theta, sign, diffble = True):
        alpha = self.model.prior_params['alpha'].get()

        prior_density = np.expand_dims(sp.stats.beta.pdf(theta, 1, alpha), \
                                        axis = 0)

        influence = np.einsum('i, i...->...', g_eta, \
                        self.total_influence_function(theta, diffble = diffble))

        return np.maximum(sign * influence, 0)**2 * prior_density

    def get_worst_case_fun_sens(self, g_eta):
        worst_case_neg = lambda x : \
                self.get_worst_case_perturbation_unscaled(
                                        g_eta = g_eta, \
                                        theta = x, \
                                        sign = -1,
                                        diffble = False)

        worst_case_pos = lambda x : \
                self.get_worst_case_perturbation_unscaled(
                                        g_eta = g_eta, \
                                        theta = x, \
                                        sign = 1,
                                        diffble = False)

        worst_case_fun_sens_mat_pos = np.zeros(
        (len(self.optimal_global_free_params), self.model.k_approx - 1))
        worst_case_fun_sens_mat_neg = np.zeros(
            (len(self.optimal_global_free_params), self.model.k_approx - 1))

        for k in range(self.model.k_approx - 1):
            worst_case_fun_sens_mat_pos[:, k] = \
                self.get_functional_sensitivity(lambda x : worst_case_pos(x), k)

            worst_case_fun_sens_mat_neg[:, k] = \
                self.get_functional_sensitivity(lambda x : worst_case_neg(x), k)


        return worst_case_fun_sens_mat_pos, worst_case_fun_sens_mat_neg

#################################
# Functions to reload the model
#################################
def get_model_from_checkpoint(fit_dict):
    # load model from fit dictionary
    np_string = checkpoints.np_string

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
    model.weights = json_tricks.loads(fit_dict['sample_weights' + np_string])

    model.set_optimal_z()

    return model
