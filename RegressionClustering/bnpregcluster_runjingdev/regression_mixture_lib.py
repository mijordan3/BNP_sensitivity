"""Functions for estimaing a discrete mixture of regressions.
"""

import LinearResponseVariationalBayes.ExponentialFamilies as ef

import autograd
import autograd.numpy as np
import autograd.scipy as sp

from numpy.polynomial.hermite import hermgauss

import paragami

from copy import deepcopy
import json_tricks
import json

import scipy as osp
from sklearn.cluster import KMeans

from aistats2019_ij_paper import regression_mixture_lib as gmm_lib
from bnpmodeling_runjingdev import modeling_lib as bnp_modeling_lib

import time

from aistats2019_ij_paper.regression_lib import get_regression_array_pattern
from aistats2019_ij_paper.regression_mixture_lib import get_prior_params_pattern
from aistats2019_ij_paper.regression_mixture_lib import get_base_prior_params
#from aistats2019_ij_paper.regression_mixture_lib import get_log_lik_nk

def get_gmm_params_pattern(obs_dim, num_components):
    """A ``paragami`` pattern for a mixture model.

    ``centroids`` are the locations of the clusters.
    ``stick_propn_mean`` is the variational mean of the logit-space stick
        proportions.
    ``stick_propn_info`` is the variational inverse variatiance of logit-space
        stick proportions.
    """
    gmm_params_pattern = paragami.PatternDict()
    gmm_params_pattern['centroids'] = \
        paragami.NumericArrayPattern((num_components, obs_dim))
    # Old:
    #``probs`` are the a priori probabilities of each cluster.
    # gmm_params_pattern['probs'] = \
    #     paragami.SimplexArrayPattern(
    #         simplex_size=num_components, array_shape=(1,))
    gmm_params_pattern['stick_propn_mean'] = \
        paragami.NumericArrayPattern(shape = (num_components - 1,))
    gmm_params_pattern['stick_propn_info'] = \
        paragami.NumericArrayPattern(shape = (num_components - 1,), lb = 1e-4)
    return gmm_params_pattern


def get_log_prior(centroids, stick_propn_mean, stick_propn_info,
                  gh_loc, gh_weights,
                  prior_params):
    num_components = centroids.shape[0]
    obs_dim = centroids.shape[1]

    log_prior = 0
    #log_probs = np.log(probs[0, :])
    #log_prior += ef.dirichlet_prior(prior_params['probs_alpha'], log_probs)

    log_prior += \
        bnp_modeling_lib.get_e_logitnorm_dp_prior(
            stick_propn_mean, stick_propn_info,
            prior_params['probs_alpha'], gh_loc, gh_weights)

    for k in range(num_components):
        log_prior += ef.mvn_prior(
            prior_params['centroid_prior_mean'][k, :],
            prior_params['centroid_prior_info'],
            centroids[k, :],
            np.zeros((obs_dim, obs_dim)))

    return(log_prior)


from aistats2019_ij_paper.regression_mixture_lib import get_e_z
from aistats2019_ij_paper.regression_mixture_lib import get_kl

from aistats2019_ij_paper.regression_mixture_lib import kmeans_init
from aistats2019_ij_paper.regression_mixture_lib import wrap_get_loglik_terms
from aistats2019_ij_paper.regression_mixture_lib import wrap_get_e_z

# Likelihoods

def get_loglik_obs_by_nk(gmm_params, reg_params):
    """ Evaluate a matrix of log P(x_n | centroid_k, x_info_n) where
    x ~ N(centroid_k , 1 / x_info_n), without the normalization constant.
    """
    centroids = gmm_params['centroids']
    x = reg_params['beta_mean']
    x_infos = reg_params['beta_info']

    loglik_obs_by_nk = \
        -0.5 * (-2 * np.einsum('ni,kj,nij->nk', x, centroids, x_infos) +
                np.einsum('ki,kj,nij->nk', centroids, centroids, x_infos))
    return loglik_obs_by_nk


def get_z_nat_params(gmm_params, reg_params, gh_loc, gh_weights):
    """Natural parameters for an optimal z step.  Factorized out for testing.
    """
    loglik_obs_by_nk = get_loglik_obs_by_nk(gmm_params, reg_params)
    e_log_cluster_probs = \
        bnp_modeling_lib.get_e_log_cluster_probabilities(
            gmm_params['stick_propn_mean'], gmm_params['stick_propn_info'],
            gh_loc, gh_weights)

    # So called because it is the optimal natural parameter for a z update.
    z_nat_param = loglik_obs_by_nk + e_log_cluster_probs

    return z_nat_param


def get_loglik_terms(gmm_params, reg_params, gh_loc, gh_weights):
    z_nat_param = \
        get_z_nat_params(gmm_params, reg_params, gh_loc, gh_weights)

    log_const = sp.special.logsumexp(z_nat_param, axis=1)
    e_z = np.exp(z_nat_param - log_const[:, None])

    return z_nat_param, e_z


def get_entropy(gmm_params, e_z, gh_loc, gh_weights):
    z_entropy = bnp_modeling_lib.multinom_entropy(e_z)
    stick_entropy = \
        bnp_modeling_lib.get_stick_breaking_entropy(
            gmm_params['stick_propn_mean'],
            gmm_params['stick_propn_info'],
            gh_loc, gh_weights)

    return z_entropy + stick_entropy


def _collapse(x):
    # Convert a single-element array to a number.
    return np.atleast_1d(x)[0]


def get_kl(gmm_params, reg_params, prior_params, gh_loc, gh_weights):
    z_nat_param, e_z = \
        get_loglik_terms(gmm_params, reg_params, gh_loc, gh_weights)

    log_prior = get_log_prior(
        gmm_params['centroids'],
        gmm_params['stick_propn_mean'],
        gmm_params['stick_propn_info'],
        gh_loc, gh_weights,
        prior_params)

    loglik = _collapse(np.sum(z_nat_param * e_z) + log_prior)
    assert(np.isfinite(loglik))

    entropy = _collapse(get_entropy(gmm_params, e_z, gh_loc, gh_weights))
    assert(np.isfinite(entropy))

    return -1 * (entropy + loglik)


def backtrack(fun, theta0, v0, alpha=0.5, max_tries=10, tol=1e-7):
    """Backtracking search for a descent direction.  Use to take a
    safe newton step.
    """
    if alpha <= 0 or alpha >= 1:
        raise ValueError('alpha must be in (0, 1)')

    i = 0
    v = v0
    f0 = fun(theta0)
    while i < max_tries:
        fv = fun(theta0 + v)
        print(f0 - fv)
        if not np.isnan(fv):
            if fun(theta0 + v) < f0 - tol:
                print('Success.')
                success = True
                return theta0 + v, success
        print('Shrinking step.')
        v *= alpha
        if np.max(np.abs(v)) < tol:
            print('Under tolerance.')
            success = False
            return theta0, success
        i += 1

    print('Maximum tries reached.')
    success = False
    return theta0, success


##################
# Mixture model class

class GMM(gmm_lib.GMM):
    def __init__(self, num_components, prior_params, reg_params, gh_deg=8):
        """A class for estimating mixtures of regressions.

        Parameters
        ------------
        num_components : `int`
            The number of components in the mixture.
        prior_params
            A prior parameters object.  See ``get_base_prior_params``.
        reg_params
            A dictionary of observations to be clustered containing elements
            ``beta_mean`` and ``beta_info``, which are arrays of observation
            means and information matrices respectively.  See
            ``get_regression_array_pattern`` for more details.

        """

        # Model for regression output.
        self.num_components = num_components
        self.reg_params = reg_params

        self.gh_loc, self.gh_weights = hermgauss(gh_deg)

        self.obs_dim = self.reg_params['beta_mean'].shape[1]
        self.num_obs = self.reg_params['beta_mean'].shape[0]
        if not self.reg_params['beta_info'].shape[0] == self.num_obs:
            raise ValueError('Wrong number of observations for beta_info')

        if not (self.reg_params['beta_info'].shape[1] == self.obs_dim and
                self.reg_params['beta_info'].shape[2] == self.obs_dim):
            raise ValueError('Wrong shape for beta_info')

        self.gmm_params_pattern = get_gmm_params_pattern(
            obs_dim=self.obs_dim,
            num_components=self.num_components)

        self.prior_params = prior_params

        ############################
        # Set up the objectives
        self.get_params_kl_flat = paragami.FlattenFunctionInput(
            self.get_params_kl,
            free=True,
            patterns=self.gmm_params_pattern)

        # This objective is only used for calculating derivatives.
        self.kl_obj = paragami.OptimizationObjective(
            self.get_params_kl_flat)

        # We actually optimize a preconditioned objective.
        self.get_kl_conditioned = \
            paragami.PreconditionedFunction(self.get_params_kl_flat)
        self.initialize_preconditioner()
        self.conditioned_obj = paragami.OptimizationObjective(
            self.get_kl_conditioned)

    def transform_regression_parameters(self):
        raise NotImplementedError

    def set_regression_params(self):
        raise NotImplementedError

    def get_reg_params_kl(self):
        raise NotImplementedError

    def get_reg_params_kl(self):
        raise NotImplementedError

    def get_params_kl(self, gmm_params):
        """Get the optimization objective as a function of the mixture
        parameters.
        """
        return get_kl(
            gmm_params, self.reg_params, self.prior_params,
            self.gh_loc, self.gh_weights)

    def get_e_z(self, gmm_params):
        _, e_z = get_loglik_terms(
            gmm_params, self.reg_params, self.gh_loc, self.gh_weights)
        return e_z


    def optimize_fully(self, init_x,
                       x_tol=1e-7, f_tol=1e-7, g_tol=1e-7, maxiter=500,
                       kl_hess=None, max_num_restarts=5, verbose=False):
        """Optimize, repeatedly re-calculating the preconditioner,
        until the optimal parameters don't change more than ``x_tol``.

        Returns
        ------------
        opt_results
            A dictionary containing a summary of the optimization
        new_x : `numpy.ndarray`
            The optimal free flat value of the mixture parameters.
        """
        def verbose_print(s):
            if verbose:
                print(s)

        # Reset the logging.
        self.conditioned_obj.reset()

        if kl_hess is None:
            kl_hess = self.update_preconditioner(init_x)

        last_x = None
        last_f = None
        last_g = None
        def log_iteration(x, f, g):
            nonlocal last_x
            nonlocal last_f
            nonlocal last_g
            last_x = deepcopy(x)
            last_f = f
            last_g = deepcopy(g)

        log_iteration(init_x, self.kl_obj.f(init_x), self.kl_obj.grad(init_x))

        def check_converged(it, x, f, g):
            """ iteration, argument, function value, gradient

            Returns terminate, success
            """
            if np.mean(np.abs(x - last_x)) < x_tol:
                verbose_print('x convergence.')
                return True, True

            if last_f - f < f_tol:
                verbose_print('f convergence.')
                return True, True

            if np.mean(np.abs(g - last_g)) < g_tol:
                verbose_print('g convergence.')
                return True, True

            if it > max_num_restarts:
                return True, False

            return False, False

        restart_num = 1
        terminate = False
        success = None
        while not terminate:
            verbose_print('Preconditioned iteration {}'.format(restart_num))

            # On the first round, use the existing preconditioner.
            # After that re-calculate the preconditioner using the last
            # optimal value.
            if restart_num > 1:
                verbose_print('  Getting Hessian and preconditioner.')
                kl_hess = self.update_preconditioner(last_x)

            verbose_print('  Taking Newton step.')
            print(kl_hess.shape)
            print(last_g.shape)
            newton_step = -1 * np.linalg.solve(kl_hess, last_g)
            if np.mean(np.abs(newton_step)) > g_tol:
                newton_x, _ = backtrack(self.kl_obj.f, last_x, newton_step)
                verbose_print('  Running preconditioned optimization.')
                gmm_opt_cond, new_x = self.optimize(
                    newton_x, maxiter=maxiter, gtol=g_tol)

                new_f = self.kl_obj.f(new_x)
                new_g = self.kl_obj.grad(new_x)
                restart_num += 1
                terminate, success = \
                    check_converged(restart_num, new_x, new_f, new_g)
                log_iteration(new_x, new_f, new_g)
            else:
                verbose_print('  Converging with small Newton step.')
                terminate = True
                success = True

        opt_results = dict()
        opt_results['gmm_opt_cond'] = gmm_opt_cond
        opt_results['converged'] = success
        opt_results['kl_hess'] = kl_hess

        return opt_results, new_x


    def to_json(self):
        gmm_dict = dict()
        gmm_dict['num_components'] = self.num_components

        gmm_dict['obs_dim'] = self.obs_dim
        gmm_dict['num_components'] = self.num_components
        prior_params_pattern = \
            get_prior_params_pattern(
                self.obs_dim, self.num_components)
        gmm_dict['prior_params_flat'] = \
            list(prior_params_pattern.flatten(
                 self.prior_params, free=False))

        gmm_dict['beta_mean'] = self.reg_params['beta_mean']
        gmm_dict['beta_info'] = self.reg_params['beta_info']

        hess_dim = self.gmm_params_pattern.flat_length(free=True)
        gmm_dict['preconditioner_json'] = \
            json_tricks.dumps(
                self.get_kl_conditioned.get_preconditioner(hess_dim))

        return json.dumps(gmm_dict)

    @classmethod
    def from_json(cls, json_str, regs, reg_params):
        """Instantiate a gmm object from a json object.
        """
        gmm_dict = json.loads(json_str)

        prior_params_pattern = \
            get_prior_params_pattern(
                gmm_dict['obs_dim'], gmm_dict['num_components'])
        prior_params = prior_params_pattern.fold(
            gmm_dict['prior_params_flat'], free=False)

        reg_params = { 'beta_mean': gmm_dict['beta_mean'],
                       'beta_info': gmm_dict['beta_info']}
        gmm = cls(num_components=gmm_dict['num_components'],
                  prior_params=prior_params,
                  reg_params=reg_params)

        preconditioner = json_tricks.loads(gmm_dict['preconditioner_json'])
        gmm.get_kl_conditioned.set_preconditioner_matrix(preconditioner)

        return gmm
