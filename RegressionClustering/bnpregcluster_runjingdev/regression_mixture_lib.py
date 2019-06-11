"""Functions for estimaing a discrete mixture of regressions.
"""

import LinearResponseVariationalBayes.ExponentialFamilies as ef

import autograd
import autograd.numpy as np
import autograd.scipy as sp

import paragami

from copy import deepcopy
import json_tricks
import json

import scipy as osp
from sklearn.cluster import KMeans

from aistats2019_ij_paper import regression_mixture_lib as gmm_lib
# from . import regression_lib as reg_lib
# from . import transform_regression_lib as trans_reg_lib

import time

from aistats2019_ij_paper.regression_lib import get_regression_array_pattern

from aistats2019_ij_paper.regression_mixture_lib import get_gmm_params_pattern
from aistats2019_ij_paper.regression_mixture_lib import get_prior_params_pattern
from aistats2019_ij_paper.regression_mixture_lib import get_base_prior_params
from aistats2019_ij_paper.regression_mixture_lib import get_log_lik_nk


def get_log_prior(centroids, probs, prior_params):
    num_components = centroids.shape[0]
    obs_dim = centroids.shape[1]

    log_prior = 0
    log_probs = np.log(probs[0, :])
    #log_prior += ef.dirichlet_prior(prior_params['probs_alpha'], log_probs)

    assert False
    # TODO: BNP prior here

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

def wrap_get_kl(gmm_params, reg_params, prior_params):
    log_lik_by_nk, e_z = \
        wrap_get_loglik_terms(gmm_params, reg_params)

    # Use the BNP log prior function
    log_prior = get_log_prior(
        gmm_params['centroids'], gmm_params['probs'], prior_params)
    return get_kl(log_lik_by_nk, e_z, log_prior)


##################
# Mixture model class

class GMM(object):
    def __init__(self, num_components, prior_params, reg_params):
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

        # # Terms that affect how the regression covariance is used
        # # in clustering.
        # self.inflate_coef_cov = inflate_coef_cov
        # self.cov_regularization = cov_regularization
        #
        # # Set up the transformed parameters.  These are what we actually
        # # cluster.
        # if (transform_mat is not None) and (unrotate_transform_mat is not None):
        #     self.transform_mat = transform_mat
        #     self.unrotate_transform_mat = unrotate_transform_mat
        #     if self.transform_mat.shape[1] != regs.x.shape[1]:
        #         raise ValueError(
        #             'transform_mat has the wrong number of columns.')
        #     if self.unrotate_transform_mat.shape[1] != transform_mat.shape[0]:
        #         raise ValueError(
        #             'unrotate_transform_mat has the wrong number of columns.')
        #     if self.unrotate_transform_mat.shape[0] != regs.x.shape[0]:
        #         raise ValueError(
        #             'unrotate_transform_mat has the wrong number of rows.')
        # else:
        #     if (transform_mat is not None) or \
        #        (unrotate_transform_mat is not None):
        #        raise ValueError(
        #         '``transform_mat`` and ``unrotate_transform_mat`` ' +
        #         'must be both either ``None`` or not ``None``.')
        #     self.transform_mat, self.unrotate_transform_mat = \
        #         trans_reg_lib.get_reversible_predict_and_demean_matrix(
        #             regs.x)
        #
        # self.obs_dim = self.transform_mat.shape[0]

        # self.set_regression_params(reg_params)

        self.reg_params = reg_params

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

        # This is for the cross Hessian between the regression and gmm.
        # self.get_reg_params_kl_flat = paragami.FlattenFunctionInput(
        #     self.get_reg_params_kl,
        #     free=True,
        #     patterns=[self.gmm_params_pattern, regs.reg_params_pattern])
        # self.get_reg_params_kl_flat_grad = autograd.grad(
        #     self.get_reg_params_kl_flat, argnum=0)
        # self.get_reg_params_kl_flat_cross_hess = autograd.jacobian(
        #     self.get_reg_params_kl_flat_grad, argnum=1)

    def initialize_preconditioner(self):
        self.get_kl_conditioned.set_preconditioner_matrix(
            np.eye(self.gmm_params_pattern.flat_length(free=True)))

    def update_preconditioner(self, gmm_params_free, ev_min=1e-4):
        """Set the optimization preconditioner using the objective
        Hessian at ``gmm_params_free``.
        """
        hessian = self.kl_obj.hessian(gmm_params_free)
        self.get_kl_conditioned.set_preconditioner_with_hessian(
            hessian=hessian, ev_min=ev_min)
        return hessian

    # def transform_regression_parameters(self, reg_params):
    #     """Transform the regression parameters ``reg_params`` into the
    #     space being clustered.
    #     """
    #     return transform_regression_parameters(
    #         reg_params, self.transform_mat,
    #         cov_regularization=self.cov_regularization,
    #         inflate_coef_cov=self.inflate_coef_cov)

    # def set_regression_params(self, reg_params):
    #     """Set the class's ``transformed_reg_params`` from ``reg_params``.
    #     It is this values in ``transformed_reg_params`` that are clustered.
    #     """
    #     self.transformed_reg_params = \
    #         self.transform_regression_parameters(reg_params)
    #     self.num_obs = self.transformed_reg_params['beta_mean'].shape[0]

    # def get_reg_params_kl(self, gmm_params, reg_params):
    #     """Get the optimization objective as a function of both the mixture
    #     and regression parameters.
    #     """
    #     # In order to avoid ArrayBoxes, don't set
    #     # ``self.tranform_reg_params`` here.
    #     transformed_reg_params = \
    #         self.transform_regression_parameters(reg_params)
    #     return wrap_get_kl(
    #         gmm_params, transformed_reg_params, self.prior_params)

    def get_params_kl(self, gmm_params):
        """Get the optimization objective as a function of the mixture
        parameters.
        """
        return wrap_get_kl(
            gmm_params, self.reg_params, self.prior_params)

    def optimize(self, init_x, maxiter=500, gtol=1e-6):
        """Optimize the preconditioned objective starting at ``init_x``,
        ``init_x`` should be a flat free value of the gmm parameters.
        """
        init_x_cond = self.get_kl_conditioned.precondition(init_x)
        gmm_opt_cond = osp.optimize.minimize(
            self.conditioned_obj.f,
            x0=init_x_cond,
            jac=self.conditioned_obj.grad,
            hessp=self.conditioned_obj.hessian_vector_product,
            method='trust-ncg',
            options={'maxiter': maxiter, 'gtol': gtol})

        # Remember that gmm_opt_cond.x is in the conditioned space..
        return \
            gmm_opt_cond, \
            self.get_kl_conditioned.unprecondition(gmm_opt_cond.x)

    def optimize_fully(self, init_x,
                       x_tol=1e-6, maxiter=500, gtol=1e-6,
                       max_num_restarts=4, verbose=False):
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

        last_x = deepcopy(init_x)
        kl_hess = None

        restart_num = 1
        x_diff = float('inf')
        while x_diff > x_tol and restart_num <= max_num_restarts:
            verbose_print('Preconditioned iteration {}'.format(restart_num))

            # On the first round, use the existing preconditioner.
            # After that re-calculate the preconditioner using the last
            # optimal value.
            if restart_num > 1:
                verbose_print('  Getting Hessian and preconditioner.')
                kl_hess = self.update_preconditioner(last_x)

            verbose_print('  Running preconditioned optimization.')
            gmm_opt_cond, new_x = self.optimize(
                last_x, maxiter=maxiter, gtol=gtol)

            restart_num += 1
            x_diff = np.linalg.norm(new_x - last_x)
            last_x = deepcopy(new_x)

        if x_diff <= x_tol:
            verbose_print('Converged.')
        else:
            verbose_print(
                'Terminated after {} iterations without converging.'.format(
                    max_num_restarts))

        opt_results = dict()
        opt_results['gmm_opt_cond'] = gmm_opt_cond
        opt_results['x_diff'] = x_diff
        opt_results['converged'] = x_diff <= x_tol
        opt_results['kl_hess'] = kl_hess

        return opt_results, new_x

    def to_json(self):
        gmm_dict = dict()
        gmm_dict['num_components'] = self.num_components
        gmm_dict['inflate_coef_cov'] = self.inflate_coef_cov
        gmm_dict['cov_regularization'] = self.cov_regularization

        gmm_dict['obs_dim'] = self.obs_dim
        gmm_dict['num_components'] = self.num_components
        prior_params_pattern = \
            get_prior_params_pattern(
                self.obs_dim, self.num_components)
        gmm_dict['prior_params_flat'] = \
            list(prior_params_pattern.flatten(
                 self.prior_params, free=False))

        hess_dim = self.gmm_params_pattern.flat_length(free=True)
        gmm_dict['preconditioner_json'] = \
            json_tricks.dumps(
                self.get_kl_conditioned.get_preconditioner(hess_dim))

        # Because the svd is not unique, save these to make sure they
        # are the same.
        gmm_dict['transform_mat_json'] = \
            json_tricks.dumps(self.transform_mat)
        gmm_dict['unrotate_transform_mat_json'] = \
            json_tricks.dumps(self.unrotate_transform_mat)

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

        transform_mat = json_tricks.loads(
            gmm_dict['transform_mat_json'])
        unrotate_transform_mat = json_tricks.loads(
            gmm_dict['unrotate_transform_mat_json'])

        gmm = cls(num_components=gmm_dict['num_components'],
                  prior_params=prior_params,
                  regs=regs,
                  reg_params=reg_params,
                  transform_mat=transform_mat,
                  unrotate_transform_mat=unrotate_transform_mat,
                  cov_regularization=gmm_dict['cov_regularization'],
                  inflate_coef_cov=gmm_dict['inflate_coef_cov'])

        preconditioner = json_tricks.loads(gmm_dict['preconditioner_json'])
        gmm.get_kl_conditioned.set_preconditioner_matrix(preconditioner)

        return gmm


# def draw_time_weights(regs, w_sum=None):
#     if w_sum is None:
#         w_sum = regs.y_obs_dim
#     w_draw = osp.stats.multinomial.rvs(
#         n=w_sum, p=np.full(regs.y_obs_dim, 1 / regs.y_obs_dim))
#     return w_draw
#
# def refit_with_time_weights(
#         gmm, regs, time_w, gmm_init_par, **opt_args):
#
#     if len(time_w) != len(regs.time_w):
#         raise ValueError('Time weights are the wrong length.')
#     regs.time_w = deepcopy(time_w)
#     opt_reg_params = regs.get_optimal_regression_params()
#     gmm.set_regression_params(opt_reg_params)
#     gmm_opt_dict, gmm_opt_x = gmm.optimize_fully(
#         init_x=gmm_init_par, **opt_args)
#
#     return gmm_opt_dict, gmm_opt_x, opt_reg_params
