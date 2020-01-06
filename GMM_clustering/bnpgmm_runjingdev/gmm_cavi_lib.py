import autograd
import autograd.numpy as np
import autograd.scipy as sp

import bnpgmm_runjingdev.gmm_clustering_lib as gmm_lib
import bnpmodeling_runjingdev.modeling_lib as modeling_lib


def update_centroids(y, e_z, prior_params_dict):
    n_obs = e_z.sum(0, keepdims = True)

    centroid_sums = np.einsum('ij, ik -> ijk', y, e_z).sum(0)

    prior_lambda = prior_params_dict['prior_lambda']
    mu0 = prior_params_dict['prior_centroid_mean']

    return (prior_lambda * mu0 + centroid_sums) / (prior_lambda + n_obs)

def update_cluster_info(y, e_z, centroids, prior_params_dict):
    prior_mean = prior_params_dict['prior_centroid_mean']
    prior_lambda = prior_params_dict['prior_lambda']
    df = prior_params_dict['prior_wishart_df']
    wishart_rate = prior_params_dict['prior_wishart_rate']

    data_diff = (y[:, None, :] - centroids.transpose()[None])
    est_cov = (np.einsum('nki, nkj -> nkij', data_diff, data_diff) * \
                e_z[:, :, None, None]).sum(0)

    prior_diff = centroids - prior_mean
    prior_cov = prior_lambda * \
                    np.einsum('ik, jk -> kij', prior_diff, prior_diff)

    cov_update = (wishart_rate[None] + est_cov + prior_cov) / \
                    (df - est_cov.shape[-1] - 1 + e_z.sum(0))[:, None, None]

    info_update = np.linalg.inv(cov_update)

    return 0.5 * (info_update.transpose(0, 2, 1) + info_update)

def _get_sticks_loss(y, stick_free_params, stick_params_paragmi,
                     e_z, vb_params_dict, prior_params_dict, gh_loc, gh_weights):
    # returns the loss as a function of the stick-breaking parameters

    vb_params_dict['stick_params'] = \
        stick_params_paragmi.fold(stick_free_params, free = True)

    return gmm_lib.get_kl(y, vb_params_dict, prior_params_dict,
                            gh_loc, gh_weights,
                            e_z = e_z)

def _get_sticks_psloss(y, stick_free_params, stick_params_paragmi,
                     e_z, vb_params_dict, prior_params_dict, gh_loc, gh_weights):
    # returns a "pseudo-loss" as a function of the stick-breaking parameters:
    # that is, the terms of the loss that are a function of the stick parameters only

    vb_params_dict['stick_params'] = \
        stick_params_paragmi.fold(stick_free_params, free = True)

    stick_propn_mean = vb_params_dict['stick_params']['stick_propn_mean']
    stick_propn_info = vb_params_dict['stick_params']['stick_propn_info']

    e_loglik_ind = modeling_lib.loglik_ind(stick_propn_mean, stick_propn_info,
                            e_z, gh_loc, gh_weights)

    stick_entropy = modeling_lib.get_stick_breaking_entropy(\
                                stick_propn_mean, stick_propn_info,
                                gh_loc, gh_weights)

    alpha = prior_params_dict['alpha']
    dp_prior = \
        modeling_lib.get_e_logitnorm_dp_prior(stick_propn_mean, stick_propn_info,
                                            alpha, gh_loc, gh_weights)

    return - e_loglik_ind - dp_prior - stick_entropy


def update_sticks(y, e_z, vb_params_dict, prior_params_dict,
                   vb_params_paragami, gh_loc, gh_weights):

    # we use a logitnormal approximation to the sticks : thus, updates
    # can't be computed in closed form. We take a Newton step

    init_stick_free_param = \
        vb_params_paragami['stick_params'].flatten(vb_params_dict['stick_params'],
                                                              free = True)

    get_stick_hess = autograd.hessian(_get_sticks_loss, 1)
    get_stick_grad = autograd.elementwise_grad(_get_sticks_loss, 1)

    stick_hess = get_stick_hess(y, init_stick_free_param,
                                    vb_params_paragami['stick_params'],
                                    e_z, vb_params_dict,
                                    prior_params_dict, gh_loc, gh_weights)

    stick_grad = get_stick_grad(y, init_stick_free_param,
                                vb_params_paragami['stick_params'],
                                    e_z, vb_params_dict, prior_params_dict,
                                    gh_loc, gh_weights)

    update_stick_free_param = init_stick_free_param - \
                                np.linalg.solve(stick_hess, stick_grad)

    return vb_params_paragami['stick_params'].fold(update_stick_free_param, free = True)

def run_cavi(y, vb_params_dict,
                    vb_params_paragami,
                    prior_params_dict,
                    gh_loc, gh_weights,
                    max_iter = 100,
                    tol = 1e-8,
                    debug = False):

    kl_vec = np.zeros(max_iter)

    diff = 1e16
    kl_old = 1e16

    for i in range(max_iter):
        # update e_z
        e_z = gmm_lib.get_optimal_z_from_vb_params_dict(y, vb_params_dict,
                                                        gh_loc, gh_weights)
        if debug:
            kl_new = gmm_lib.get_kl(y, vb_params_dict, prior_params_dict,
                                gh_loc, gh_weights,
                                e_z = e_z)
            assert kl_new < kl_old, 'e_z update failed'
            kl_old = kl_new


        # update centroids
        # take step
        vb_params_dict['cluster_params']['centroids'] = \
            update_centroids(y, e_z, prior_params_dict)

        if debug:
            kl_new = gmm_lib.get_kl(y, vb_params_dict, prior_params_dict,
                                gh_loc, gh_weights,
                                e_z = e_z)
            assert kl_new < kl_old, 'centroid update failed'
            kl_old = kl_new

        # update cluster info
        vb_params_dict['cluster_params']['cluster_info'] = \
            update_cluster_info(y, e_z,
                                vb_params_dict['cluster_params']['centroids'],
                                prior_params_dict)

        if debug:
            kl_new = gmm_lib.get_kl(y, vb_params_dict, prior_params_dict,
                                gh_loc, gh_weights,
                                e_z = e_z)
            assert kl_new < kl_old, 'cluster info update failed'
            kl_old = kl_new


        # update sticks
        vb_params_dict['stick_params'] = \
            update_sticks(y, e_z, vb_params_dict,
                                    prior_params_dict,
                                    vb_params_paragami,
                                    gh_loc, gh_weights)

        if debug:
            kl_new = gmm_lib.get_kl(y, vb_params_dict, prior_params_dict,
                                gh_loc, gh_weights,
                                e_z = e_z)
            assert kl_new < kl_old, 'stick update failed; diff = {}'.format(kl_new - kl_old)
            kl_old = kl_new

        # get loss
        kl_vec[i] = gmm_lib.get_kl(y, vb_params_dict, prior_params_dict,
                            gh_loc, gh_weights,
                            e_z = e_z)

        if i > 0:
            diff = kl_vec[i] - kl_vec[i-1]
            assert diff <= 0

        if np.abs(diff) < tol:
            print('done. num iterations = {}'.format(i))
            break

    # get e_z optima
    e_z = gmm_lib.get_optimal_z_from_vb_params_dict(y, vb_params_dict,
                                                    gh_loc, gh_weights)

    return vb_params_dict, e_z, kl_vec[0:i]
