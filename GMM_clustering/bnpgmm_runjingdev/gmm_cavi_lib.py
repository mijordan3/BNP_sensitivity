import jax
import jax.numpy as np
import jax.scipy as sp

import bnpgmm_runjingdev.gmm_clustering_lib as gmm_lib

import bnpmodeling_runjingdev.modeling_lib as modeling_lib

import time

from copy import deepcopy


update_z = jax.jit(gmm_lib.get_optimal_z_from_vb_dict)

@jax.jit
def update_centroids(y, e_z, prior_params_dict):
    n_obs = e_z.sum(0, keepdims = True)

    centroid_sums = np.einsum('ij, ik -> ijk', y, e_z).sum(0)

    prior_lambda = prior_params_dict['prior_lambda']
    mu0 = prior_params_dict['prior_centroid_mean']

    return (prior_lambda * mu0 + centroid_sums) / (prior_lambda + n_obs)

@jax.jit
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

    return 0.5 * (info_update.transpose((0, 2, 1)) + info_update)

def _get_sticks_psloss(y, stick_free_params, stick_params_paragmi,
                     e_z, prior_params_dict,
                     gh_loc, gh_weights,
                     e_log_phi = None):

    # returns a "pseudo-loss" as a function of the stick-breaking parameters:
    # that is, the terms of the loss that are a function of the stick parameters only

    stick_params_dict = \
        stick_params_paragmi.fold(stick_free_params, free = True)

    stick_means = stick_params_dict['stick_means']
    stick_infos = stick_params_dict['stick_infos']

    e_loglik_ind = modeling_lib.loglik_ind(stick_means, stick_infos,
                            e_z, gh_loc, gh_weights)

    stick_entropy = modeling_lib.get_stick_breaking_entropy(\
                                stick_means, stick_infos,
                                gh_loc, gh_weights)

    alpha = prior_params_dict['alpha']
    dp_prior = \
        modeling_lib.get_e_logitnorm_dp_prior(stick_means, stick_infos,
                                            alpha, gh_loc, gh_weights).squeeze()
    if e_log_phi is not None:
        e_log_pert = e_log_phi(stick_means, stick_infos)
    else:
        e_log_pert = 0.0

    return - e_loglik_ind - dp_prior - stick_entropy - e_log_pert


def update_sticks(stick_obj_fun, stick_grad_fun, e_z,
                    stick_param_dict, stick_params_paragami):

    # we use a logitnormal approximation to the sticks : thus, updates
    # can't be computed in closed form. We take a gradient step satisfying wolfe conditions

    # initial parameters
    init_stick_free_param = \
        stick_params_paragami.flatten(stick_param_dict, free = True)

    # initial loss
    init_ps_loss = stick_obj_fun(init_stick_free_param, e_z)

    # get gradient
    stick_grad = stick_grad_fun(init_stick_free_param, e_z)

    # direction of step
    step = - stick_grad

    # choose stepsize
    kl_new = 1e16
    counter = 0.
    rho = 0.5
    alpha = 1.0 / rho
    correction = np.sum(stick_grad * step)
    # for my sanity
    assert correction < 0
    while (kl_new > (init_ps_loss + 1e-4 * alpha * correction)):
        alpha *= rho

        update_stick_free_param = init_stick_free_param + alpha * step

        kl_new = stick_obj_fun(update_stick_free_param, e_z)
        counter += 1

        if counter > 10:
            print('could not find stepsize for stick optimizer')
            break

    return stick_params_paragami.fold(update_stick_free_param,
                                                    free = True)

def run_cavi(y, vb_params_dict,
                    vb_params_paragami,
                    prior_params_dict,
                    gh_loc, gh_weights,
                    e_log_phi = None,
                    max_iter = 1000,
                    x_tol = 1e-3,
                    debug = False):
    # runs coordinate ascent in a gmm model

    x_old = 1e16
    kl_old = 1e16

    e_z_time = 0.0
    cluster_time = 0.0
    stick_time = 0.0

    n_obs = y.shape[0]

    # jit the stick objective and gradient
    # used to update the logitnormal sticks
    stick_params_paragmi = vb_params_paragami['stick_params']
    stick_obj_fun = lambda stick_free_params, e_z : \
                        _get_sticks_psloss(y, stick_free_params, stick_params_paragmi,
                                             e_z, prior_params_dict,
                                             gh_loc, gh_weights,
                                             e_log_phi)

    stick_obj_fun_jit = jax.jit(stick_obj_fun)
    stick_grad_fun = jax.jit(jax.grad(stick_obj_fun, 0))

    # jit the flatten computation  ...
    # this is actually slow otherwise
    flatten_vb_params = lambda vb_params_dict : \
                            vb_params_paragami.flatten(vb_params_dict,
                                                        free = True,
                                                        # otherwise wont work with jit
                                                        validate_value=False)
    flatten_vb_params = jax.jit(flatten_vb_params)

    success = False

    # Compile cavi functions
    stick_free_params = vb_params_paragami['stick_params'].flatten(\
                                vb_params_dict['stick_params'], free = True)
    compile_cav_updates(stick_obj_fun_jit, stick_grad_fun, flatten_vb_params,
                        y, vb_params_dict, prior_params_dict, stick_free_params,
                        gh_loc, gh_weights)

    print('\nRunning CAVI ... ')
    time0 = time.time()
    for i in range(max_iter):
        # update e_z
        t0 = time.time()
        e_z = update_z(y, vb_params_dict,
                                                        gh_loc, gh_weights)
        e_z_time += time.time() - t0
        if debug:
            kl_new = gmm_lib.get_kl(y, vb_params_dict, prior_params_dict,
                                gh_loc, gh_weights,
                                e_z = e_z)
            assert kl_new <= kl_old, 'e_z update failed'
            kl_old = kl_new

        # update centroids
        # take step
        t0 = time.time()
        vb_params_dict['cluster_params']['centroids'] = \
            update_centroids(y, e_z, prior_params_dict)

        if debug:
            kl_new = gmm_lib.get_kl(y, vb_params_dict, prior_params_dict,
                                gh_loc, gh_weights,
                                e_z = e_z)
            assert kl_new <= kl_old, 'centroid update failed'
            kl_old = kl_new

        # update cluster info
        vb_params_dict['cluster_params']['cluster_info'] = \
            update_cluster_info(y, e_z,
                                vb_params_dict['cluster_params']['centroids'],
                                prior_params_dict)

        cluster_time += time.time() - t0
        if debug:
            kl_new = gmm_lib.get_kl(y, vb_params_dict, prior_params_dict,
                                gh_loc, gh_weights,
                                e_z = e_z)
            assert kl_new <= kl_old, 'cluster info update failed'
            kl_old = kl_new


        # update sticks
        t0 = time.time()
        vb_params_dict['stick_params'] = \
            update_sticks(stick_obj_fun_jit, stick_grad_fun, e_z,
                                vb_params_dict['stick_params'],
                                vb_params_paragami['stick_params'])

        stick_time += time.time() - t0
        if debug:
            kl_new = gmm_lib.get_kl(y, vb_params_dict, prior_params_dict,
                                gh_loc, gh_weights,
                                e_z = e_z)
            assert kl_new <= kl_old, 'stick update failed; diff = {}'.format(kl_new - kl_old)
            kl_old = kl_new

        x_new = flatten_vb_params(vb_params_dict)
        diff = np.abs(x_new - x_old).max()
        if diff < x_tol:
            print('done. num iterations = {}'.format(i))
            success = True
            break
        else:
            x_old = x_new

    if not success:
        print('warning, maximum iterations reached')

    # get e_z optima
    # block for timing results
    e_z = update_z(y, vb_params_dict, gh_loc, gh_weights).block_until_ready()

    print('stick_time: {0:.3g}sec'.format(stick_time))
    print('cluster_time: {0:.3g}sec'.format(cluster_time))
    print('e_z_time: {0:.3g}sec'.format(e_z_time))
    cavi_time = time.time() - time0
    print('**CAVI time: {0:.3g}sec**'.format(cavi_time))

    return vb_params_dict, e_z, cavi_time

def compile_cav_updates(stick_obj_fun, stick_grad_fun, flatten_vb_params,
                            y, vb_params_dict, prior_params_dict, stick_free_params,
                            gh_loc, gh_weights):

    print('Compiling CAVI update functions ... ')
    t0 = time.time()

    # compile z-update
    e_z = update_z(y, vb_params_dict,
                                                    gh_loc, gh_weights)

    # compile centroid updates
    _ = update_centroids(y, e_z, prior_params_dict)

    # compile flatten function
    _ = flatten_vb_params(vb_params_dict)

    # compile stick objective an dfunctions
    _ = stick_obj_fun(stick_free_params, e_z)
    _ = stick_grad_fun(stick_free_params, e_z)

    print('CAVI compile time: {0:.3g}sec'.format(time.time() - t0))
