import autograd
import paragami
import time

import autograd.scipy as sp
import autograd.numpy as np

from autograd.extend import primitive, defvjp, defjvp

import bnpgmm_runjingdev.gmm_clustering_lib as gmm_lib

from copy import deepcopy

################
# Functions to compute a block of the Hessian
# corresponding to the k-largest clusters

@primitive
def replace(x_sub, x, inds):
    x_new = np.full(x.shape, float('nan'))
    x_new[:] = x
    x_new[inds] = x_sub
    return x_new

defvjp(replace,
       lambda ans, x_sub, x, inds: lambda g: g[inds],
       lambda ans, x_sub, x, inds: lambda g: replace(0, g, inds),
       None)

defjvp(replace,
       lambda g, ans, x_sub, x, inds: replace(g, np.zeros(len(x)), inds),
       lambda g, ans, x_sub, x, inds: replace(0, g, inds),
       None)

def get_subvb_params(which_k, vb_opt, vb_params_paragami):
    bool_dict = vb_params_paragami.empty_bool(False)

    dim = bool_dict['cluster_params']['cluster_info'].shape[-1]
    k_approx = bool_dict['cluster_params']['cluster_info'].shape[0]

    bool_dict['cluster_params']['centroids'][:, which_k] = True
    bool_dict['cluster_params']['cluster_info'][which_k] = True

    # last stick is deterministic; one less stick param than k_approx
    if(np.any(which_k == (k_approx - 1))):
        _which_k = which_k[which_k != (k_approx - 1)]
    else:
        _which_k = which_k[:-1]

    bool_dict['stick_params']['stick_propn_mean'][_which_k] = True
    bool_dict['stick_params']['stick_propn_info'][_which_k] = True

    # free indices corresponding to sub vb parameters
    indx = vb_params_paragami.flat_indices(bool_dict, free = True)

    # get paragami and dictionary for sub vb parameters
    _, sub_vb_params_paragami = \
        gmm_lib.get_vb_params_paragami_object(dim, len(which_k))
    sub_vb_params_dict = sub_vb_params_paragami.fold(vb_opt[indx], free = True)

    return sub_vb_params_dict, sub_vb_params_paragami, indx

def _get_kl_subparams(y, which_k, sub_vb_params_dict, vb_params_dict,
                            prior_params_dict, gh_loc, gh_weights):

    which_k_sorted = np.sort(which_k)

    vb_params_dict_copy = deepcopy(vb_params_dict)

    k_approx = vb_params_dict['cluster_params']['cluster_info'].shape[0]

    # centroids: need to be transposed, so we're indexing into the first dimension
    _centroids = np.transpose(vb_params_dict['cluster_params']['centroids'])
    _centroids_repl = np.transpose(sub_vb_params_dict['cluster_params']['centroids'])
    _centroids = replace(_centroids_repl, _centroids, which_k_sorted)
    vb_params_dict_copy['cluster_params']['centroids'] = np.transpose(_centroids)

    # cluster info
    vb_params_dict_copy['cluster_params']['cluster_info'] = \
        replace(sub_vb_params_dict['cluster_params']['cluster_info'],
                vb_params_dict['cluster_params']['cluster_info'],
                which_k_sorted)

    # get indices for sticks
    # last stick is deterministic; one less stick param than k_approx
    if(np.any(which_k == (k_approx - 1))):
        _which_k = which_k[which_k != (k_approx - 1)]
    else:
        _which_k = which_k[:-1]

    _which_k_sorted = np.sort(_which_k)

    # construct vb params for sticks
    vb_params_dict_copy['stick_params']['stick_propn_mean'] = \
        replace(sub_vb_params_dict['stick_params']['stick_propn_mean'],
               vb_params_dict['stick_params']['stick_propn_mean'],
               _which_k_sorted)

    vb_params_dict_copy['stick_params']['stick_propn_info'] = \
        replace(sub_vb_params_dict['stick_params']['stick_propn_info'],
               vb_params_dict['stick_params']['stick_propn_info'],
               _which_k_sorted)


    return gmm_lib.get_kl(y, vb_params_dict_copy, prior_params_dict, \
                            gh_loc, gh_weights), vb_params_dict_copy

# this is more straight-forward, but slower
def _get_kl_subparams2(y, indx, sub_vb_params,
                       vb_opt, vb_params_paragami,
                       prior_params_dict, gh_loc, gh_weights):

    vb_opt_repl = replace(sub_vb_params, vb_opt, indx)

    return gmm_lib.get_kl(y, vb_params_paragami.fold(vb_opt_repl, free = True), \
                       prior_params_dict, gh_loc, gh_weights)


def get_large_clusters_hessian(y, which_k, vb_opt, vb_params_paragami,
                                prior_params_dict,
                                gh_loc, gh_weights):

    sub_vb_params_dict, sub_vb_params_paragami, indx = \
        get_subvb_params(which_k, vb_opt, vb_params_paragami)

    kl_objective = paragami.FlattenFunctionInput(
                                original_fun= _get_kl_subparams,
                                patterns = sub_vb_params_paragami,
                                free = True,
                                argnums = 2)

    vb_params_dict = vb_params_paragami.fold(vb_opt, free = True)
    kl_objective_fun = lambda x : kl_objective(y, which_k, x,
                                        vb_params_dict, prior_params_dict,
                                        gh_loc, gh_weights)[0]

    get_sub_hess = autograd.hessian(kl_objective_fun)

    sub_hess = get_sub_hess(vb_opt[indx])

    est_hess = est_hess = np.zeros((len(vb_opt), len(vb_opt)))
    for i in range(len(indx)):
        for j in range(len(indx)):
            est_hess[indx[i], indx[j]] = sub_hess[i, j]

    return est_hess, indx


################
# Class that breaks down the Hessian of the KL into three components
# dkl/dtheta^2; dkl/(dtheta deta); deta / dtheta
################
def convert_nat_param_to_ez(z_nat_param):
    log_const = sp.special.logsumexp(z_nat_param, axis=1)
    return np.exp(z_nat_param - log_const[:, None])

class HessianComponents:
    def __init__(self, features, vb_params_paragami,
                    prior_params_dict, gh_loc, gh_weights,
                    use_bnp_prior = True):

        self.features = features
        self.vb_params_paragami = vb_params_paragami

        self.prior_params_dict = prior_params_dict
        self.gh_loc = gh_loc
        self.gh_weights = gh_weights
        self.use_bnp_prior = use_bnp_prior

        self.n_obs = features.shape[0]
        self.k_approx = len(vb_params_paragami.random()['stick_params']['stick_propn_mean']) + 1

    def get_kl_objective(self, vb_opt, e_z, ez_is_nat = False):
        vb_params_dict = self.vb_params_paragami.fold(vb_opt, free = True)

        if ez_is_nat:
            # if ez is in its natural paramterization
            _e_z = convert_nat_param_to_ez(e_z)
        else:
            _e_z = e_z

        return gmm_lib.get_kl(self.features, vb_params_dict, self.prior_params_dict,
                        self.gh_loc, self.gh_weights, _e_z,
                        use_bnp_prior=self.use_bnp_prior)

    def get_ez_nat(self, vb_opt):
        vb_params_dict = self.vb_params_paragami.fold(vb_opt, free = True)

        stick_propn_mean = vb_params_dict['stick_params']['stick_propn_mean']
        stick_propn_info = vb_params_dict['stick_params']['stick_propn_info']
        centroids = vb_params_dict['cluster_params']['centroids']
        cluster_info = vb_params_dict['cluster_params']['cluster_info']

        return gmm_lib.get_z_nat_params(self.features, stick_propn_mean,
                            stick_propn_info, centroids, cluster_info,
                            self.gh_loc, self.gh_weights,
                            use_bnp_prior = self.use_bnp_prior)[0]


    def get_dkl_dtheta2(self, vb_opt, e_z):
        get_kl_thetatheta = autograd.hessian(self.get_kl_objective, argnum = 0)
        return get_kl_thetatheta(vb_opt, e_z)


    def get_dznat_dtheta(self, vb_opt):
        foo = lambda x, y: (self.get_ez_nat(x) * y).sum()
        foo_grad = autograd.grad(foo, argnum = 0)
        get_ez_grad = autograd.jacobian(foo_grad, argnum = 1)
        return get_ez_grad(vb_opt, np.zeros((self.n_obs, self.k_approx))).transpose((1,2,0))

    def get_dkl_dtheta_dznat(self, vb_opt, e_z_nat):
        get_kl_theta = autograd.grad(self.get_kl_objective, argnum = 0)
        get_kl_thetaez = autograd.jacobian(get_kl_theta, argnum = 1)

        return get_kl_thetaez(vb_opt, e_z_nat, ez_is_nat = True)

    def get_cross_term(self, vb_opt):
        grad_kl_theta =  autograd.grad(self.get_kl_objective, argnum = 0)

        grad_cross_term = autograd.jacobian(lambda x : grad_kl_theta(vb_opt, self.get_ez_nat(xgit ), ez_is_nat = True))

        return grad_cross_term(vb_opt)
