import autograd
import paragami
import time

import autograd.scipy as sp
import autograd.numpy as np

from bnpgmm_runjingdev.gmm_clustering_lib import \
        get_optimal_z_from_vb_params_dict, get_z_nat_params, get_kl

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

        return get_kl(self.features, vb_params_dict, self.prior_params_dict,
                        self.gh_loc, self.gh_weights, _e_z,
                        use_bnp_prior=self.use_bnp_prior)

    def get_ez_nat(self, vb_opt):
        vb_params_dict = self.vb_params_paragami.fold(vb_opt, free = True)

        stick_propn_mean = vb_params_dict['stick_params']['stick_propn_mean']
        stick_propn_info = vb_params_dict['stick_params']['stick_propn_info']
        centroids = vb_params_dict['cluster_params']['centroids']
        cluster_info = vb_params_dict['cluster_params']['cluster_info']

        return get_z_nat_params(self.features, stick_propn_mean,
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
