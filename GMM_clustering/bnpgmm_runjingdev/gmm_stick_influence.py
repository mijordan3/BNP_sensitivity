import jax
import jax.numpy as np
import jax.scipy as sp


########################
# Functions to evaluate
# the influence function
########################

class InfluenceFunction(object):
    def __init__(self, vb_opt, vb_params_paragami, hessian_solver, alpha0):

        self.vb_opt = vb_opt
        self.vb_params_paragami = vb_params_paragami
        self.hessian_solver = hessian_solver
        self.alpha0 = alpha0

        # stick densities
        # this returns the per stick density
        self.get_log_qk = lambda theta, vb_free_params : \
                        get_log_qk_from_free_params(theta,
                                                    vb_free_params,
                                                    self.vb_params_paragami)

        # this returns the sum over sticks
        self.get_log_q = lambda theta, vb_free_params : \
                            self.get_log_qk(theta, vb_free_params).sum(0)

        self.grad_log_q = jax.jacobian(self.get_log_q, argnums = 1)

        # a len(vb_opt) x (k_approx - 1) binary matrix
        # with 1 if the ith vb free parameter affects the jth stick distribution.
        self.stick_params_mapping = \
                get_stick_params_mapping(self.vb_opt,
                                         self.vb_params_paragami)

    def get_influence(self, theta):

        # this is len(vb_opt) x len(theta)
        grad_log_q = self.grad_log_q(theta, self.vb_opt).transpose()

        # this is (k_approx - 1) x len(theta)
        prior_ratio = np.exp(self.get_q_prior_log_ratio(theta))

        # map each stick to appropriate vb free param
        # this is len(vb_opt) x len(theta)
        prior_ratio_expanded = np.dot(self.stick_params_mapping, prior_ratio)

        # combine prior ratio and grad log q
        grad_log_q_prior_rat = grad_log_q * prior_ratio_expanded

        # solve
        # somehow jax.lax.map is super slow?
        # jut using python for loop here ...
        influence_operator = \
            - np.stack([self.hessian_solver(x) \
                        for x in grad_log_q_prior_rat.transpose()])

        return influence_operator.transpose()

    def get_q_prior_log_ratio(self, theta):
        # this is log q(theta)  - log p_0(theta)
        # returns a matrix of (k_approx - 1) x length(theta)


        log_beta_prior = get_log_beta_prior(theta, self.alpha0)
        log_ratio = self.get_log_qk(theta, self.vb_opt) \
                        - np.expand_dims(log_beta_prior, 0)

        return log_ratio

def stick_params_mapping_obj(vb_free_params, vb_params_paragami):
    vb_params_dict = vb_params_paragami.fold(vb_free_params, free = True)
    return vb_params_dict['stick_params']['stick_propn_mean'] + \
                vb_params_dict['stick_params']['stick_propn_info']

stick_params_mapping_jac = jax.jacobian(stick_params_mapping_obj, argnums = 0)

def get_stick_params_mapping(vb_free_params, vb_params_paragami):
    return stick_params_mapping_jac(vb_free_params,
                                    vb_params_paragami).transpose() != 0

# get explicit density (not expectations) for sticks
# log q
def get_log_logitnormal_density(theta, mean, info):

    # returns a len(mean) x len(theta) matrix of densities
    # row i , column j contains the log-pdf of a logitnormal
    # with mean[i] and info[i] evaluated at theta[j]

    assert len(mean) == len(info)

    theta = np.expand_dims(theta, 0)
    mean = np.expand_dims(mean, 1)
    info = np.expand_dims(info, 1)

    return - 0.5 * (np.log(2 * np.pi) - np.log(info)) + \
            -0.5 * info * (sp.special.logit(theta) - mean) ** 2 + \
            -np.log(theta) - np.log(1 - theta)

# log p_0
def get_log_beta_prior(pi, alpha):
    # pi are the stick lengths
    # alpha is the DP parameter

    return sp.stats.beta.logpdf(pi, a = 1., b = alpha)


def get_log_qk_from_free_params(theta, vb_free_params, vb_params_paragami):
    vb_params_dict = vb_params_paragami.fold(vb_free_params, free = True)

    mean = vb_params_dict['stick_params']['stick_propn_mean']
    info = vb_params_dict['stick_params']['stick_propn_info']

    return get_log_logitnormal_density(theta, mean, info)
