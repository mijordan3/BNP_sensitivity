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
        self.get_log_qk = lambda logit_stick, vb_free_params : \
                        get_log_qk_from_free_params(logit_stick,
                                                    vb_free_params,
                                                    self.vb_params_paragami)

        # this returns the sum over sticks
        self.get_log_q = lambda logit_stick, vb_free_params : \
                            self.get_log_qk(logit_stick, vb_free_params).sum(0)

        self.grad_log_q = jax.jacobian(self.get_log_q, argnums = 1)

        # a len(vb_opt) x (k_approx - 1) binary matrix
        # with 1 if the ith vb free parameter affects the jth stick distribution.
        self.stick_params_mapping = \
                get_stick_params_mapping(self.vb_opt,
                                         self.vb_params_paragami)

    def get_influence(self, logit_stick):

        # this is len(vb_opt) x len(logit_stick)
        grad_log_q = self.grad_log_q(logit_stick, self.vb_opt).transpose()

        # this is (k_approx - 1) x len(logit_stick)
        prior_ratio = np.exp(self.get_q_prior_log_ratio(logit_stick))

        # map each stick to appropriate vb free param
        # this is len(vb_opt) x len(logit_stick)
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

    def get_q_prior_log_ratio(self, logit_stick):
        # this is log q(logit_stick)  - log p_0(logit_stick)
        # returns a matrix of (k_approx - 1) x length(logit_stick)


        log_beta_prior = get_log_logitstick_prior(logit_stick, self.alpha0)
        log_ratio = self.get_log_qk(logit_stick, self.vb_opt) \
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

# get explicit density (not expectations) for logit-sticks
# log p_0
def get_log_logitstick_prior(logit_stick, alpha):
    # pi are the stick lengths
    # alpha is the DP parameter

    stick = sp.special.expit(logit_stick)
    return sp.stats.beta.logpdf(stick, a = 1., b = alpha) + \
                np.log(stick) + np.log(1 - stick)


def get_log_qk_from_free_params(logit_stick, vb_free_params, vb_params_paragami):
    vb_params_dict = vb_params_paragami.fold(vb_free_params, free = True)

    mean = vb_params_dict['stick_params']['stick_propn_mean']
    info = vb_params_dict['stick_params']['stick_propn_info']

    logit_stick = np.expand_dims(logit_stick, 0)
    mean = np.expand_dims(mean, 1)
    info = np.expand_dims(info, 1)

    return sp.stats.norm.logpdf(logit_stick, mean, scale = 1 / np.sqrt(info))
