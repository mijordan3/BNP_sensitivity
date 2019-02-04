#!/usr/bin/env python3

import autograd
import autograd.numpy as np
import autograd.scipy as sp

from autograd.core import primitive, defvjp, defjvp
from autograd.test_util import check_grads

import mixtures

n_num = 200
k_num = 10

x = np.random.random((n_num, k_num)) * 0.1

def get_e_z(x):
    log_norm = sp.misc.logsumexp(x, axis=1, keepdims=True)
    return np.exp(x - log_norm)

e_z = get_e_z(x)
assert np.max(np.abs(np.sum(e_z, axis=1) - 1)) < 1e-8
print(np.sum(e_z * x) - mixtures.get_mixture_sum(x, np.empty((0, 0)), 0, 1, 1))


@primitive
def get_mixture_sum(x, g, alpha, beta, p):
    return mixtures.get_mixture_sum(
        x, g, alpha, beta, p, np.empty((0, 0)))

@primitive
def get_mixture_terms(x, g, alpha, beta, p):
    result = np.empty_like(x)
    return mixtures.get_mixture_sum(
        x, g, alpha, beta, p, result)


def get_mixture_sum_vjp(ans, x, g, alpha, beta, p):
    def vjp(h):
        print('vjp h shape: ', h.shape)
        k_num = x.shape[1]
        term1 = get_mixture_sum(x, g, p * alpha + beta, p * beta, p)
        term2 = get_mixture_sum(x, g, k_num * p * alpha, k_num * p * beta, p + 1)
        return h * (term1 - term2)
    return vjp

defvjp(get_mixture_sum, get_mixture_sum_vjp)

get_mixture_sum_grad = autograd.grad(get_mixture_sum, argnum=0)
x_grad = get_mixture_sum_grad(x, np.ones((n_num, k_num)), 0, 1, 1)
print('x grad ', x_grad, x_grad.shape)

check_grads(get_mixture_sum, modes=['rev'])(x, np.ones((n_num, k_num)), 0, 1, 1)




# def get_mixture_sum_jvp(h, ans, x, g, alpha, beta, p):
#     # Forward mode
#     print('jvp h shape: ', h.shape)
#     empty_arr = np.empty((0, 0))
#     term1 = get_mixture_sum(x, g * h, p * alpha + beta, p * beta, p)
#     h_n = np.sum(h, axis=1, keepdims=True)
#     term2 = get_mixture_sum(x, g * h_n, p * alpha, p * beta, p + 1)
#     return term1 - term2
#
# defjvp(get_mixture_sum, get_mixture_sum_jvp)
