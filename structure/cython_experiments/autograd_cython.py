#!/usr/bin/env python3

import autograd
import autograd.numpy as np
import autograd.scipy as sp

from autograd.core import primitive, defvjp, defjvp
from autograd.test_util import check_grads

from copy import deepcopy
import mixtures

np.random.seed(42)

n_num = 20
k_num = 3

x = np.random.random((n_num, k_num)) * 0.1
g = np.ones_like(x)
alpha = 0
beta = 1
p = 1

def get_e_z(x):
    log_norm = sp.misc.logsumexp(x, axis=1, keepdims=True)
    return np.exp(x - log_norm)

e_z = get_e_z(x)
assert np.max(np.abs(np.sum(e_z, axis=1) - 1)) < 1e-8
print('sum accuracy ', np.sum(e_z * x) -
      mixtures.get_mixture_sum(x, np.empty((0, 0)), 0, 1, 1, np.empty((0, 0))))


@primitive
def get_mixture_sum(x, g, alpha, beta, p):
    return mixtures.get_mixture_sum(
        x, g, alpha, beta, p, np.empty((0, 0)))

@primitive
def get_mixture_terms(x, g, alpha, beta, p):
    result = np.empty_like(x)
    mixtures.get_mixture_sum(
        x, g, alpha, beta, p, result)
    return result

print('terms accuracy',
    np.max(np.abs(e_z * x - get_mixture_terms(x, g, alpha, beta, p))))

def get_mixture_sum_vjpmaker_0(ans, x, g, alpha, beta, p):
    def vjp(h):
        terms1 = get_mixture_terms(
            x, g, p * alpha + beta, p * beta, p)
        terms2 = get_mixture_terms(
            x, g, p * alpha, p * beta, p + 1)
        terms2_sum = np.sum(terms2, axis=1, keepdims=True)
        return h * (terms1 - terms2_sum)
    return vjp

def get_mixture_sum_vjpmaker_1(ans, x, g, alpha, beta, p):
    def vjp(h):
        terms = get_mixture_terms(x, np.empty((0, 0)), alpha, beta, p)
        return h * terms
    return vjp

defvjp(get_mixture_sum,
       get_mixture_sum_vjpmaker_0,
       argnums=[0])

# defvjp(get_mixture_sum,
#        get_mixture_sum_vjpmaker_1,
#        argnums=[1])

# defvjp(get_mixture_sum,
#        get_mixture_sum_vjpmaker_0,
#        get_mixture_sum_vjpmaker_1,
#        argnums=[0, 1])

get_mixture_sum_grad = autograd.grad(get_mixture_sum, argnum=0)
x_grad = get_mixture_sum_grad(x, np.ones((n_num, k_num)), 0, 1, 1)

epsilon = 1e-5
mix_sum = get_mixture_sum(x, np.ones((n_num, k_num)), 0, 1, 1)
num_grad = np.zeros_like(x_grad)
for n in range(n_num):
    for k in range(k_num):
        x_new = deepcopy(x)
        x_new[n, k] += epsilon
        num_grad[n, k] = (get_mixture_sum(x_new, np.ones((n_num, k_num)), 0, 1, 1) - mix_sum) / epsilon

print('grad diff\n', num_grad - x_grad)
print('num grad\n', num_grad)
print('x grad\n', x_grad)
# print('x grad ', x_grad, x_grad.shape)
#
check_grads(get_mixture_sum, modes=['rev'], order=1)(x, np.ones((n_num, k_num)), 0, 1, 1)




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
