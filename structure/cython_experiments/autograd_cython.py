#!/usr/bin/env python3

import autograd
import autograd.numpy as np
import autograd.scipy as sp

from autograd.core import primitive, defvjp, defjvp
from autograd.test_util import check_grads

import mixtures

@primitive
def square(x):
    return mixtures.square(x)

defvjp(square,
       lambda ans, x: lambda g: 2.0 * x * g)

# defjvp(square,
#        lambda g, ans, x: 2.0 * x * g)

print('5 squared: ')
print(mixtures.square(5))
print(square(5))

square_grad = autograd.grad(square)
print('square grad:')
print(square_grad(5.0))

check_grads(square, modes=['rev', 'fwd'])

n_num = 20
k_num = 4

x = np.random.random((n_num, k_num)) * 0.1

def get_e_z(x):
    log_norm = sp.misc.logsumexp(x, axis=1, keepdims=True)
    return np.exp(x - log_norm)

e_z = get_e_z(x)
print(e_z)
print(np.sum(e_z, axis=1))
print(np.sum(e_z * x), mixtures.mixture_sum(x))
