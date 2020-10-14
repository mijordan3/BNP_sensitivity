import jax

import jax.numpy as np
import jax.scipy as sp

import time

import numpy as onp

onp.random.seed(34535)

n_loci = 500
n_obs = 200
k_approx = 12

x = onp.random.random((n_loci, n_obs))
y = onp.random.random((n_loci, k_approx))

def outer_fun(x_l, y_l):
    outer = np.outer(x_l, y_l)
    return ((outer + 4.0)**2).mean()

def fun(x, y):

    init_val = 0.0

    body_fun = lambda val, l : outer_fun(x[l], y[l]) + val
    scan_fun = lambda val, l : (body_fun(val, l), None)
    out = jax.lax.scan(scan_fun, init_val, xs = np.arange(x.shape[0]))[0]

    return out

x = np.array(x)
y = np.array(y)

fun_jitted = jax.jit(fun)

print('Compiling function ...')
t0 = time.time()
_ = fun_jitted(x, y).block_until_ready()
print('Compile time: {}sec'.format(time.time() - t0))

t0 = time.time()
for i in range(100):
    _ = fun_jitted(x, y).block_until_ready()
elapsed = time.time() - t0
print('function time: {}sec'.format(elapsed / 100))
