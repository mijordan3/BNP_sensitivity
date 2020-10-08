import jax

import jax.numpy as np
import jax.scipy as sp

import time

import numpy as onp

onp.random.seed(34535)

n_obs = 1000
dim = 500
x = onp.random.random((n_obs, dim))
y = onp.random.random((n_obs, dim))

def outer_fun(x_l, y_l):
    return ((np.outer(x_l, y_l) + 4.0)**2).mean()

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
