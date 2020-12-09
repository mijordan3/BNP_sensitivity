#!/usr/bin/env python3

import autograd
import autograd.numpy as np
import autograd.scipy as sp

from autograd.core import primitive, defvjp, defjvp
from autograd.test_util import check_grads

import mixtures

@primitive
def myprod(x, y, z):
    return mixtures.myprod(x, y) + z

myprod(3., 2., 10.)

# Reverse mode
def myprod_vjpmaker_0(ans, x, y, z):
    def myprod_vjp_0(g):
        return g * mixtures.myprod(1, y)
    return myprod_vjp_0

def myprod_vjpmaker_1(ans, x, y, z):
    def myprod_vjp_1(g):
        return g * 2 * x * y
    return myprod_vjp_1

defvjp(myprod, myprod_vjpmaker_0, myprod_vjpmaker_1, argnums=[0, 1])

# Forward mode
def myprod_jvp0(g, ans, x, y, z):
    return g * myprod(1, y, 0)

def myprod_jvp1(g, ans, x, y, z):
    return g * 2 * x * y

defjvp(myprod, myprod_jvp0, myprod_jvp1, argnums=[0, 1])

check_grads(myprod, modes=['rev', 'fwd'])(3., 2., 10.)

print('ok')

#
#
#
# @primitive
# def square(x):
#     return mixtures.square(x)
#
# defvjp(square,
#        lambda ans, x: lambda g: 2.0 * x * g)
#
# @primitive
# def mypow(x, p):
#     return mixtures.mypow(x, p)
#
# def mypow_jvp(ans, x, p):
#     def jvp(g):
#         return p * x * g
#     return jvp
#
# defvjp(mypow, mypow_jvp)
# mypow_grad = autograd.grad(mypow)
# print('mypow_grad', mypow_grad(5.0, 2))
#
# def mysquare(x, p=2):
#     return mypow(x, p)
#
# mysquare_grad = autograd.grad(mysquare)
# print('mysquare_grad', mysquare_grad(5.0))
#
# print('5 squared: ')
# print(mixtures.square(5))
# print(square(5))
#
# square_grad = autograd.grad(square)
# print('square grad:')
# print(square_grad(5.0))
#
# check_grads(square, modes=['rev', 'fwd'])
