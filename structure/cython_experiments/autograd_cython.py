#!/usr/bin/env python3

import autograd
from autograd.core import primitive, defvjp, defjvp
from autograd.test_util import check_grads

import helloworld

@primitive
def square(x):
    return helloworld.square(x)

defvjp(square,
       lambda ans, x: lambda g: 2.0 * x * g)

# defjvp(square,
#        lambda g, ans, x: 2.0 * x * g)

print('5 squared: ')
print(helloworld.square(5))
print(square(5))

square_grad = autograd.grad(square)
print('square grad:')
print(square_grad(5.0))

check_grads(square, modes=['rev', 'fwd'])
