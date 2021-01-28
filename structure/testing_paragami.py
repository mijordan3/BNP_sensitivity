import jax
import jax.numpy as np

from vb_lib import structure_model_lib

import paragami

print(jax.__version__)

import jaxlib
print(jaxlib.__version__)


n_obs = 25
n_loci = 100
k_approx = 7

vb_params_dict, vb_params_paragami = \
    structure_model_lib.get_vb_params_paragami_object(n_obs, n_loci, k_approx,
                                                      use_logitnormal_sticks = True)

print(vb_params_paragami)


def flatten(vb_params_dict):
    return vb_params_paragami.flatten(vb_params_dict, free = True)


jflatten = jax.jit(flatten)

vb_params_free = flatten(vb_params_dict)

print(flatten(vb_params_dict))
print(jflatten(vb_params_dict))


def fold(vb_params_free):
    return vb_params_paragami.flatten(vb_params_free, free = True)

jfold = jax.jit(fold)

print(fold(vb_params_free))
print(jfold(vb_params_free))
