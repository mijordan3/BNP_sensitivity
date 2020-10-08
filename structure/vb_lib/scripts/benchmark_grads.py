import jax

import jax.numpy as np
import jax.scipy as sp

from numpy.polynomial.hermite import hermgauss

from vb_lib import structure_model_lib, data_utils, cavi_lib

from bnpmodeling_runjingdev.optimization_lib import construct_and_compile_derivatives

import paragami

import time

import numpy as onp
onp.random.seed(53453)

################
# load data
################
n_obs = 200
n_loci = 500
n_pop = 4

data_file = '../simulated_data/simulated_structure_data_nobs{}_nloci{}_npop{}.npz'.format(n_obs, n_loci, n_pop)
data = np.load(data_file)

g_obs = np.array(data['g_obs'])

assert g_obs.shape[0] == n_obs
assert g_obs.shape[1] == n_loci

################
# get prior
################
prior_params_dict, prior_params_paragami = \
    structure_model_lib.get_default_prior_params()

dp_prior_alpha = prior_params_dict['dp_prior_alpha']
allele_prior_alpha = prior_params_dict['allele_prior_alpha']
allele_prior_beta = prior_params_dict['allele_prior_beta']

################
# get vb params
################
k_approx = 8

gh_deg = 8
gh_loc, gh_weights = hermgauss(gh_deg)

use_logitnormal_sticks = False

vb_params_dict, vb_params_paragami = \
    structure_model_lib.get_vb_params_paragami_object(n_obs, n_loci, k_approx,
                                    use_logitnormal_sticks = use_logitnormal_sticks)

################
# initialize
################
vb_params_dict = \
        structure_model_lib.set_init_vb_params(g_obs, k_approx, vb_params_dict,
                                                seed = 34221)

################
# set up loss
################
_kl_fun_free = paragami.FlattenFunctionInput(
                            original_fun=structure_model_lib.get_kl,
                            patterns = vb_params_paragami,
                            free = True,
                            argnums = 1)

kl_fun_free = lambda x : _kl_fun_free(g_obs, x, prior_params_dict,
                                                 gh_loc, gh_weights,
                                                 log_phi = None,
                                                 epsilon = 0.)

# initial free parameters
init_vb_free = vb_params_paragami.flatten(vb_params_dict, free = True)

################
# set up gradients
################
optim_objective, optim_objective_np, optim_grad_np, optim_hvp_np = \
    construct_and_compile_derivatives(kl_fun_free, init_vb_free, compile_hvp = False)

optim_objective.set_print_every(1000)

################
# time
################
t0 = time.time()
for i in range(100):
    _ = optim_objective_np(init_vb_free)

elapsed = (time.time() - t0) / 100
print('\nobjective time: {}sec'.format(elapsed))


t0 = time.time()
for i in range(100):
    _ = optim_grad_np(init_vb_free)

elapsed = (time.time() - t0) / 100
print('gradient time: {}sec'.format(elapsed))
