import jax

import jax.numpy as np
import jax.scipy as sp

from numpy.polynomial.hermite import hermgauss

from vb_lib import structure_model_lib
from vb_lib.preconditioner_lib import get_mfvb_cov_matmul
from bnpmodeling_runjingdev.sensitivity_lib import HyperparameterSensitivityLinearApproximation

import paragami

from copy import deepcopy

import os
import argparse
parser = argparse.ArgumentParser()

parser.add_argument('--data_file', type=str)
parser.add_argument('--fit_file', type=str)

parser.add_argument('--out_folder', type=str)
parser.add_argument('--out_file', type=str)

args = parser.parse_args()

def validate_args():
    assert os.path.exists(args.out_folder), args.out_folder
    assert os.path.isfile(args.data_file), args.data_file
    assert os.path.isfile(args.fit_file), args.fit_file


validate_args()

##################
# Load data
##################
data = np.load(args.data_file)
g_obs = np.array(data['g_obs'])

##################
# Load initial fit
##################
print('loading fit from ', args.fit_file)
vb_opt_dict, vb_params_paragami, meta_data = \
    paragami.load_folded(args.fit_file)

vb_opt = vb_params_paragami.flatten(vb_opt_dict, free = True)

k_approx = vb_opt_dict['pop_freq_beta_params'].shape[1]

gh_deg = int(meta_data['gh_deg'])
gh_loc, gh_weights = hermgauss(gh_deg)

gh_loc = np.array(gh_loc)
gh_weights = np.array(gh_weights)

use_logitnormal_sticks = meta_data['use_logitnormal_sticks'] == 1

###############
# Get prior
###############
prior_params_dict, prior_params_paragami = \
    structure_model_lib.get_default_prior_params()

prior_params_dict['dp_prior_alpha'] = np.array(meta_data['dp_prior_alpha'])
prior_params_dict['allele_prior_alpha'] = np.array(meta_data['allele_prior_alpha'])
prior_params_dict['allele_prior_beta'] = np.array(meta_data['allele_prior_beta'])

print(prior_params_dict)

prior_params_free = prior_params_paragami.flatten(prior_params_dict, free = True)

###############
# check KL
###############
kl = structure_model_lib.get_kl(g_obs, vb_opt_dict, prior_params_dict,
                                gh_loc = gh_loc, gh_weights = gh_weights)
# check KL's match
print(np.abs(kl - meta_data['final_kl']))
assert np.abs(kl - meta_data['final_kl']) < 1e-8

###############
# get sensitivity object
###############
# initial prior alpha
use_free_alpha = True
prior_alpha0 = prior_params_paragami['dp_prior_alpha'].flatten(prior_params_dict['dp_prior_alpha'],
                                                              free = use_free_alpha)

# set up objective as function of vb params and prior param
def objective_fun(vb_params_dict, alpha):

    _prior_params_dict = deepcopy(prior_params_dict)
    _prior_params_dict['dp_prior_alpha'] = alpha

    return structure_model_lib.get_kl(g_obs, vb_params_dict, _prior_params_dict,
                    gh_loc = gh_loc, gh_weights = gh_weights)


objective_fun_free = paragami.FlattenFunctionInput(
                                original_fun=objective_fun,
                                patterns = [vb_params_paragami, prior_params_paragami['dp_prior_alpha']],
                                free = [True, use_free_alpha],
                                argnums = [0, 1])

# define preconditioner
precon_fun = lambda v : get_mfvb_cov_matmul(v, vb_opt_dict,
                                            vb_params_paragami,
                                            return_info = True)

print('Computing sensitivity ...')
vb_sens = HyperparameterSensitivityLinearApproximation(objective_fun_free,
                                                        vb_opt,
                                                        prior_alpha0,
                                                        cg_precond=precon_fun)

dinput_dyper_file = os.path.join(args.out_folder, args.out_file)
print('saving derivative into ', dinput_dyper_file)
np.save(dinput_dyper_file, vb_sens.dinput_dhyper)

print('done. ')
