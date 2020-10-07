import jax

import jax.numpy as np
import jax.scipy as sp

from numpy.polynomial.hermite import hermgauss
import scipy as osp

from vb_lib import structure_model_lib, data_utils, cavi_lib
from vb_lib.preconditioner_lib import get_mfvb_cov_matmul
from bnpmodeling_runjingdev.sensitivity_lib import HyperparameterSensitivityLinearApproximation, get_jac_hvp_fun

import paragami
from vittles import solver_lib

from copy import deepcopy

import time

from bnpmodeling_runjingdev import cluster_quantities_lib, modeling_lib

data_dir = '/scratch/users/genomic_times_series_bnp/structure/data/'
fit_dir = '/scratch/users/genomic_times_series_bnp/structure/fits/fits_20201007/'

##################
# Load data
##################
n_obs = 100
n_loci = 2000
n_pop = 4

data_file = data_dir + 'simulated_structure_data_nobs{}_nloci{}_npop{}.npz'.format(n_obs, n_loci, n_pop)
print('loading data from ', data_file)
data = np.load(data_file)

g_obs = np.array(data['g_obs'])

##################
# Load initial fit
##################
alpha0 = 3.5

fit_file = fit_dir + 'structure_fit_nobs{}_nloci{}_npop{}_alpha{}.npz'.format(n_obs,
                                                                              n_loci,
                                                                              n_pop,
                                                                              alpha0)
print('loading fit from ', fit_file)
vb_opt_dict, vb_params_paragami, meta_data = \
    paragami.load_folded(fit_file)

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

dinput_dyper_file = fit_dir + 'lr_nobs{}_nloci{}_npop{}_alpha{}'.format(n_obs, n_loci, n_pop, prior_params_dict['dp_prior_alpha'][0])
print('saving derivative into ', dinput_dyper_file)
np.save(dinput_dyper_file, vb_sens.dinput_dhyper)

print('done. ')
