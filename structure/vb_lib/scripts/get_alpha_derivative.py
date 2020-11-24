import jax

import jax.numpy as np
import jax.scipy as sp
from jax.scipy.sparse.linalg import cg

from numpy.polynomial.hermite import hermgauss

from vb_lib import structure_model_lib
from vb_lib.preconditioner_lib import get_mfvb_cov_matmul
import vb_lib.structure_optimization_lib as s_optim_lib

import bnpmodeling_runjingdev.exponential_families as ef

from bnpmodeling_runjingdev.sensitivity_lib import \
        HyperparameterSensitivityLinearApproximation

import paragami

from copy import deepcopy

import time

import re
import os
import argparse
parser = argparse.ArgumentParser()

parser.add_argument('--data_file', type=str)
parser.add_argument('--out_folder', type=str)
parser.add_argument('--fit_file', type=str)

args = parser.parse_args()

fit_file = os.path.join(args.out_folder, args.fit_file)

def validate_args():
    assert os.path.exists(args.out_folder), args.out_folder
    assert os.path.isfile(args.data_file), args.data_file
    
    assert args.fit_file.endswith('.npz')
    assert os.path.isfile(fit_file), fit_file


validate_args()

##################
# Load data
##################
data = np.load(args.data_file)
g_obs = np.array(data['g_obs'])

##################
# Load initial fit
##################
print('loading fit from ', fit_file)
vb_opt_dict, vb_params_paragami, meta_data = \
    paragami.load_folded(fit_file)

vb_opt = vb_params_paragami.flatten(vb_opt_dict, free = True)

k_approx = vb_opt_dict['pop_freq_beta_params'].shape[1]

gh_deg = int(meta_data['gh_deg'])
gh_loc, gh_weights = hermgauss(gh_deg)

gh_loc = np.array(gh_loc)
gh_weights = np.array(gh_weights)

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
# Define objective and check KL
###############
# this also contains the hvp
stru_objective = s_optim_lib.StructureObjective(g_obs, 
                                                vb_params_paragami,
                                                prior_params_dict, 
                                                gh_loc, gh_weights, 
                                                jit_functions = False)

# check KL's match
kl = stru_objective.f(vb_opt)
diff = np.abs(kl - meta_data['final_kl'])
assert diff < 1e-8, diff

###############
# Define preconditioner
###############
cg_precond = lambda v : get_mfvb_cov_matmul(v, vb_opt_dict,
                                            vb_params_paragami,
                                            return_sqrt = False, 
                                            return_info = True)


###############
# Hyper-parameter objective
###############
def _hyper_par_objective_fun(vb_params_dict, alpha): 
    
    means = vb_params_dict['ind_admix_params']['stick_means']
    infos = vb_params_dict['ind_admix_params']['stick_infos']

    e_log_1m_sticks = \
        ef.get_e_log_logitnormal(
            lognorm_means = means,
            lognorm_infos = infos,
            gh_loc = gh_loc,
            gh_weights = gh_weights)[1]

    return - (alpha - 1) * np.sum(e_log_1m_sticks)

hyper_par_objective_fun = paragami.FlattenFunctionInput(
                                original_fun=_hyper_par_objective_fun, 
                                patterns = [vb_params_paragami, prior_params_paragami['dp_prior_alpha']],
                                free = [True, True],
                                argnums = [0, 1])


###############
# Sensitivity class
###############
alpha0 = prior_params_dict['dp_prior_alpha']
alpha_free = prior_params_paragami['dp_prior_alpha'].flatten(alpha0, 
                                                              free = True)

vb_sens = HyperparameterSensitivityLinearApproximation(
                    objective_fun = stru_objective.f, 
                    opt_par_value = vb_opt, 
                    hyper_par_value0 = alpha_free, 
                    obj_fun_hvp = stru_objective.hvp, 
                    hyper_par_objective_fun = hyper_par_objective_fun, 
                    cg_precond = cg_precond)

###############
# Save results 
###############
outfile = re.sub('.npz', '_lralpha', fit_file)
print('saving alpha derivative into: ', outfile)
np.savez(outfile, 
         dinput_dhyper = vb_sens.dinput_dhyper, 
         alpha_derivative_time = vb_sens.lr_time,
         vb_opt = vb_opt, 
         alpha0 = alpha0, 
         kl = kl) 
      
print('done. ')
