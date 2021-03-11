import jax

import jax.numpy as np
import jax.scipy as sp

from structure_vb_lib import structure_model_lib
from structure_vb_lib import data_utils

from bnpmodeling_runjingdev import log_phi_lib

from bnpmodeling_runjingdev.sensitivity_lib import \
        HyperparameterSensitivityLinearApproximation

import paragami

from copy import deepcopy

import time

import re
import os
import argparse
parser = argparse.ArgumentParser()

# the genome dataset
parser.add_argument('--data_file', type=str)

# folder where the structure fit was saved
parser.add_argument('--out_folder', type=str)

# name of the structure fit 
parser.add_argument('--fit_file', type=str)

# tolerance of CG solver
parser.add_argument('--cg_tol', type=float, default=1e-2)

args = parser.parse_args()

fit_file = os.path.join(args.out_folder, args.fit_file)

def validate_args():
    assert os.path.exists(args.out_folder), args.out_folder
    assert os.path.isfile(args.data_file), args.data_file
    
    assert args.fit_file.endswith('.npz')
    assert os.path.isfile(fit_file), fit_file


validate_args()

outfile = re.sub('.npz', '_lrderivatives', fit_file)
print('derivative outfile: ', outfile)

##################
# Load data
##################
g_obs = data_utils.load_thrush_data(args.data_file)[0]

print(g_obs.shape)

n_obs = g_obs.shape[0]
n_loci = g_obs.shape[1]
n_allele = g_obs.shape[-1]


##################
# Load initial fit
##################
print('loading fit from ', fit_file)
vb_opt_dict, vb_params_paragami, \
    prior_params_dict, prior_params_paragami, \
        gh_loc, gh_weights, meta_data = \
            structure_model_lib.load_structure_fit(fit_file)

vb_opt = vb_params_paragami.flatten(vb_opt_dict, free = True)

###############
# Define objective and check KL
###############

def objective_fun(vb_params_free, epsilon): 
    
    # this actually does not depend on epsilon! 
    # we will set the perturbation later
    
    vb_params_dict = vb_params_paragami.fold(vb_params_free, free = True)
    
    return structure_model_lib.get_kl(g_obs, 
                                      vb_params_dict,
                                      prior_params_dict,
                                      gh_loc, 
                                      gh_weights).squeeze()

# check KL's match
kl = objective_fun(vb_opt, 0.)
diff = np.abs(kl - meta_data['final_kl'])
assert diff < 1e-8, diff


# Define the linear sensitivity class
vb_sens = HyperparameterSensitivityLinearApproximation(
                    objective_fun = objective_fun, 
                    opt_par_value = vb_opt, 
                    hyper_par_value0 = np.array([0.]),
                    # null for now. will set later
                    hyper_par_objective_fun = lambda x, y: 0., 
                    cg_tol = args.cg_tol,
                    cg_maxiter = None)

print('cg tol: ')
print(vb_sens.cg_tol)
print(vb_sens.cg_maxiter)

# save what we need
vars_to_save = dict()

def save_derivatives(vars_to_save): 
    print('saving into: ', outfile)
    np.savez(outfile,
             vb_opt = vb_opt,
             dp_prior_alpha = prior_params_dict['dp_prior_alpha'],
             kl= kl,
             cg_tol = vb_sens.cg_tol,
             cg_maxiter = vb_sens.cg_maxiter,
             **vars_to_save)

save_derivatives(vars_to_save)

    
###############
# Derivative wrt to functional perturbations
###############
f_obj_all = log_phi_lib.LogPhiPerturbations(vb_params_paragami, 
                                            prior_params_dict['dp_prior_alpha'],
                                            gh_loc, 
                                            gh_weights,
                                            stick_key = 'ind_admix_params')

def compute_derivatives_and_save(pert_name):
    
    print('###############')
    print('Computing derviative for ' + pert_name + ' functional perturbation ...')
    print('###############')

    
    # get hyper parameter objective function
    f_obj = getattr(f_obj_all, 'f_obj_' + pert_name)
    
    # compute derivative 
    print('computing derivative...')
    vb_sens._set_cross_hess_and_solve(f_obj.hyper_par_objective_fun)
    
    # save what we need
    vars_to_save['dinput_dfun_' + pert_name] = deepcopy(vb_sens.dinput_dhyper)
    vars_to_save['lr_time_' + pert_name] = deepcopy(vb_sens.lr_time)
    save_derivatives(vars_to_save)

compute_derivatives_and_save('sigmoidal')

compute_derivatives_and_save('alpha_pert_pos')
# compute_derivatives_and_save('alpha_pert_neg')

# compute_derivatives_and_save('alpha_pert_pos_xflip')
# compute_derivatives_and_save('alpha_pert_neg_xflip')

compute_derivatives_and_save('gauss_pert1')
compute_derivatives_and_save('gauss_pert2')


print('done. ')
