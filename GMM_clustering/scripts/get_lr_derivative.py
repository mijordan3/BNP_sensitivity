import jax

import jax.numpy as np
import jax.scipy as sp

from numpy.polynomial.hermite import hermgauss

# BNP gmm libraries
import bnpgmm_runjingdev.gmm_clustering_lib as gmm_lib
import bnpgmm_runjingdev.gmm_cavi_lib as cavi_lib
import bnpgmm_runjingdev.utils_lib as utils_lib

import bnpmodeling_runjingdev.functional_sensitivity_lib as func_sens_lib
import bnpmodeling_runjingdev.exponential_families as ef
from bnpmodeling_runjingdev import influence_lib, log_phi_lib

from bnpmodeling_runjingdev.sensitivity_lib import \
        HyperparameterSensitivityLinearApproximation

import paragami

from copy import deepcopy

import time

import re
import os
import argparse
parser = argparse.ArgumentParser()

# folder where the structure fit was saved
parser.add_argument('--out_folder', type=str)

# name of the iris fit 
parser.add_argument('--fit_file', type=str)

# tolerance of CG solver
parser.add_argument('--cg_tol', type=float, default=1e-3)

args = parser.parse_args()

fit_file = os.path.join(args.out_folder, args.fit_file)

def validate_args():
    assert os.path.exists(args.out_folder), args.out_folder
    
    assert args.fit_file.endswith('.npz')
    assert os.path.isfile(fit_file), fit_file


validate_args()

outfile = re.sub('.npz', '_lrderivatives', fit_file)
print('derivative outfile: ', outfile)

##################
# Load data
##################
dataset_name = 'iris'
iris_obs, iris_species = utils_lib.load_iris_data()
dim = iris_obs.shape[1]
n_obs = len(iris_species)

iris_obs = np.array(iris_obs)


##################
# Load initial fit
##################
print('loading fit from ', fit_file)
vb_opt_dict, vb_params_paragami, meta_data = \
        paragami.load_folded(fit_file)
vb_opt = vb_params_paragami.flatten(vb_opt_dict, free = True)    

# gauss-hermite parameters
gh_deg = int(meta_data['gh_deg'])
gh_loc, gh_weights = hermgauss(gh_deg)

gh_loc = np.array(gh_loc)
gh_weights = np.array(gh_weights)
    
# load prior parameters
prior_params_dict, prior_params_paragami = gmm_lib.get_default_prior_params(dim)

# set initial alpha
alpha0 = meta_data['alpha']
prior_params_dict['alpha'] = alpha0
print('alpha: ', prior_params_dict['alpha'])

###############
# Define objective and check KL
###############
# this also contains the hvp
def objective_fun(vb_free, epsilon): 
    # NOTE! epsilon doesn't actual enter 
    # into this function. 
    
    # since the initial fit is at epsilon = 0, 
    # we just return the actual KL
    
    # we will set the hyper-param objective function 
    # appropriately, later. 
    
    vb_params_dict = vb_params_paragami.fold(vb_free, free = True)
    
    return gmm_lib.get_kl(iris_obs,
                          vb_params_dict,
                          prior_params_dict,
                          gh_loc,
                          gh_weights).squeeze()

# check KL's match
kl = objective_fun(vb_opt, None)
diff = np.abs(kl - meta_data['final_kl'])
assert diff < 1e-8, diff

###############
# Define the linear sensitivity class
###############
vb_sens = HyperparameterSensitivityLinearApproximation(
                    objective_fun = objective_fun, 
                    opt_par_value = vb_opt, 
                    hyper_par_value0 = np.array([0.]), 
                    # will set appropriately later
                    hyper_par_objective_fun = lambda x, y : 0., 
                    cg_tol = args.cg_tol)
    
###############
# Derivative wrt to functional perturbations
###############
f_obj_all = log_phi_lib.LogPhiPerturbations(vb_params_paragami, 
                                            prior_params_dict['alpha'],
                                            gh_loc, 
                                            gh_weights,
                                            stick_key = 'stick_params')

vars_to_save = dict()

def save_derivatives(vars_to_save): 
    print('saving into: ', outfile)
    np.savez(outfile,
             vb_opt = vb_opt,
             alpha0 = alpha0,
             kl= kl,
             **vars_to_save)


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
compute_derivatives_and_save('alpha_pert_neg')

compute_derivatives_and_save('alpha_pert_pos_xflip')
compute_derivatives_and_save('alpha_pert_neg_xflip')

compute_derivatives_and_save('gauss_pert1')
compute_derivatives_and_save('gauss_pert2')


print('done. ')
