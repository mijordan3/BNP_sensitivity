import jax

import jax.numpy as np
import jax.scipy as sp

import numpy as onp
from numpy.polynomial.hermite import hermgauss

import vb_lib.structure_model_lib as structure_model_lib
import vb_lib.cavi_lib as cavi_lib
import vb_lib.structure_optimization_lib as s_optim_lib

import bnpmodeling_runjingdev.functional_sensitivity_lib as func_sens_lib
from bnpmodeling_runjingdev import influence_lib, log_phi_lib

import paragami

import argparse
import distutils.util

import os

import time

parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, default=42)

# data file
parser.add_argument('--data_file', type=str)

# where to save the structure fit
parser.add_argument('--out_folder', default='../fits/')
parser.add_argument('--out_filename', default='structure_fit', type=str)

# the initial fit
parser.add_argument('--init_fit', type=str)

# which epsilon 
parser.add_argument('--epsilon_indx', type=int, default = 0)

# delta 
parser.add_argument('--delta', type=float, default = 1.0)

# which perturbation
parser.add_argument('--perturbation', type=str, default = 'worst_case')

# file where the influence file is stored
parser.add_argument('--influence_file', type=str)

args = parser.parse_args()

def validate_args():
    assert os.path.exists(args.out_folder), args.out_folder
    assert os.path.isfile(args.init_fit), args.init_fit
    assert os.path.isfile(args.data_file), args.data_file
    
    if args.perturbation == 'worst-case': 
        # check influence file exists
        os.path.exists(args.influence_file), args.out_folder

validate_args()

onp.random.seed(args.seed)

##################
# Load data
##################
print('loading data from ', args.data_file)
data = np.load(args.data_file)
g_obs = np.array(data['g_obs'])

##################
# Load initial fit
##################
print('loading fit from ', args.init_fit)
vb_params_dict, vb_params_paragami, \
    prior_params_dict, prior_params_paragami, \
        gh_loc, gh_weights, fit_meta_data = \
            structure_model_lib.load_structure_fit(args.init_fit)

##################
# Define perturbation
##################
# set epsilon 
epsilon_vec = np.linspace(0, 1, 20)[1:]**2 
epsilon = epsilon_vec[args.epsilon_indx]
print('epsilon = ', epsilon)
print('epsilon_indx = ', args.epsilon_indx)

print('refitting with perturbation = ', args.perturbation)
if args.perturbation == 'worst_case': 
    # worst case perturbation
    print('Loading influence function from ', args.influence_file)

    # load influence function
    lr_data = np.load(args.influence_file)
    
    # check KL's match
    assert np.abs(fit_meta_data['final_kl'] - lr_data['kl']) < 1e-8
    
    logit_v_grid = np.array(lr_data['logit_v_grid'])
    influence_grid = np.array(lr_data['influence_grid'])
else: 
    logit_v_grid = None
    influence_grid = None

f_obj_all = log_phi_lib.LogPhiPerturbations(vb_params_paragami, 
                                                 prior_params_dict['dp_prior_alpha'],
                                                 gh_loc, 
                                                 gh_weights,
                                                 logit_v_grid = logit_v_grid, 
                                                 influence_grid = influence_grid,
                                                 delta = args.delta, 
                                                 stick_key = 'ind_admix_params')

f_obj = getattr(f_obj_all, 'f_obj_' + args.perturbation)
e_log_phi = lambda means, infos : f_obj.e_log_phi_epsilon(means, infos, epsilon)



# # warm start w linear response 
# vb_opt = vb_params_paragami.flatten(vb_params_dict, 
#                                     free = True)
# vb_opt_pert = vb_opt + lr_data['dinput_dfun_' + args.perturbation] * epsilon
# vb_params_dict = vb_params_paragami.fold(vb_opt_pert, 
#                                          free = True)

######################
# OPTIMIZE
######################
t0 = time.time() 
# optimize with preconditioner 
vb_opt_dict, vb_opt, out, precond_objective, lbfgs_time = \
    s_optim_lib.run_preconditioned_lbfgs(g_obs, 
                        vb_params_dict, 
                        vb_params_paragami,
                        prior_params_dict,
                        gh_loc, gh_weights, 
                        e_log_phi = e_log_phi)

######################
# save optimization results
######################
outfile = os.path.join(args.out_folder, args.out_filename)

print('saving structure model to ', outfile)

print('Optim time (ignoring compilation time) {:.3f}secs'.format(lbfgs_time))

# save final KL
final_kl = structure_model_lib.get_kl(g_obs, vb_opt_dict,
                            prior_params_dict,
                            gh_loc = gh_loc,
                            gh_weights = gh_weights, 
                            e_log_phi = e_log_phi)

# save paragami object
structure_model_lib.save_structure_fit(outfile, 
                                       vb_opt_dict,
                                       vb_params_paragami, 
                                       prior_params_dict,
                                       fit_meta_data['gh_deg'], 
                                       epsilon = epsilon,
                                       delta = args.delta,
                                       data_file = args.data_file, 
                                       final_kl = final_kl, 
                                       optim_time = lbfgs_time)

print('Total optim time: {:.3f} secs'.format(time.time() - t0))


print('done. ')
