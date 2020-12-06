import jax

import jax.numpy as np
import jax.scipy as sp

import numpy as onp
from numpy.polynomial.hermite import hermgauss

import vb_lib.structure_model_lib as structure_model_lib
import vb_lib.cavi_lib as cavi_lib
import vb_lib.structure_optimization_lib as s_optim_lib
import vb_lib.functional_perturbation_lib as fpert_lib

import bnpmodeling_runjingdev.influence_lib as influence_lib

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

# which perturbation
parser.add_argument('--perturbation', type=str, default = 'worst-case')

# file where the influence file is stored
parser.add_argument('--influence_file', type=str)

args = parser.parse_args()

def validate_args():
    assert os.path.exists(args.out_folder), args.out_folder
    assert os.path.isfile(args.init_fit), args.init_fit
    assert os.path.isfile(args.data_file), args.data_file

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
epsilon_vec = np.linspace(0, 1, 12)[1:]**2
epsilon = epsilon_vec[args.epsilon_indx]
print('epsilon = ', epsilon)
print('epsilon_indx = ', args.epsilon_indx)

if args.perturbation = 'worst-case': 
    # worst case perturbation
    print('Refitting with worst-case perturbation')
    print('Loading influence function from ', args.influence_file)

    # load influence function
    lr_data = np.load(args.influence_file)

    logit_v_grid = np.array(lr_data['logit_v_grid'])
    influence_grid = np.array(lr_data['influence_grid'])

    # check model by comparing KLs
    kl = structure_model_lib.get_kl(g_obs, vb_params_dict,
                                    prior_params_dict,
                                    gh_loc = gh_loc,
                                    gh_weights = gh_weights)
    assert np.abs(kl - lr_data['kl']) < 1e-8

    # compute worst-case perturbation
    worst_case_pert = influence_lib.WorstCasePerturbation(influence_fun = None, 
                                                          logit_v_grid = logit_v_grid, 
                                                          cached_influence_grid = influence_grid)
    _e_log_phi = lambda means, infos : worst_case_pert.\
                        get_e_log_linf_perturbation(means.flatten(), 
                                                    infos.flatten())


# get perturbation
functional_pert = fpert_lib.FunctionalPerturbation(_e_log_phi, 
                                                   vb_params_paragami)

e_log_phi = lambda means, infos : functional_pert.e_log_phi_epsilon(means, infos, epsilon)

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
                                       epsilon=epsilon,
                                       data_file = args.data_file, 
                                       final_kl = final_kl, 
                                       optim_time = lbfgs_time)

print('Total optim time: {:.3f} secs'.format(time.time() - t0))


print('done. ')
