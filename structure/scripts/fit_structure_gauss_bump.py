import jax

import jax.numpy as np
import jax.scipy as sp

import numpy as onp
from numpy.polynomial.hermite import hermgauss

import structure_vb_lib.structure_model_lib as structure_model_lib
import structure_vb_lib.cavi_lib as cavi_lib
import structure_vb_lib.structure_optimization_lib as s_optim_lib

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

# which mu
parser.add_argument('--mu_indx', type=int, default = 0)

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
epsilon_vec = np.linspace(0, 1, 12)**2
epsilon = epsilon_vec[args.epsilon_indx]

mu_vec = np.linspace(1, 5, 6)
mu = mu_vec[args.mu_indx]

print('epsilon = ', epsilon)
print('mu = ', mu)

def gauss_bump(x, mu): 
    scale = 0.2
    pdf = sp.stats.norm.pdf(x, 
                            loc = mu, 
                            scale = scale)
    # normalize so max is 1
    pdf *= np.sqrt(2 * np.pi * scale ** 2)
    
    return pdf

f_obj = func_sens_lib.FunctionalPerturbationObjective(lambda x : gauss_bump(x, mu), 
                                                          vb_params_paragami, 
                                                          gh_loc = gh_loc, 
                                                          gh_weights = gh_weights, 
                                                          stick_key = 'ind_admix_params')

e_log_phi = lambda means, infos : f_obj.e_log_phi_epsilon(means, infos, epsilon)


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
final_kl = structure_model_lib.get_kl(g_obs, 
                                      vb_opt_dict,
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
                                       mu = mu,
                                       data_file = args.data_file, 
                                       final_kl = final_kl, 
                                       optim_time = lbfgs_time)

print('Total optim time: {:.3f} secs'.format(time.time() - t0))


print('done. ')
