import jax

import jax.numpy as np
import jax.scipy as sp

import numpy as onp
from numpy.polynomial.hermite import hermgauss

import vb_lib.structure_model_lib as structure_model_lib
import vb_lib.cavi_lib as cavi_lib
import vb_lib.structure_optimization_lib as s_optim_lib

from bnpmodeling_runjingdev import influence_lib
import bnpmodeling_runjingdev.functional_sensitivity_lib as func_sens_lib

import paragami

from copy import deepcopy

import argparse
import distutils.util

import os

import time

parser = argparse.ArgumentParser()

# epsilon index
parser.add_argument('--epsilon_indx', type = int)

# data file
parser.add_argument('--data_file', type=str)

# where to save the structure fit
parser.add_argument('--out_folder', default='../fits/')
parser.add_argument('--out_filename', default='structure_fit', type=str)

# initial fit
parser.add_argument('--init_fit', type=str)

# whether to use worst-case perturbation
parser.add_argument('--use_worst_case', type=distutils.util.strtobool, default='True')

args = parser.parse_args()

def validate_args():
    assert os.path.exists(args.out_folder), args.out_folder

    assert os.path.isfile(args.init_fit), args.init_fit

    assert os.path.isfile(args.data_file), args.data_file

validate_args()

######################
# Load Data
######################
print('loading data from ', args.data_file)
data = np.load(args.data_file)
g_obs = np.array(data['g_obs'], dtype = int)

n_obs = g_obs.shape[0]
n_loci = g_obs.shape[1]

print('g_obs.shape', g_obs.shape)

######################
# LOAD INITIAL FIT 
######################
print('intial fit: ', args.init_fit)
vb_params_dict, vb_params_paragami, meta_data = \
    paragami.load_folded(args.init_fit)

# logitnormal parameters
gh_deg = int(meta_data['gh_deg'])
gh_loc, gh_weights = hermgauss(gh_deg)

gh_loc = np.array(gh_loc)
gh_weights = np.array(gh_weights)

print(vb_params_paragami)

# get prior parameters
prior_params_dict, prior_params_paragami = \
    structure_model_lib.get_default_prior_params()
prior_params_dict['dp_prior_alpha'] = np.array(meta_data['dp_prior_alpha'])
prior_params_dict['allele_prior_alpha'] = np.array(meta_data['allele_prior_alpha'])
prior_params_dict['allele_prior_beta'] = np.array(meta_data['allele_prior_beta'])

######################
# Get perturbed prior 
######################
epsilon_vec = np.linspace(0, 1, 12)[1:12]**2
saved_influence = np.load('./influence_grid.npz')
assert prior_params_dict['dp_prior_alpha'] == saved_influence['alpha0']

logit_v_grid = np.array(saved_influence['logit_v_grid'])
influence_grid = np.array(saved_influence['influence_grid'])

delta = saved_influence['delta']
epsilon = epsilon_vec[args.epsilon_indx]
print('Prior perturbation with epsilon = ', epsilon)
print('delta = ', delta)

if args.use_worst_case: 
    worst_case_pert = influence_lib.WorstCasePerturbation(influence_fun = None, 
                                                          logit_v_grid = logit_v_grid, 
                                                          cached_influence_grid = influence_grid, 
                                                          delta = delta)
    def get_e_log_perturbation(means, infos): 
        return epsilon * worst_case_pert.get_e_log_linf_perturbation(means.flatten(), 
                                                                     infos.flatten())

else: 
    def log_phi(logit_v):
        return - sp.stats.norm.pdf(logit_v, loc = -1.5, scale = 0.5)
        # return((logit_v < -0.55) * (logit_v > -2.56) * delta * -1)
    
    logit_v_grid = np.linspace(-5, 5, 200)
    scale_factor = np.abs(log_phi(logit_v_grid)).max()

    def rescaled_log_phi(logit_v): 
        return log_phi(logit_v) / scale_factor * delta

    def get_e_log_perturbation(means, infos): 
        return func_sens_lib.get_e_log_perturbation(rescaled_log_phi,
                                                    means, infos,
                                                    epsilon, 
                                                    gh_loc, gh_weights, 
                                                    sum_vector=True)

######################
# OPTIMIZE
######################
# optimize with preconditioner 
init_optim_time = time.time() 

vb_opt_dict, vb_opt, out, precond_objective = \
    s_optim_lib.run_preconditioned_lbfgs(g_obs, 
                                        vb_params_dict,
                                        vb_params_paragami,
                                        prior_params_dict,
                                        gh_loc = gh_loc,
                                        gh_weights = gh_weights,
                                        e_log_phi = lambda means, infos : \
                                                           get_e_log_perturbation(means, infos))

######################
# save optimizaiton results
######################
outfile = os.path.join(args.out_folder, args.out_filename)
print('saving structure model to ', outfile)

optim_time = time.time() - init_optim_time


# save final KL
final_kl = structure_model_lib.get_kl(g_obs, vb_opt_dict,
                            prior_params_dict,
                            gh_loc = gh_loc,
                            gh_weights = gh_weights, 
                            e_log_phi = lambda means, infos : 
                                      get_e_log_perturbation(means, infos))

# save paragami object
paragami.save_folded(outfile,
                     vb_opt_dict,
                     vb_params_paragami,
                     data_file = args.data_file,
                     epsilon = epsilon, 
                     dp_prior_alpha = prior_params_dict['dp_prior_alpha'],
                     allele_prior_alpha = prior_params_dict['allele_prior_alpha'],
                     allele_prior_beta = prior_params_dict['allele_prior_beta'],
                     gh_deg = gh_deg,
                     final_kl = final_kl,
                     optim_time = optim_time)

print('Total optimization time: {:03f} secs'.format(optim_time))


print('done. ')
