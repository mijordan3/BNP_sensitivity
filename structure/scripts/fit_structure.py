import jax

import jax.numpy as np
import jax.scipy as sp

import numpy as onp
from numpy.polynomial.hermite import hermgauss

import structure_vb_lib.structure_model_lib as structure_model_lib
import structure_vb_lib.cavi_lib as cavi_lib
import structure_vb_lib.structure_optimization_lib as s_optim_lib

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

# whether to use a warm start
parser.add_argument('--warm_start', type=distutils.util.strtobool, default='False')
parser.add_argument('--init_fit', type=str)

# whether to initialize with cavi
parser.add_argument('--init_cavi_steps', type=int, default=200)

# model parameters
parser.add_argument('--alpha', type=float, default = 4.0)
parser.add_argument('--k_approx', type = int, default = 15)

args = parser.parse_args()

def validate_args():
    assert os.path.exists(args.out_folder), args.out_folder

    if args.warm_start:
        assert os.path.isfile(args.init_fit), args.init_fit

    assert os.path.isfile(args.data_file), args.data_file

validate_args()

onp.random.seed(args.seed)

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
# GET PRIOR
######################
prior_params_dict, prior_params_paragami = \
    structure_model_lib.get_default_prior_params()

prior_params_dict['dp_prior_alpha'] = np.array([args.alpha])

print('prior params: ')
print(prior_params_dict)

######################
# GET VB PARAMS AND INITIALIZE
######################
k_approx = args.k_approx
gh_deg = 8
gh_loc, gh_weights = hermgauss(gh_deg)

init_optim_time = time.time() 

cavi_init_time = 0.

if args.warm_start:
    print('warm start from ', args.init_fit)
    vb_params_dict, vb_params_paragami, _ = \
        paragami.load_folded(args.init_fit)
    
else:     
    vb_params_dict, vb_params_paragami = \
        structure_model_lib.\
            get_vb_params_paragami_object(n_obs, 
                                          n_loci,
                                          k_approx,
                                          use_logitnormal_sticks = True, 
                                          seed = args.seed)
    # initialize with some cavi steps
    if args.init_cavi_steps > 0: 
        vb_params_dict, cavi_init_time = \
            s_optim_lib.initialize_with_cavi(g_obs, 
                                 vb_params_paragami, 
                                 prior_params_dict, 
                                 gh_loc, gh_weights, 
                                 print_every = 20, 
                                 max_iter = args.init_cavi_steps, 
                                 seed = args.seed)


print(vb_params_paragami)

######################
# OPTIMIZE
######################
# optimize with preconditioner 
vb_opt_dict, vb_opt, out, precond_objective, lbfgs_time = \
    s_optim_lib.run_preconditioned_lbfgs(g_obs, 
                        vb_params_dict, 
                        vb_params_paragami,
                        prior_params_dict,
                        gh_loc, gh_weights)

######################
# save optimizaiton results
######################
outfile = os.path.join(args.out_folder, args.out_filename)
print('saving structure model to ', outfile)

optim_time = cavi_init_time + lbfgs_time
print('Optim time (ignoring compilation time) {:.3f}secs'.format(optim_time))

# save final KL
final_kl = structure_model_lib.get_kl(g_obs, vb_opt_dict,
                            prior_params_dict,
                            gh_loc = gh_loc,
                            gh_weights = gh_weights)

# save paragami object
structure_model_lib.save_structure_fit(outfile, 
                                       vb_opt_dict,
                                       vb_params_paragami, 
                                       prior_params_dict,
                                       gh_deg, 
                                       data_file = args.data_file, 
                                       final_kl = final_kl, 
                                       optim_time = optim_time)

print('Total optim time: {:.3f} secs'.format(time.time() - init_optim_time))


print('done. ')
