import jax

import jax.numpy as np
import jax.scipy as sp
from jax.scipy.sparse.linalg import cg

from numpy.polynomial.hermite import hermgauss

from vb_lib import structure_model_lib
from vb_lib.preconditioner_lib import get_mfvb_cov_matmul
import vb_lib.structure_optimization_lib as s_optim_lib

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
# the hessian vector product
###############
stru_hvp = jax.jit(lambda x : stru_objective.hvp(vb_opt, x))

print('compiling hessian vector products ...')
t0 = time.time()
_ = stru_hvp(vb_opt).block_until_ready()
print('Elapsed: {:.3f}'.format(time.time() - t0))

# define preconditioner
cg_precond = jax.jit(lambda v : get_mfvb_cov_matmul(v, vb_opt_dict,
                                            vb_params_paragami,
                                            return_sqrt = False, 
                                            return_info = True))

print('compiling preconditioner ...')
_ = cg_precond(vb_opt).block_until_ready()


###############
# cross-hessian
###############
alpha0 = prior_params_dict['dp_prior_alpha']
alpha_free = prior_params_paragami['dp_prior_alpha'].flatten(alpha0, 
                                                              free = True)
def _hyper_par_objective_fun(vb_params_dict, alpha): 
    
    _prior_params_dict = deepcopy(prior_params_dict)
    _prior_params_dict['dp_prior_alpha'] = alpha
    
    return structure_model_lib.get_kl(g_obs,
                                      vb_params_dict, _prior_params_dict,
                                      gh_loc = gh_loc, gh_weights = gh_weights)

hyper_par_objective_fun = paragami.FlattenFunctionInput(
                                original_fun=_hyper_par_objective_fun, 
                                patterns = [vb_params_paragami, prior_params_paragami['dp_prior_alpha']],
                                free = [True, True],
                                argnums = [0, 1])


dobj_dhyper = jax.jacobian(hyper_par_objective_fun, 1)
dobj_dhyper_dinput = jax.jit(jax.jacobian(dobj_dhyper), 0)

# compiling ... 
print('compiling cross-hess ...')
_ = dobj_dhyper_dinput(vb_opt, alpha_free).block_until_ready()

###############
# Get alpha sensitivity 
###############

print('Computing alpha sensitivity derivative... ')

# actual runtime: 
t0 = time.time() 
cross_hessian = dobj_dhyper_dinput(vb_opt, alpha_free)


dinput_dhyper = \
    - cg(A = stru_hvp, 
         b = cross_hessian.squeeze(),
         M = cg_precond)[0].block_until_ready()


alpha_derivative_time = time.time() - t0

print('Elapsed: {:.3f} secs'.format(alpha_derivative_time))

outfile = re.sub('.npz', '_lralpha', fit_file)
print('saving alpha derivative into: ', outfile)
np.savez(outfile, 
         dinput_dhyper = dinput_dhyper, 
         cross_hessian = cross_hessian,
         vb_opt = vb_opt, 
         alpha0 = alpha0, 
         kl = kl) 
      
print('done. ')
