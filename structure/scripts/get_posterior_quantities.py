import jax

import jax.numpy as np
import jax.scipy as sp

import paragami

import matplotlib.pyplot as plt
%matplotlib inline

from bnpmodeling_runjingdev import modeling_lib, cluster_quantities_lib

from structure_vb_lib import structure_model_lib
from structure_vb_lib import posterior_quantities_lib

import re

# data file
parser.add_argument('--data_file', type=str)

# name of the structure fit file
parser.add_argument('--fit_file', type=str)


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
# Load fit
######################
vb_opt_dict, vb_params_paragami, _, _, \
    gh_loc, gh_weights, _ = \
        structure_model_lib.load_structure_fit(args.fit_file)

######################
# Number of clusters in loci
######################
n_clusters_sampled = \
    posterior_quantities_lib.\
        get_e_num_clusters(g_obs, 
                            vb_params_dict,
                            gh_loc,
                            gh_weights, 
                            threshold = 1000,
                            n_samples = 500,
                            prng_key = jax.random.PRNGKey(2342), 
                            return_samples = True)


######################
# Number of clusters in individuals
######################
stick_means = vb_opt_dict['ind_admix_params']['stick_means']
stick_infos = vb_opt_dict['ind_admix_params']['stick_infos']
    

n_pred_clusters_sampled = \
    posterior_quantities_lib.\
        get_e_num_pred_clusters(stick_means, 
                                stick_infos,
                                gh_loc,
                                gh_weights, 
                                n_samples = 500, 
                                prng_key = jax.random.PRNGKey(331))

outfile = re.sub('.npz', '_poststats.npz', args.init_fit)
print('saving posterior statistics into: ')
print(outfile)

np.savez(outfile, 
         n_clusters_sampled = n_clusters_sampled, 
         n_pred_clusters_sampled = n_pred_clusters_sampled)
print('done. ')