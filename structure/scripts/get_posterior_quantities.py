import jax

import jax.numpy as np
import jax.scipy as sp

import paragami

import matplotlib.pyplot as plt

from bnpmodeling_runjingdev import modeling_lib, cluster_quantities_lib

from structure_vb_lib import structure_model_lib
from structure_vb_lib import posterior_quantities_lib

import re
import time

import argparse
parser = argparse.ArgumentParser()

# data file
parser.add_argument('--data_file', type=str)

# name of the structure fit file
parser.add_argument('--fit_file', type=str)

# name of lr file 
parser.add_argument('--lr_file', type=str)

# TODO with latest change, this should be necessary ... 
parser.add_argument('--perturbation', type = str)

args = parser.parse_args()

t0 = time.time()


threshold_insamp = 1000
threshold_insamp2 = 500
threshold_pred = 3

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
print('loading fit from ', args.fit_file)
vb_refit_dict, vb_params_paragami, prior_params_dict, _, \
    gh_loc, gh_weights, meta_data = \
        structure_model_lib.load_structure_fit(args.fit_file)

epsilon = meta_data['epsilon']
delta = meta_data['delta']

print('epsilon = ', epsilon)

######################
# Load linear response prediction for this fit
######################
print('loading derivatives from: ', args.lr_file)
lr_data = np.load(args.lr_file)

dinput_dhyper = lr_data['dinput_dfun_' + args.perturbation]

lr_vb_params = lr_data['vb_opt'] + dinput_dhyper * epsilon * delta
vb_lr_dict = vb_params_paragami.fold(lr_vb_params, free = True)

######################
# Number loci per cluster
######################
saved_results = dict()

saved_results['e_n_loci_refit']= \
    posterior_quantities_lib.get_e_num_loci_per_cluster(g_obs,
                                                        vb_refit_dict, 
                                                        gh_loc,
                                                        gh_weights)
saved_results['e_n_loci_lr'] = \
    posterior_quantities_lib.get_e_num_loci_per_cluster(g_obs,
                                                        vb_lr_dict, 
                                                        gh_loc,
                                                        gh_weights)



######################
# Number individuals per cluster
######################
saved_results['e_n_ind_refit'] = \
    posterior_quantities_lib.get_e_num_ind_per_cluster(vb_refit_dict, 
                                                       gh_loc,
                                                       gh_weights)

saved_results['e_n_ind_lr'] = \
    posterior_quantities_lib.get_e_num_ind_per_cluster(vb_lr_dict, 
                                                       gh_loc,
                                                       gh_weights)

######################
# Number of clusters in loci
######################

# this one is super slow ... 
def get_e_n_clusters(vb_params_dict, threshold):

    return posterior_quantities_lib.\
        get_e_num_clusters(g_obs, 
                            vb_params_dict,
                            gh_loc,
                            gh_weights, 
                            threshold = threshold,
                            n_samples = 500,
                            prng_key = jax.random.PRNGKey(2342))

get_e_n_clusters1 = jax.jit(lambda x : get_e_n_clusters(x, threshold_insamp))

saved_results['e_n_clusters_refit'] = get_e_n_clusters1(vb_refit_dict)
saved_results['e_n_clusters_lr'] = get_e_n_clusters1(vb_lr_dict)
saved_results['threshold_insamp'] = threshold_insamp

get_e_n_clusters2 = jax.jit(lambda x : get_e_n_clusters(x, threshold_insamp2))

saved_results['e_n_clusters_refit2'] = get_e_n_clusters2(vb_refit_dict)
saved_results['e_n_clusters_lr2'] = get_e_n_clusters2(vb_lr_dict)
saved_results['threshold_insamp2'] = threshold_insamp2

######################
# Number of clusters in individuals
######################
def get_e_n_pred_clusters(vb_params_dict, threshold):

    return posterior_quantities_lib.\
            get_e_num_pred_clusters(vb_params_dict,
                                    gh_loc,
                                    gh_weights, 
                                    threshold = threshold,
                                    n_samples = 1000, 
                                    prng_key = jax.random.PRNGKey(331))

    
# not thresholded
saved_results['e_n_pred_clusters_refit'] = \
    get_e_n_pred_clusters(vb_refit_dict, threshold = 0)
saved_results['e_n_pred_clusters_lr'] = \
    get_e_n_pred_clusters(vb_lr_dict, threshold = 0)

# thresholded
saved_results['e_n_pred_clusters_refit_thresh'] = \
    get_e_n_pred_clusters(vb_refit_dict, threshold = threshold_pred)
saved_results['e_n_pred_clusters_lr_thresh'] = \
    get_e_n_pred_clusters(vb_lr_dict, threshold = threshold_pred)

saved_results['threshold_pred'] = threshold_pred


##################
# Save results
##################
outfile = re.sub('.npz', '_poststats.npz', args.fit_file)
print('saving posterior statistics into: ')
print(outfile)

# some extra data to save
saved_results['epsilon'] = epsilon
saved_results['delta'] = delta
saved_results['dp_prior_alpha'] = meta_data['dp_prior_alpha']

np.savez(outfile, **saved_results)


print('done. ')
print('elapsed: ', np.round(time.time() - t0, 3), 'secs')