# utils for loading fast-structure model fits 

import jax.numpy as np
import numpy as onp

from structure_vb_lib.structure_model_lib import get_vb_params_paragami_object

def load_fs_file_into_numpy(filename): 
    handle = open(filename)
    out = onp.array([line.strip().split() for line in handle]).astype('float')
    handle.close()
    return out

def load_fs_means_vars(filename): 
    
    # individual admixtures
    ind_admix_mean = load_fs_file_into_numpy(filename + '.meanQ')
    ind_admix_var = load_fs_file_into_numpy(filename + '.varQ')
    
    # population frequencies
    pop_freq_mean = load_fs_file_into_numpy(filename + '.meanP')
    pop_freq_var = load_fs_file_into_numpy(filename + '.varP')
    
    return ind_admix_mean, ind_admix_var, pop_freq_mean, pop_freq_var

def load_fs_to_vb_params(filename): 
    
    # load fitted means
    _, ind_admix_var, _, pop_freq_var = \
        load_fs_means_vars(filename)
    
    n_obs = ind_admix_var.shape[0]
    n_loci = pop_freq_var.shape[0]
    k_approx = ind_admix_var.shape[1]
    assert pop_freq_var.shape[1] == (k_approx * 2)
    
    # set up vb parameters dictionary 
    vb_params_dict, vb_params_paragami = \
        get_vb_params_paragami_object(n_obs, n_loci, k_approx,
                                        use_logitnormal_sticks = False)
    
    # make the vb params dict onp for the moment
    # will change back later 
    vb_params_dict['pop_freq_beta_params'] = \
        onp.array(vb_params_dict['pop_freq_beta_params'])
    vb_params_dict['ind_admix_params']['stick_beta'] = \
        onp.array(vb_params_dict['ind_admix_params']['stick_beta'])
    
    # set beta parameters for pop frequency
    vb_params_dict['pop_freq_beta_params'][:, :, 0] = pop_freq_var[:, 0:k_approx]
    vb_params_dict['pop_freq_beta_params'][:, :, 1] = pop_freq_var[:, k_approx:]
    
    # set beta parameters for individual stick breaking
    betas = onp.flip(onp.cumsum(onp.flip(ind_admix_var, 1), 1), 1)
    vb_params_dict['ind_admix_params']['stick_beta'][:, :, 0] = ind_admix_var[:, 0:(k_approx-1)]
    vb_params_dict['ind_admix_params']['stick_beta'][:, :, 1] = betas[:, 1:]
    
    return vb_params_dict, vb_params_paragami