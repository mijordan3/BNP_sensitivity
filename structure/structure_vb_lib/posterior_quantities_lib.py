import jax
import jax.numpy as np
import jax.scipy as sp

from structure_vb_lib import structure_model_lib 

import bnpmodeling_runjingdev.exponential_families as ef
from bnpmodeling_runjingdev import cluster_quantities_lib, modeling_lib


######################
# function to return ez's for the l-th locus
######################
def get_optimal_ezl_free(g_obs_l, 
                    e_log_pop_freq_l,
                    e_log_1m_pop_freq_l,
                    e_log_cluster_probs): 
    
    loglik_cond_z_l = \
        structure_model_lib.get_loglik_cond_z_l(g_obs_l, 
                                                e_log_pop_freq_l,
                                                e_log_1m_pop_freq_l,
                                                e_log_cluster_probs)

    return loglik_cond_z_l

######################
# expected number of clusters
######################
def get_e_num_clusters(g_obs, vb_params_dict, gh_loc, gh_weights, 
                        threshold = 0,
                        n_samples = 1000,
                        prng_key = jax.random.PRNGKey(0), 
                        return_samples = True): 
    
    # expected number of clusters within the observed loci
    
    e_log_sticks, e_log_1m_sticks, \
        e_log_pop_freq, e_log_1m_pop_freq = \
            structure_model_lib.get_moments_from_vb_params_dict(vb_params_dict,
                                    gh_loc = gh_loc,
                                    gh_weights = gh_weights)
    e_log_cluster_probs = \
        modeling_lib.get_e_log_cluster_probabilities_from_e_log_stick(
                            e_log_sticks, e_log_1m_sticks)
    
    n_obs = g_obs.shape[0]
    n_loci = g_obs.shape[1]
    k_approx = e_log_cluster_probs.shape[-1]
    
    
    def f(val, x): 
        # x[0] is g_obs[:, l] 
        # x[1] is e_log_pop_freq[:, l]
        # x[2] is e_log_1m_pop_freq[:, l]
        # x[3] is a sequence of subkeys
        
        # e_z_l is shaped as n_obs x k_approx x 2
        e_z_l_free = get_optimal_ezl_free(x[0], x[1], x[2], e_log_cluster_probs)
            
        # combine first and last dimension
        e_z_l_free = e_z_l_free.transpose((0, 2, 1)).\
                            reshape((n_obs * 2, k_approx))
            
        # this is n_samples x (n_obs * 2) x k_approx
        z_samples = cluster_quantities_lib.sample_ez(e_z_l_free, 
                                                     n_samples = n_samples, 
                                                     prng_key = x[3], 
                                                     e_z_is_free = True)
        
        # this is n_samples x k_approx
        sampled_counts_l = z_samples.sum(1)
        
        return sampled_counts_l + val, None
    
    # random keys
    key, *subkeys = jax.random.split(prng_key, n_loci+1)
    subkeys = np.stack(subkeys)
    
    # loop over loci and sum
    sampled_counts = np.zeros((n_samples, k_approx))
    sampled_counts = jax.lax.scan(f,
                                  init = sampled_counts, 
                                  xs = (g_obs.transpose((1, 0, 2)),
                                        e_log_pop_freq, 
                                        e_log_1m_pop_freq, 
                                        subkeys))[0]
                                  
    # the number of clusters above some threshold
    n_clusters_sampled = (sampled_counts > threshold).sum(1)
    
    if return_samples: 
        return n_clusters_sampled
    else: 
        # just return the monte carlo estimate
        return n_clusters_sampled.mean()

def get_e_num_pred_clusters(stick_means, stick_infos, gh_loc, gh_weights, 
                            n_samples = 1000,
                            seed = 0, 
                            return_samples = False): 
    
    # If I sample one more loci for every individual in my dataset, 
    # how many clusters would I expect to see?
    
    # sample sticks: shape is n_samples x n_obs x (k_approx - 1)
    sticks_sampled = cluster_quantities_lib.sample_stick_propn(stick_means, 
                                                               stick_infos, 
                                                               n_samples, 
                                                               seed)
    
    # get mixture weights: shape is n_samples x n_obs x k_approx 
    ind_admix_sampled = \
        cluster_quantities_lib.\
            get_mixture_weights_from_stick_break_propns(sticks_sampled)
        
    # for each mixture weight -- pretend these are e_zs 
    # and compute number of clusters
    get_sampled_n_clusters = jax.vmap(cluster_quantities_lib.\
                                      get_e_num_clusters_from_ez_analytic)
    n_clusters_sampled = get_sampled_n_clusters(ind_admix_sampled)
    
    if return_samples: 
        return n_clusters_sampled
    else: 
        return n_clusters_sampled.mean()

    
def get_e_num_loci_per_cluster(g_obs, vb_params_dict, gh_loc, gh_weights): 
    
    # expected number of clusters within the observed loci
    
    e_log_sticks, e_log_1m_sticks, \
        e_log_pop_freq, e_log_1m_pop_freq = \
            structure_model_lib.get_moments_from_vb_params_dict(vb_params_dict,
                                    gh_loc = gh_loc,
                                    gh_weights = gh_weights)
    e_log_cluster_probs = \
        modeling_lib.get_e_log_cluster_probabilities_from_e_log_stick(
                            e_log_sticks, e_log_1m_sticks)
    
    n_obs = g_obs.shape[0]
    n_loci = g_obs.shape[1]
    k_approx = e_log_cluster_probs.shape[-1]
    
    
    def f(val, x): 
        # x[0] is g_obs[:, l] 
        # x[1] is e_log_pop_freq[:, l]
        # x[2] is e_log_1m_pop_freq[:, l]
        
        # e_z_l is shaped as n_obs x k_approx x 2
        e_z_l_free = get_optimal_ezl_free(x[0], x[1], x[2], e_log_cluster_probs)
        e_z_l = jax.nn.softmax(e_z_l_free, 1)
        # sum all dimensions except for k_approx
        counts_per_cluster = e_z_l.sum(0).sum(-1)
        
        return counts_per_cluster + val, None
    
    counts_per_cluster = np.zeros(k_approx)
    counts_per_cluster = jax.lax.scan(f,
                                  init = counts_per_cluster, 
                                  xs = (g_obs.transpose((1, 0, 2)),
                                        e_log_pop_freq, 
                                        e_log_1m_pop_freq))[0]
                                  
    return counts_per_cluster

    
###############
# Function to return cluster belongings for all loci
###############
def get_ez_all(g_obs, vb_params_dict, gh_loc, gh_weights): 
    
    # returns a n_obs x n_loci x k_approx x 2 array of
    # cluster probabilities
    
    # warning! the whole matrix of ez may be large. 
    # use only for small datasets
    
    e_log_sticks, e_log_1m_sticks, \
        e_log_pop_freq, e_log_1m_pop_freq = \
            structure_model_lib.get_moments_from_vb_params_dict(vb_params_dict,
                                    gh_loc = gh_loc,
                                    gh_weights = gh_weights)
    e_log_cluster_probs = \
        modeling_lib.get_e_log_cluster_probabilities_from_e_log_stick(
            e_log_sticks, e_log_1m_sticks)
    
    
    get_optimal_ez = jax.vmap(lambda x : get_optimal_ezl(*x, e_log_cluster_probs))
    
    ez = get_optimal_ez((g_obs.transpose((1, 0, 2)), 
                         e_log_pop_freq, 
                         e_log_1m_pop_freq))
    
    return ez.transpose((1, 0, 2, 3))
