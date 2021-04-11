import jax
import jax.numpy as np
import jax.scipy as sp

from structure_vb_lib import structure_model_lib 

from bnpmodeling_runjingdev import cluster_quantities_lib, modeling_lib


##############
# Some useful expectations for plotting
##############
def get_vb_expectations(vb_params_dict, gh_loc, gh_weights): 
    
    e_ind_admix = cluster_quantities_lib.get_e_cluster_probabilities(
                        vb_params_dict['ind_admix_params']['stick_means'], 
                        vb_params_dict['ind_admix_params']['stick_infos'],
                        gh_loc, gh_weights)

    e_pop_freq = modeling_lib.get_e_dirichlet(vb_params_dict['pop_freq_dirichlet_params'])
    
    return e_ind_admix, e_pop_freq

######################
# function to return ez's 
######################
def get_optimal_z_from_vb_dict(g_obs, vb_params_dict, gh_loc, gh_weights): 
        
    e_log_sticks, e_log_1m_sticks, \
        e_log_cluster_probs, e_log_pop_freq = \
            structure_model_lib.get_moments_from_vb_params_dict(vb_params_dict,
                                    gh_loc = gh_loc,
                                    gh_weights = gh_weights)
    
    return structure_model_lib.get_optimal_z(g_obs,
                                             e_log_pop_freq,
                                             e_log_cluster_probs)[0]

######################
# expected number of clusters
######################
def get_e_num_clusters(g_obs, vb_params_dict, gh_loc, gh_weights, 
                        threshold = 0,
                        n_samples = 1000,
                        prng_key = jax.random.PRNGKey(0)): 
    
    # expected number of clusters within the observed loci
    
    e_z = get_optimal_z_from_vb_dict(g_obs, vb_params_dict, gh_loc, gh_weights)
    
    
    # combine n_obs, n_loci, and chromosome dimensions
    k_approx = e_z.shape[-1]
    e_z = e_z.reshape(-1, k_approx)
    
    if threshold == 0: 
        # if threshold is zero, we can return the analytic expectation
        return cluster_quantities_lib.get_e_num_clusters_from_ez_analytic(e_z)
    else: 
        return cluster_quantities_lib.get_e_num_clusters_from_ez(e_z,
                                                                 threshold = threshold,
                                                                 n_samples = n_samples,
                                                                 prng_key = prng_key)


def get_e_num_pred_clusters(vb_params_dict,
                            gh_loc, gh_weights, 
                            n_samples = 1000,
                            threshold = 0, 
                            prng_key = jax.random.PRNGKey(0), 
                            return_samples = False): 
    
    # If I sample one more loci for every individual in my dataset, 
    # how many clusters would I expect to see?
    
    stick_means = vb_params_dict['ind_admix_params']['stick_means']
    stick_infos = vb_params_dict['ind_admix_params']['stick_infos']
    
    # sample sticks: shape is n_samples x n_obs x (k_approx - 1)
    sticks_sampled = cluster_quantities_lib.sample_stick_propn(stick_means, 
                                                               stick_infos, 
                                                               n_samples, 
                                                               prng_key)
    
    # get mixture weights: shape is n_samples x n_obs x k_approx 
    ind_admix_sampled = \
        cluster_quantities_lib.\
            get_mixture_weights_from_stick_break_propns(sticks_sampled)
        
    # for each mixture weight -- pretend these are e_zs 
    # and compute number of clusters
    if threshold == 0: 
        get_sampled_n_clusters = jax.vmap(cluster_quantities_lib.\
                                          get_e_num_clusters_from_ez_analytic)
        n_clusters_sampled = get_sampled_n_clusters(ind_admix_sampled)
    else: 
        
        # sample one z from each individual admixture
        sample_one_z = lambda *x : \
            cluster_quantities_lib.sample_ez(x[0],
                                             n_samples = 1,
                                             prng_key = x[1]).squeeze(0)
        
        sample_all_zs = jax.vmap(sample_one_z)
        
        keys = jax.random.split(prng_key, n_samples)
        
        # this is shape n_samples x n_obs x k_approx
        z_samples = sample_all_zs(ind_admix_sampled, keys)
        
        # sum over observations
        counts_per_clusters_sampled = z_samples.sum(1)
        
        # get number of clusters above threshold
        n_clusters_sampled = (counts_per_clusters_sampled >= threshold).sum(-1)
        
    if return_samples: 
        return n_clusters_sampled
    else: 
        return n_clusters_sampled.mean()

######################
# expected cluster weights
######################    
def get_e_num_loci_per_cluster(g_obs, vb_params_dict, gh_loc, gh_weights): 
    
    # expected number of clusters within the observed loci
    
    e_z = get_optimal_z_from_vb_dict(g_obs, vb_params_dict, gh_loc, gh_weights)
            
    k_approx = e_z.shape[-1]
    return e_z.reshape(-1, k_approx).sum(0)

def get_e_num_ind_per_cluster(vb_params_dict, gh_loc, gh_weights): 
        
    stick_means = vb_params_dict['ind_admix_params']['stick_means']
    stick_infos = vb_params_dict['ind_admix_params']['stick_infos']
    
    e_ind_admix = cluster_quantities_lib.get_e_cluster_probabilities(stick_means, 
                                                                     stick_infos, 
                                                                     gh_loc, 
                                                                     gh_weights)
    
    return e_ind_admix.sum(0)


