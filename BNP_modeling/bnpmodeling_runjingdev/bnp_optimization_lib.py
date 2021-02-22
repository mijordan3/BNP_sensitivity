import jax
import jax.numpy as np

def update_stick_beta_params(ez, dp_prior_alpha): 
    
    # conditional on ez, returns the optimal beta parameters 
    # for the stick-breaking process. 
    
    # ez are allocations of shape n_obs x k_approx
    
    k_approx = ez.shape[1]
    
    weights = ez.sum(0)
    
    beta_update1 = weights[0:(k_approx-1)] + 1
    
    tmp = weights[1:k_approx]
    beta_update2 = np.cumsum(np.flip(tmp)) + \
                        (dp_prior_alpha - 1) + 1
    
    beta_update2 = np.flip(beta_update2)
    
    return beta_update1, beta_update2