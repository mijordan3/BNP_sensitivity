import jax
import jax.numpy as np
import jax.scipy as sp

import paragami

from paragami.optimization_lib import _get_sym_matrix_inv_sqrt_funcs, \
                                        _get_matrix_from_operator

###################
# Preconditioner stuff for dirichlet parameters
###################
def _get_log_dirichlet_covariance(dirichlet_params, return_info, return_sqrt, v):
    # returns the covariance of the score function
    # of the beta distribution

    digamma_sum = sp.special.polygamma(1, dirichlet_params.sum())

    # get Fisher's information matrix
    digammas = sp.special.polygamma(1, dirichlet_params)
    diag_term = np.diag(digammas)
    cov_log_digamma = diag_term - digamma_sum

    # mulitply by alphas and betas because we are using
    # an unconstrained parameterization, where log(alpha) = free_param
    # TODO: better way to do this using autodiff?
    out = cov_log_digamma * np.outer(dirichlet_params, dirichlet_params)
    
    if return_sqrt: 
        matmul_funs = _get_sym_matrix_inv_sqrt_funcs(out)
    
    else: 
        matmul_funs = (lambda v : np.dot(out, v), 
                       lambda v : np.linalg.solve(out, v))
        
    if return_info: 
        return matmul_funs[1](v)
    else: 
        return matmul_funs[0](v)

def _eval_dirichlet_cov_matmul(dirichlet_params, return_info, return_sqrt, v):
    
    param_dim = dirichlet_params.shape[-1]
    
    xs = dirichlet_params.reshape(-1, param_dim)
    xs = np.concatenate((xs, v.reshape(-1, param_dim)), axis = 1)

    def f(x):

        cov = _get_log_dirichlet_covariance(x[0:param_dim], 
                                            return_info,
                                            return_sqrt,
                                            x[param_dim:])

        return cov
            
    out = jax.lax.map(f, xs = xs)

    return out.flatten()

###################
# Preconditioner stuff for multinomial parameters
###################
def _get_multinomial_covariance(multinom_params, return_info, return_sqrt, v):

    # get Fisher's information matrix
    cov = np.diag(multinom_params) - np.outer(multinom_params, multinom_params)

    # mulitply by jacobian 
    jac = paragami.simplex_patterns._constrain_simplex_jacobian(multinom_params)
    out = np.dot(np.dot(jac.transpose(), cov), jac)
    
    out = out + np.eye(out.shape[0]) * 1e-6
    
    # symmetrize for stability?
    out = 0.5 * (out + out.transpose())
    
    if return_sqrt: 
        matmul_funs = _get_sym_matrix_inv_sqrt_funcs(out)
    
    else: 
        matmul_funs = (lambda v : np.dot(out, v), 
                       lambda v : np.linalg.solve(out, v))
        
    if return_info: 
        return matmul_funs[1](v)
    else: 
        return matmul_funs[0](v)

def _eval_multinomial_cov_matmul(multinom_params, return_info, return_sqrt, v):
    
    param_dim = multinom_params.shape[-1]
    
    xs = multinom_params.reshape(-1, param_dim)
    xs = np.concatenate((xs, v.reshape(-1, param_dim - 1)), axis = 1)

    def f(x):

        cov = _get_multinomial_covariance(x[0:param_dim], return_info, return_sqrt, x[param_dim:])

        return cov
            
    out = jax.lax.map(f, xs = xs)

    return out.flatten()


###################
# Preconditioner stuff for normal parameters
###################
def _eval_normal_cov_matmul(infos, return_info, return_sqrt, v): 
    
    infos = infos.flatten()

    a1 = infos 
    a2 = 0.5
        
    if return_sqrt: 
        a1 = np.sqrt(infos)
        a2 = np.sqrt(a2)

    if return_info:
        matmul_v = np.concatenate((1/a1 * v[0:len(infos)], v[len(infos):] * 1/a2))
    else:
        matmul_v = np.concatenate((a1 * v[0:len(infos)], v[len(infos):] * a2))
        
    return matmul_v

###################
# combine everything
###################

def get_mfvb_cov_matmul(v, vb_params_dict,
                        vb_params_paragami,
                        return_info = False, 
                        return_sqrt = False):
    """
    Function that returns the (square root) MFVB covariance (information)
    times a vector `v`. 
    
    The argument `M` to jax.scipy.sparse.linalg.cg should be:
    
    M = lambda v : get_mfvb_cov_matmul(v, 
                                        vb_params_dict,
                                        vb_params_paragami,
                                        return_sqrt = False, 
                                        return_info = True)
    
    Parameters
    ----------
    v : vector 
        vector to be pre-mutiplied by the (square root) covariance (information). 
    vb_params_dict : dictionary
        Dictionary of variational parameters.
    vb_params_paragami : paragami patterned dictionary
        A paragami patterned dictionary that contains the variational parameters.
    return_info : boolean
        If `True`, returns the information matrix. If `False`, returns 
        the covariance. The input to the cg solver should be `True`. 
    return_sqrt : boolean
        Whether to take the square root of the covariance (information). 
        The input to the cg solver should be `False`. 
    
    Returns
    -------
    the (square root) MFVB covariance (information)
        times `v`.  
    """

    ##############
    # blocks for the population frequency
    block1_dim = vb_params_paragami['pop_freq_dirichlet_params'].flat_length(free = True)
    block1 = _eval_dirichlet_cov_matmul(vb_params_dict['pop_freq_dirichlet_params'], 
                                        return_info, 
                                        return_sqrt,
                                        v[0:block1_dim])
    d0 = block1_dim
    
    #############
    # blocks for population stick-breaking 
    block2_dim = vb_params_paragami['pop_stick_params'].flat_length(free = True)
    block2 = _eval_normal_cov_matmul(vb_params_dict['pop_stick_params']['stick_infos'],
                                     return_info,
                                     return_sqrt, 
                                     v[d0:(d0 + block2_dim)])
    
    d0 = d0 + block2_dim
    
    #############
    # blocks for multinomial topics
    block3_dim = vb_params_paragami['pop_indx_multinom_params'].flat_length(free = True)
    block3 = _eval_multinomial_cov_matmul(vb_params_dict['pop_indx_multinom_params'],
                                          return_info,
                                          return_sqrt, 
                                          v[d0:(d0 + block3_dim)])
    d0 = d0 + block3_dim 
    
    #############
    # blocks for population stick-breaking 
    block4_dim = vb_params_paragami['ind_admix_params'].flat_length(free = True)
    block4 = _eval_normal_cov_matmul(vb_params_dict['ind_admix_params']['stick_infos'],
                                     return_info, 
                                     return_sqrt, 
                                     v[d0:(d0 + block4_dim)])
    

    return np.concatenate((block1, block2, block3, block4))
