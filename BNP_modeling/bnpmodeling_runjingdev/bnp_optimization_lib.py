import jax
import jax.numpy as np

import numpy as onp
from scipy.optimize import minimize

from sklearn.cluster import KMeans

from copy import deepcopy

import time

from bnpmodeling_runjingdev import stick_integration_lib
from bnpmodeling_runjingdev import modeling_lib
from bnpmodeling_runjingdev.sensitivity_lib import get_jac_hvp_fun

#########################
# function to initialize mixture modelswith k-means
#########################
def init_centroids_w_kmeans(y, 
                            k_approx,
                            n_kmeans_init = 10, 
                            seed = 1): 
    
    """
    Runs k-means on data set to initialize centroids
    
    Parameters
    ----------
    y : array
        The array of data, where each row is an observation. 
    k_approx : integer
        The number of cluster with which to run k-means. 
    n_kmeans_init : integer
        A number of times to run k-means, each with a 
        different random initialization. We return the run 
        with the smallest inertia. 
    seed : integer
        random seed. 
        
    Returns
    -------
    init_centroids : dictionary
        k_approx x dim of centroids
    km_best : 
        sklearn.cluster.KMeans output 
    """

    n_obs = np.shape(y)[0]
    dim = np.shape(y)[1]

    # K means init.
    for i in range(n_kmeans_init):
        km = KMeans(n_clusters = k_approx, 
                    random_state = seed).fit(y)
        inertia = km.inertia_
        if (i == 0):
            inertia_best = inertia
            km_best = deepcopy(km)
        elif (inertia < inertia_best):
            inertia_best = inertia
            km_best = deepcopy(km)
    
    init_centroids = np.array(km_best.cluster_centers_)
    
    return init_centroids, km_best

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


#########################
# Function to convert beta sticks to 
# logitnormal sticks
#########################
def convert_beta_sticks_to_logitnormal(stick_betas, 
                                       logitnorm_stick_params_dict,
                                       logitnorm_stick_params_paragami, 
                                       gh_loc, gh_weights): 
    """
    Given a set of beta parameters for stick-breaking proportions, 
    return the logitnormal stick parameters that have the same
    expected log(stick) and expected log(1 - stick). 
    
    Parameters
    ----------
    stick_betas : array
        array (... x 2) of beta parameters 
        on individual admixture stick-breaking proportions.
    logitnorm_stick_params_dict : dictionary
        parameter dictionary of logitnormal parameters
        (stick_means, stick_infos) for individual admixture
        stick-breaking proportions
    logitnorm_stick_params_paragami : paragami patterned dictionary
        A paragami patterned dictionary that contains the variational
        parameters
    gh_loc : vector
        Locations for gauss-hermite quadrature. 
    gh_weights : vector
        Weights for gauss-hermite quadrature. 
        
    Returns
    -------
    opt_logitnorm_stick_params : dictionary
        A dictionary that contains the variational parameters
        for individual admixture stick-breaking
        proportions. 
    out : scipy.optimize.minimize output
    """

    
    # check shapes
    assert stick_betas.shape[-1] == 2
    
    # the moments from the beta parameters
    target_sticks, target_1m_sticks = modeling_lib.get_e_log_beta(stick_betas)
    
    # square error loss
    def _loss(stick_params_free): 

        logitnorm_stick_params_dict = \
            logitnorm_stick_params_paragami.fold(stick_params_free, 
                                                 free = True)

        stick_means = logitnorm_stick_params_dict['stick_means']
        stick_infos = logitnorm_stick_params_dict['stick_infos']

        e_log_sticks, e_log_1m_sticks = \
            stick_integration_lib.get_e_log_logitnormal(
                lognorm_means = stick_means,
                lognorm_infos = stick_infos,
                gh_loc = gh_loc,
                gh_weights = gh_weights)
    
        loss = (e_log_sticks - target_sticks)**2 +\
                (e_log_1m_sticks - target_1m_sticks)**2
        
        return loss.sum()
    
    # optimize
    loss = jax.jit(_loss)
    loss_grad = jax.jit(jax.grad(_loss))
    loss_hvp = jax.jit(get_jac_hvp_fun(_loss))
    
    stick_params_free = \
        logitnorm_stick_params_paragami.flatten(logitnorm_stick_params_dict, 
                                                free = True)
    
    out = minimize(fun = lambda x : onp.array(loss(x)), 
                                  x0 = stick_params_free, 
                                  jac = lambda x : onp.array(loss_grad(x)), 
                                  hessp = lambda x,v : onp.array(loss_hvp(x, v)), 
                                  method = 'trust-ncg')
    
    opt_logitnorm_stick_params = \
        logitnorm_stick_params_paragami.fold(out.x, free = True)
    
    return opt_logitnorm_stick_params, out


#################
# A generic optimizer
#################
def optimize_kl(get_kl_loss,
               vb_params_dict, 
               vb_params_paragami,
               get_grad = None,
               get_hvp = None,
               run_lbfgs = True,
               run_newton = True): 
    """
    Parameters 
    ----------
    get_kl_loss : callable
        Objective as function of vb parameters (in flattened space)
        and prior parameter. 
    vb_params_dict : dictionary
        A dictionary that contains the initial variational parameters.
    vb_params_paragami : paragami patterned dictionary
        A paragami patterned dictionary that contains the variational parameters.
    get_grad : callable, optional
         Returns the gradient of `get_kl_loss` as 
         function of vb parameters (in flattened space). 
         If none, this is computed automatically using jax derivatives.
    get_hvp : callable, optional
        Returns the hessian vector product as 
        function of vb parameters (in flattened space) and 
        and some vector of equal length as the vb parameters.
        If none, this is computed automatically using jax derivatives.
    run_lbfgs : boolean, optional
        Whether to run LBFGS. At least one of `run_blfgs` and 
        `run_newton` must be true. 
    run_newton : boolean, optional
        Whether to run newton-ncg. At least one of `run_blfgs` and 
        `run_newton` must be true. 
        
    Returns
    ----------
    vb_opt_dict : dictionary
        A dictionary that contains the optimized variational parameters.
    vb_opt : array 
        The unconstrained vector of optimized variational parameters.
    out : 
        The output of scipy.optimize.minimize.
    optim_time : 
        The time elapsed, not including compile times. 
    """
    
    # at least one should be true
    assert run_lbfgs or run_newton
    
    get_loss = jax.jit(get_kl_loss)
    
    if get_grad is None: 
        get_grad = jax.jit(jax.grad(get_kl_loss))

    if get_hvp is None: 
        get_hvp = jax.jit(get_jac_hvp_fun(get_kl_loss))
    
    ################
    # compile objective functions
    ################
    # intial point
    x0 = vb_params_paragami.flatten(vb_params_dict, free = True)
    
    print('compiling objective and derivatives ... ')
    t0 = time.time()
    _ = get_loss(x0).block_until_ready()
    _ = get_grad(x0).block_until_ready()
    if run_newton: 
        _ = get_hvp(x0, x0).block_until_ready()
    print('done. Compile time: {0:.03f}sec'.format(time.time() - t0))
    
    ################
    # initialize with L-BFGS-B
    ################
    t0 = time.time() 
    if run_lbfgs: 
        print('Running L-BFGS-B ...')
        out = minimize(fun = lambda x : onp.array(get_loss(x)), 
                             x0 = x0, 
                             method = 'L-BFGS-B', 
                             jac = lambda x : onp.array(get_grad(x)))    
        print('L-BFGS-B time: {:.03f}sec'.format(time.time() - t0))
        lbfgs_opt = out.x
        print('BFGS out: ', out.message)
    else: 
        lbfgs_opt = x0
        
    ################
    # run a few more newton steps
    ################
    if run_newton: 
        t1 = time.time() 
        print('Running trust-ncg ... ')
        out = minimize(fun = lambda x : onp.array(get_loss(x)), 
                   x0 = lbfgs_opt, 
                   method = 'trust-ncg', 
                   jac = lambda x : onp.array(get_grad(x)), 
                   hessp = lambda x,v : onp.array(get_hvp(x, v)))
        print('Newton time: {:.03f}sec'.format(time.time() - t1))
    
    optim_time = time.time() - t0
    
    vb_opt = out.x
    vb_opt_dict = vb_params_paragami.fold(vb_opt, free = True)
    print('Newton out: ', out.message)
    
    print('done. ')
    
    return vb_opt_dict, vb_opt, out, optim_time
