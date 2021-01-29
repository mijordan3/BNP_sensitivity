import jax
import jax.numpy as np
import jax.scipy as sp

from jax.experimental import loops

from scipy import optimize 

import numpy as onp
from sklearn.decomposition import NMF

import paragami

from structure_vb_lib import structure_model_lib
from structure_vb_lib.cavi_lib import run_cavi
from structure_vb_lib.preconditioner_lib import get_mfvb_cov_matmul

import bnpmodeling_runjingdev.exponential_families as ef
from bnpmodeling_runjingdev import cluster_quantities_lib, modeling_lib
from bnpmodeling_runjingdev.sensitivity_lib import get_jac_hvp_fun

import time 
    
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
        array (n_obs x (k_approx - 1) x 2) of beta parameters 
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
    assert logitnorm_stick_params_dict['stick_means'].shape[0] == \
                stick_betas.shape[0]
    assert logitnorm_stick_params_dict['stick_means'].shape[1] == \
                stick_betas.shape[1]
    assert stick_betas.shape[2] == 2
    
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
            ef.get_e_log_logitnormal(
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
    
    out = optimize.minimize(fun = lambda x : onp.array(loss(x)), 
                                  x0 = stick_params_free, 
                                  jac = lambda x : onp.array(loss_grad(x)), 
                                  hessp = lambda x,v : onp.array(loss_hvp(x, v)), 
                                  method = 'trust-ncg')
    
    opt_logitnorm_stick_params = \
        logitnorm_stick_params_paragami.fold(out.x, free = True)
    
    return opt_logitnorm_stick_params, out

#########################
# Initializes model with some CAVI steps
#########################
def initialize_with_cavi(g_obs, 
                         vb_params_paragami, 
                         prior_params_dict, 
                         gh_loc, gh_weights, 
                         print_every = 20, 
                         debug_cavi = False,
                         max_iter = 100, 
                         seed = 0): 
    """
    Initializes model with by running CAVI steps on the model 
    using on beta-sticks. Then, convert beta sticks to logit-normal sticks. 
    The returned parameter dictionary contains the logit-normal sticks. 
    
    Parameters
    ----------
    g_obs : ndarray
        The array of one-hot encoded genotypes, of shape (n_obs, n_loci, 3)
    vb_params_paragami : paragami patterned dictionary
        A paragami patterned dictionary that contains the variational parameters.
    prior_params_dict : dictionary
        Dictionary of prior parameters.
        parameters
    gh_loc : vector
        Locations for gauss-hermite quadrature. 
    gh_weights : vector
        Weights for gauss-hermite quadrature. 
    debug_cavi : boolean
        whether to check the KL after every cavi step to assert
        the KL decreased. 
        (Only used for testing. Computing the KL may be expensive). 
        
    Returns
    -------
    vb_params_dict : dictionary
        A dictionary that contains the variational parameters
        after cavi optimization. 
    cavi_time : float
        Optimization time (excluding compiling). 
    """

    # this is just a place-holder
    vb_params_dict = vb_params_paragami.random()
    
    # read off data dimensions
    n_obs = vb_params_dict['ind_admix_params']['stick_means'].shape[0]
    n_loci = vb_params_dict['pop_freq_beta_params'].shape[0]
    k_approx = vb_params_dict['pop_freq_beta_params'].shape[1]
    
    # random init: these are beta sticks!
    vb_params_dict_beta, vb_params_paragami_beta = \
        structure_model_lib.get_vb_params_paragami_object(n_obs, 
                                                          n_loci,
                                                          k_approx,
                                                          use_logitnormal_sticks = False, 
                                                          seed = seed)
    
    # run cavi
    vb_params_dict_beta, _, _, cavi_time = \
        run_cavi(g_obs, 
                 vb_params_dict_beta,
                 vb_params_paragami_beta,
                 prior_params_dict, 
                 print_every = print_every, 
                 debug = debug_cavi,
                 max_iter = max_iter)
    
    cavi_time = cavi_time[-1]
    
    # convert to logitnormal sticks 
    t0 = time.time()
    
    stick_betas = vb_params_dict_beta['ind_admix_params']['stick_beta']
    lnorm_stick_params_dict = vb_params_dict['ind_admix_params']
    lnorm_stick_params_paragami = vb_params_paragami['ind_admix_params']
    
    stick_params_dict, out = \
        convert_beta_sticks_to_logitnormal(stick_betas, 
                                           lnorm_stick_params_dict,
                                           lnorm_stick_params_paragami, 
                                           gh_loc, gh_weights)
    conversion_time = time.time() - t0
    print('Stick conversion time: {:.3f}secs'.format(conversion_time))
    
    # update vb_params
    vb_params_dict['pop_freq_beta_params'] = vb_params_dict_beta['pop_freq_beta_params']
    vb_params_dict['ind_admix_params'] = stick_params_dict
    
    return vb_params_dict, cavi_time + conversion_time

#########################
# The structure objective
#########################
class StructureObjective():
    """
    This class contains the structure objective. 
    The method self.f returns the KL objective 
    as a function of vb parameters in flattened space. 
    
    The methods self.grad and self.hvp return the 
    gradient and hessian vector product, respectively. 
    """
    
    def __init__(self,
                 g_obs, 
                 vb_params_paragami,
                 prior_params_dict, 
                 gh_loc, gh_weights, 
                 e_log_phi = None, 
                 jit_functions = True): 
        """
        Parameters
        ----------
        g_obs : ndarray
            The array of one-hot encoded genotypes, of shape (n_obs, n_loci, 3)
        vb_params_paragami : paragami patterned dictionary
            A paragami patterned dictionary that contains the variational parameters.
        prior_params_dict : dictionary
            Dictionary of prior parameters.
            parameters
        gh_loc : vector
            Locations for gauss-hermite quadrature. 
        gh_weights : vector
            Weights for gauss-hermite quadrature. 
        e_log_phi : function
            Function with arguments stick_means and stick_infos 
            and returns the expected log-multiplicative perturbation.
        jit_functions : boolean
            Whether or not to call jax.jit on the function and 
            gradients
        """

        self.g_obs = g_obs
        self.vb_params_paragami = vb_params_paragami 
        self.prior_params_dict = prior_params_dict 
            
        self.dim_vb_free = len(vb_params_paragami.flatten(\
                                vb_params_paragami.random(), \
                                free = True))
        
        self.gh_loc = gh_loc
        self.gh_weights = gh_weights 
        self.e_log_phi = e_log_phi 
        
        self._grad = jax.grad(self._kl_ps_loss)

        self.jit_functions = jit_functions
        self._jit_functions()
        if self.jit_functions: 
            self._compile_functions()
        

    def _f(self, vb_free_params):
        # this is the objective function 
        
        vb_params_dict = self.vb_params_paragami.fold(vb_free_params, free = True)

        return structure_model_lib.get_kl(self.g_obs, vb_params_dict, 
                                          self.prior_params_dict, 
                                          self.gh_loc, self.gh_weights, 
                                          e_log_phi = self.e_log_phi, 
                                          detach_ez = False)
    
    def _kl_ps_loss(self, vb_free_params): 
        # this returns the KL **without** the z-entropy, and 
        # prevents gradients from backpropagating through the z's
        # (detach_ez = True). 
        
        # gradients of this ``pseudo-loss" are gradients of the KL 
        # with ez's **fixed**. 
        
        # it turns out that gradients of `_kl_ps_loss` is equivalent 
        # to gradients of `._f` (by optimality of the ez's).
        # (we save some computation by note evaluating the z-entropy)
        
        # however! hessians of this function does not equal 
        # hessians of `._f`. See our custom HVP method below
        
        vb_params_dict = self.vb_params_paragami.fold(vb_free_params, free = True)

        return structure_model_lib.get_kl(self.g_obs, vb_params_dict, 
                                          self.prior_params_dict, 
                                          self.gh_loc, self.gh_weights, 
                                          e_log_phi = self.e_log_phi, 
                                          detach_ez = True)
    
    def _hvp(self, vb_free_params, v): 
        # this is my custom hessian vector product implementation. 
        # note that self._kl_ps_loss detaches the ez, so the naive hvp on 
        # self._kl_ps_loss will the hvp with the ez's **fixed**. 
        
        # this is the HVP with z fixed ....
        kl_theta2_v = jax.jvp(jax.grad(self._kl_ps_loss), 
                              (vb_free_params, ),
                              (v, ))[1]
        
        # this is the corection term taking into account e_zs
        kl_z2_v = self._kl_z2(vb_free_params, v)
        
        return kl_theta2_v - kl_z2_v
    
    def _kl_z2(self, vb_free_params, v): 
        
        # let "theta" be the vb free parameters
        # "moments" are the sufficient statistics of the vb params
        # "zeta" are **unconstrained** cluster belongings
        # "z" are **constrained** cluster belongings
        
        # "f_zz" is the hessian wrt to the z's
        
        # this method returns the second term of the schur complement: 
        # [dmoments/dtheta]^T[dzeta/dmoments]^T[dz/dzeta]^T ... 
        # [f_zz][dz/dzeta][dzeta/dmoments][dmoments/dtheta]
        
        
        # returns [dmoments/dtheta] v
        moments_tuple, moments_jvp = jax.jvp(self._get_moments_from_vb_free_params, \
                                             (vb_free_params, ), (v, ))
        
        # function that returns [dmoments/dtheta]^T v
        moments_vjp = jax.vjp(self._get_moments_from_vb_free_params, 
                              vb_free_params)[1]
        
        def scan_fun(val, x): 
            # x[0] is g_obs[:, l]
            # x[1] is e_log_pop
            # x[2] is e_log_pop jvp

            fun = lambda clust_probs, pop_freq : \
                    self._get_ez_free_from_moments(x[0], clust_probs, pop_freq)
            
            # multiply by [dzeta/dmoments] 
            ez_free, zeta_jvp = jax.jvp(fun, 
                            (moments_tuple[0], x[1]), 
                            (moments_jvp[0], x[2]))
            
            ez = jax.nn.softmax(ez_free, -1)
            
            # multiply by [dz/dzeta]
            dz_jvp = self._constrain_ez_free_jvp(ez, zeta_jvp)
            
            # mutliply by f_zz
            fzz_jvp = dz_jvp / ez
            
            # multiply by [dz/dzeta]^T (same bc it is symmetric)     
            ez_vjp = self._constrain_ez_free_jvp(ez, fzz_jvp)
            
            # multiply by [dzeta/dmoments]^T
            _zeta_vjp = jax.vjp(fun, *(moments_tuple[0], x[1]))[1](ez_vjp)

            return _zeta_vjp[0] + val, _zeta_vjp[1]
        
        zeta_vjp = jax.lax.scan(scan_fun,
                             init = np.zeros(moments_tuple[0].shape), 
                             xs = (self.g_obs.transpose((1, 0, 2)), 
                                   moments_tuple[1], 
                                   moments_jvp[1]))
        
        # finally return [dmoments/dtheta]^T
        return moments_vjp(zeta_vjp)[0]
        
    
    def _get_moments_from_vb_free_params(self, vb_free_params): 
        # returns moments (expected log cluster belongings, 
        # expected log population frequencies) 
        # as a function of vb free parameters
        vb_params_dict = self.vb_params_paragami.fold(vb_free_params, free = True)
        
        pop_freq_beta_params = vb_params_dict['pop_freq_beta_params']
        e_log_pop_freq, e_log_1m_pop_freq = \
            modeling_lib.get_e_log_beta(pop_freq_beta_params)

        # cluster probabilitites
        e_log_sticks, e_log_1m_sticks = \
            ef.get_e_log_logitnormal(
                lognorm_means = vb_params_dict['ind_admix_params']['stick_means'],
                lognorm_infos = vb_params_dict['ind_admix_params']['stick_infos'],
                gh_loc = self.gh_loc,
                gh_weights = self.gh_weights)

        e_log_cluster_probs = \
            modeling_lib.get_e_log_cluster_probabilities_from_e_log_stick(
                                e_log_sticks, e_log_1m_sticks)
        
        return e_log_cluster_probs, \
                np.dstack((e_log_pop_freq, e_log_1m_pop_freq))
    
    @staticmethod
    def _get_ez_free_from_moments(g_obs_l, 
                                  e_log_cluster_probs, 
                                  e_log_pop_freq_l): 
        # returns the ez (in its free parameterization)
        # as a function of the necessary moments
        
        e_log_pop = e_log_pop_freq_l[:, 0]
        e_log_1mpop = e_log_pop_freq_l[:, 1]
        
        return structure_model_lib.\
                           get_loglik_cond_z_l(g_obs_l, 
                                           np.expand_dims(e_log_pop, 0),
                                           np.expand_dims(e_log_1mpop, 0),
                                           e_log_cluster_probs)
    
    
    
    @staticmethod
    def _constrain_ez_free_jvp(ez, v): 
        # jvp of the softmax function 
        # specifically, this is the jacobian of 
        # jax.nn.softmax(ez_free, 1)
        
        term1 = ez * v
        term2 = ez * term1.sum(-1, keepdims = True)

        return term1 - term2
    
    def _jit_functions(self): 
        if self.jit_functions: 
            self.f = jax.jit(self._f)
            self.grad = jax.jit(self._grad)
            self.hvp = jax.jit(self._hvp)
            
        else: 
            self.f = self._f
            self.grad = self._grad
            self.hvp = self._hvp
    
    def _compile_functions(self): 
        x = self.vb_params_paragami.flatten(self.vb_params_paragami.random(), 
                                            free = True)
        
        print('compiling objective ... ')
        t0 = time.time()
        _ = self.f(x).block_until_ready()
        _ = self.grad(x).block_until_ready()
        _ = self.hvp(x, x).block_until_ready()
        print('done. Elasped: {0:3g}'.format(time.time() - t0))
        

class StructurePrecondObjective(StructureObjective):
    """
    This class contains the structure objective. 
    the method self.f returns the KL objective 
    as a function of vb parameters in free space. 
    
    The methods self.grad and self.hvp return the 
    gradient and hessian vector product, respectively. 
    
    The corresponding methods self.f_precond, 
    self.grad_precond, self.hvp_precond are 
    preconditioned functions, using the MFVB covariance 
    as a preconditioner.
    
    The preconditioned functions take two arguments: the 
    parameters at which to evaluate the function 
    (in preconditioned space), 
    and the vb parameters at which the preconditioner is 
    precomputed. (This is so that we do not need to 
    re-compile everything
    when we want to update the preconditioner).
    """
    
    def __init__(self,
                 g_obs, 
                 vb_params_paragami,
                 prior_params_dict, 
                 gh_loc, gh_weights, 
                 compile_hvp = True,
                 e_log_phi = None): 
        
        """
        Parameters
        ----------
        g_obs : ndarray
            The array of one-hot encoded genotypes, of shape (n_obs, n_loci, 3)
        vb_params_paragami : paragami patterned dictionary
            A paragami patterned dictionary that contains the variational parameters.
        prior_params_dict : dictionary
            Dictionary of prior parameters.
            parameters
        gh_loc : vector
            Locations for gauss-hermite quadrature. 
        gh_weights : vector
            Weights for gauss-hermite quadrature. 
        e_log_phi : function
            Function with arguments stick_means and stick_infos 
            and returns the expected log-multiplicative perturbation.
        """

        super().__init__(g_obs, 
                         vb_params_paragami,
                         prior_params_dict, 
                         gh_loc, gh_weights, 
                         e_log_phi = e_log_phi, 
                         jit_functions = False)
        
        self.compile_preconditioned_objectives(compile_hvp)
        
    def _precondition(self, x, precond_params): 

        vb_params_dict = self.vb_params_paragami.fold(precond_params, free = True)
        
        return get_mfvb_cov_matmul(x, vb_params_dict,
                                    self.vb_params_paragami,
                                    return_info = False, 
                                    return_sqrt = True)
    
    def _unprecondition(self, x_c, precond_params): 
        
        vb_params_dict = self.vb_params_paragami.fold(precond_params, free = True)
        
        return get_mfvb_cov_matmul(x_c, vb_params_dict,
                                    self.vb_params_paragami,
                                    return_info = True, 
                                    return_sqrt = True)
        
    def _f_precond(self, x_c, precond_params): 
        return self.f(self._unprecondition(x_c, precond_params))
    
    def _grad_precond(self, x_c, precond_params): 
        x = self._unprecondition(x_c, precond_params)
        return self._unprecondition(self.grad(x), precond_params)
        
    def _hvp_precond(self, x_c, precond_params, v): 
        
        x = self._unprecondition(x_c, precond_params)
        v1 = self._unprecondition(v, precond_params)
        hvp = self.hvp(x, v1)
        hvp = self._unprecondition(hvp, precond_params)

        return hvp
    
    def compile_preconditioned_objectives(self, compile_hvp): 
        self.f_precond = jax.jit(self._f_precond)
        self.precondition = jax.jit(self._precondition)
        self.unprecondition = jax.jit(self._unprecondition)

        self.grad_precond = jax.jit(self._grad_precond)
        
        if compile_hvp: 
            self.hvp_precond = jax.jit(self._hvp_precond)
        else: 
            self.hvp_precond = self._hvp_precond
        
        x = self.vb_params_paragami.flatten(self.vb_params_paragami.random(), 
                                            free = True)
        
        print('compiling preconditioned objective ... ')
        t0 = time.time()
        _ = self.f_precond(x, x).block_until_ready()
        _ = self.precondition(x, x).block_until_ready()
        _ = self.unprecondition(x, x).block_until_ready()
        
        _ = self.grad_precond(x, x).block_until_ready()
        _ = self.hvp_precond(x, x, x).block_until_ready()
        print('done. Elasped: {0:3g}'.format(time.time() - t0))
        
        
def run_preconditioned_lbfgs(g_obs, 
                            vb_params_dict, 
                            vb_params_paragami,
                            prior_params_dict,
                            gh_loc, gh_weights, 
                            e_log_phi = None, 
                            precondition_every = 20, 
                            maxiter = 2000, 
                            x_tol = 1e-2, 
                            f_tol = 1e-2): 
    """
    Parameters
    ----------
    g_obs : ndarray
        The array of one-hot encoded genotypes, of shape (n_obs, n_loci, 3)
    vb_params_dict : dictionary
        A dictionary that contains the initial variational parameters.
    vb_params_paragami : paragami patterned dictionary
        A paragami patterned dictionary that contains the variational parameters.
    prior_params_dict : dictionary
        Dictionary of prior parameters.
        parameters
    gh_loc : vector
        Locations for gauss-hermite quadrature. 
    gh_weights : vector
        Weights for gauss-hermite quadrature. 
    e_log_phi : function
        Function with arguments stick_means and stick_infos 
        and returns the expected log-multiplicative perturbation.
    precondition_every : integer
        We re-compute the preconditioner after `precondition_every` steps. 
    """

    # preconditioned objective 
    precon_objective = StructurePrecondObjective(g_obs, 
                                vb_params_paragami,
                                prior_params_dict,
                                gh_loc = gh_loc, 
                                gh_weights = gh_weights,                       
                                e_log_phi = e_log_phi, 
                                # we don't need hessian-vector-products
                                compile_hvp = False)
    
    t0 = time.time()
    
    vb_params_free = vb_params_paragami.flatten(vb_params_dict, free = True)
    print('init kl: {:.6f}'.format(precon_objective.f(vb_params_free)))
    
    # precondition and run
    iters = 0
    old_kl = 1e16
    while (iters < maxiter): 
        t1 = time.time() 
        
        # transform into preconditioned space
        x0 = vb_params_free
        x0_c = precon_objective.precondition(x0, vb_params_free)
        
        # optimize
        out = optimize.minimize(lambda x : onp.array(precon_objective.f_precond(x, vb_params_free)),
                        x0 = onp.array(x0_c),
                        jac = lambda x : onp.array(precon_objective.grad_precond(x, vb_params_free)),
                        method='L-BFGS-B', 
                        options = {'maxiter': precondition_every})
        
        iters += out.nit
                
        print('iteration [{}]; kl:{:.6f}; elapsed: {:.3f}secs'.format(iters,
                                                                      out.fun,
                                                                      time.time() - t1))

        # transform to original parameterization
        vb_params_free = precon_objective.unprecondition(out.x, vb_params_free)
        
        # check convergence
        if out.success: 
            print('lbfgs converged successfully')
            break

        x_tol_success = np.abs(vb_params_free - x0).max() < x_tol
        if x_tol_success:
            print('x-tolerance reached')
            break
        
        f_tol_success = np.abs(old_kl - out.fun) < f_tol
        if f_tol_success: 
            print('f-tolerance reached')
            break
        else: 
            old_kl = out.fun
            
    vb_opt = vb_params_free
    vb_opt_dict = vb_params_paragami.fold(vb_opt, free = True)
    
    optim_time = time.time() - t0
    print('done. Elapsed {}'.format(round(optim_time, 4)))
    
    return vb_opt_dict, vb_opt, out, precon_objective, optim_time