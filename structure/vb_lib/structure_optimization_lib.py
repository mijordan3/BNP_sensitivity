import jax
import jax.numpy as np
import jax.scipy as sp

from jax.experimental import loops

from scipy import optimize 

import numpy as onp
from sklearn.decomposition import NMF

import paragami

from vb_lib import structure_model_lib
from vb_lib.cavi_lib import run_cavi
from vb_lib.preconditioner_lib import get_mfvb_cov_matmul

import bnpmodeling_runjingdev.exponential_families as ef
from bnpmodeling_runjingdev import cluster_quantities_lib, modeling_lib
from bnpmodeling_runjingdev.sensitivity_lib import get_jac_hvp_fun

import time 

###############
# functions for initializing
###############
def cluster_and_get_init(g_obs, k, seed):
    # g_obs should be n_obs x n_loci x 3,
    # a one-hot encoding of genotypes
    assert len(g_obs.shape) == 3

    # convert one-hot encoding to probability of A genotype, {0, 0.5, 1}
    x = g_obs.argmax(axis = 2) / 2

    # run NMF
    model = NMF(n_components=k, init='random', random_state = seed)
    init_ind_admix_propn_unscaled = model.fit_transform(onp.array(x))
    init_pop_allele_freq_unscaled = model.components_.T

    # divide by largest allele frequency, so all numbers between 0 and 1
    denom_pop_allele_freq = np.max(init_pop_allele_freq_unscaled)
    init_pop_allele_freq = init_pop_allele_freq_unscaled / \
                                denom_pop_allele_freq

    # normalize rows
    denom_ind_admix_propn = \
        init_ind_admix_propn_unscaled.sum(axis = 1, keepdims = True)
    init_ind_admix_propn = \
        init_ind_admix_propn_unscaled / denom_ind_admix_propn
    # clip again and renormalize
    init_ind_admix_propn = init_ind_admix_propn.clip(0.05, 0.95)
    init_ind_admix_propn = init_ind_admix_propn / \
                            init_ind_admix_propn.sum(axis = 1, keepdims = True)

    return np.array(init_ind_admix_propn), \
            np.array(init_pop_allele_freq.clip(0.05, 0.95))

def set_nmf_init_vb_params(g_obs, k_approx, vb_params_dict, seed):
    # set the vb parameters at the NMF results 
    
    # get initial admixtures, and population frequencies
    init_ind_admix_propn, init_pop_allele_freq = \
            cluster_and_get_init(g_obs, k_approx, seed = seed)

    # set bnp parameters for individual admixture
    # set mean to be logit(stick_breaking_propn), info to be 1
    stick_break_propn = \
        cluster_quantities_lib.get_stick_break_propns_from_mixture_weights(init_ind_admix_propn)

    use_logitnormal_sticks = 'stick_means' in vb_params_dict['ind_admix_params'].keys()
    if use_logitnormal_sticks:
        ind_mix_stick_propn_mean = np.log(stick_break_propn) - np.log(1 - stick_break_propn)
        ind_mix_stick_propn_info = np.ones(stick_break_propn.shape)
        vb_params_dict['ind_admix_params']['stick_means'] = ind_mix_stick_propn_mean
        vb_params_dict['ind_admix_params']['stick_infos'] = ind_mix_stick_propn_info
    else:
        ind_mix_stick_beta_param1 = np.ones(stick_break_propn.shape) 
        ind_mix_stick_beta_param2 = (1 - stick_break_propn) / stick_break_propn
        vb_params_dict['ind_admix_params']['stick_beta'] = \
            np.concatenate((ind_mix_stick_beta_param1[:, :, None],
                            ind_mix_stick_beta_param2[:, :, None]), axis = 2)

    # set beta paramters for population paramters
    # set beta = 1, alpha to have the correct mean
    beta0 = 1
    pop_freq_beta_params1 = beta0 * init_pop_allele_freq / (1 - init_pop_allele_freq)
    pop_freq_beta_params2 = np.ones(init_pop_allele_freq.shape) * beta0
    pop_freq_beta_params = np.concatenate((pop_freq_beta_params1[:, :, None],
                                       pop_freq_beta_params2[:, :, None]), axis = 2)

    vb_params_dict['pop_freq_beta_params'] = pop_freq_beta_params

    return vb_params_dict

    
#########################
# Function to convert beta sticks to 
# logitnormal sticks
#########################
def convert_beta_sticks_to_logitnormal(stick_betas, 
                                       logitnorm_stick_params_dict,
                                       logitnorm_stick_params_paragami, 
                                       gh_loc, gh_weights): 
    
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
                         max_iter = 100, 
                         seed = 0): 
    
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
    def __init__(self,
                 g_obs, 
                 vb_params_paragami,
                 prior_params_dict, 
                 gh_loc, gh_weights, 
                 e_log_phi = None, 
                 jit_functions = True): 
        
        self.g_obs = g_obs
        self.vb_params_paragami = vb_params_paragami 
        self.prior_params_dict = prior_params_dict 
            
        self.dim_vb_free = len(vb_params_paragami.flatten(\
                                vb_params_paragami.random(), \
                                free = True))
        
        self.gh_loc = gh_loc
        self.gh_weights = gh_weights 
        self.e_log_phi = e_log_phi 
        
        self.grad_unjitted = jax.grad(self.f_unjitted)

        self.jit_functions = jit_functions
        self._jit_functions()
        if self.jit_functions: 
            self._compile_functions()
        

    def f_unjitted(self, vb_free_params):
        # note that we detach the ez! 
        # the gradient wrt to self.f_unjitted does not change, and will be faster. 
        # the hvp with respect to self.f_unjitted will **not** be correct! 
        # (this will be the hessian of parameters with the e_z fixed). 
        
        # See the custom HVP implementation below. 
        # these adjustments were made so that HVPs for the HGDP data 
        # do not crash ... 
        
        vb_params_dict = self.vb_params_paragami.fold(vb_free_params, free = True)

        return structure_model_lib.get_kl(self.g_obs, vb_params_dict, 
                                          self.prior_params_dict, 
                                          self.gh_loc, self.gh_weights, 
                                          e_log_phi = self.e_log_phi, 
                                          detach_ez = True)
    
    def hvp_unjitted(self, vb_free_params, v): 
        # this is my custom hessian vector product implementation. 
        # note that self.f_unjitted detaches the ez, so the naive hvp on 
        # self.f_unjitted will the hvp with the ez's **fixed**. 
        
        # this is the HVP with z fixed ....
        kl_theta2_v = jax.jvp(jax.grad(self.f_unjitted), (vb_free_params, ), (v, ))[1]
        
        # this is the corection term for the e_zs
        kl_zz_v = self._kl_zz(vb_free_params, v)
        
        return kl_theta2_v - kl_zz_v
    
    def _kl_zz(self, vb_free_params, v): 
        
        moments_tuple = \
            self._get_moments_from_vb_free_params(vb_free_params)
        
        moments_jvp = jax.jvp(self._get_moments_from_vb_free_params, \
                                      (vb_free_params, ), (v, ))[1]
        
        moments_vjp = jax.vjp(self._get_moments_from_vb_free_params, 
                             vb_free_params)[1]
        
        with loops.Scope() as s:
            s.hvp = np.zeros(self.dim_vb_free)
            for l in s.range(self.g_obs.shape[1]):

                fun = lambda x, y, z: self._ps_loss_zl(x, y, z, l)
                
                jvp1 = jax.jvp(fun, moments_tuple, moments_jvp)[1]
                vjp1 = jax.vjp(fun, *moments_tuple)[1](jvp1)
                
                s.hvp += moments_vjp(vjp1)[0]
                
        return s.hvp
        
    
    def _get_moments_from_vb_free_params(self, vb_free_params): 
        
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
        
        return e_log_pop_freq, e_log_1m_pop_freq, e_log_cluster_probs
    
    def _ps_loss_zl(self, 
                    e_log_pop_freq, e_log_1m_pop_freq, 
                    e_log_cluster_probs, 
                    l_indx): 
        
        return 2 * np.sqrt(structure_model_lib.\
                           get_optimal_ezl(self.g_obs[:, l_indx], 
                                           np.expand_dims(e_log_pop_freq[l_indx], 0),
                                           np.expand_dims(e_log_1m_pop_freq[l_indx], 0),
                                           e_log_cluster_probs,
                                           detach_ez = False)[1]).flatten()
    
    
    
    def _jit_functions(self): 
        if self.jit_functions: 
            self.f = jax.jit(self.f_unjitted)
            self.grad = jax.jit(self.grad_unjitted)
            self.hvp = jax.jit(self.hvp_unjitted)
            
        else: 
            self.f = self.f_unjitted
            self.grad = self.grad_unjitted
            self.hvp = self.hvp_unjitted
    
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
    def __init__(self,
                    g_obs, 
                    vb_params_paragami,
                    prior_params_dict, 
                    gh_loc, gh_weights, 
                    e_log_phi = None): 
        
        super().__init__(g_obs, 
                         vb_params_paragami,
                         prior_params_dict, 
                         gh_loc, gh_weights, 
                         e_log_phi = None, 
                         jit_functions = False)
        
        self.compile_preconditioned_objectives()
        
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
        return self.f_unjitted(self._unprecondition(x_c, precond_params))
    
    def _grad_precond(self, x_c, precond_params): 
        x = self._unprecondition(x_c, precond_params)
        return self._unprecondition(self.grad(x), precond_params)
        
    def _hvp_precond(self, x_c, precond_params, v): 
        # again, can't just use hvp of self.f ... 
        # it is not correct
        
        x = self._unprecondition(x_c, precond_params)
        v1 = self._unprecondition(v, precond_params)
        hvp = self.hvp_unjitted(x, v1)
        hvp = self._unprecondition(hvp, precond_params)

        return hvp
    
    def compile_preconditioned_objectives(self): 
        self.f_precond = jax.jit(self._f_precond)
        self.precondition = jax.jit(self._precondition)
        self.unprecondition = jax.jit(self._unprecondition)
        
        # self.grad_precond = jax.jit(jax.grad(self._f_precond, argnums = 0))
        self.grad_precond = jax.jit(self._grad_precond)
        self.hvp_precond = jax.jit(self._hvp_precond)
        
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
    
    # preconditioned objective 
    precon_objective = StructurePrecondObjective(g_obs, 
                                vb_params_paragami,
                                prior_params_dict,
                                gh_loc = gh_loc, gh_weights = gh_weights,                       
                                e_log_phi = e_log_phi)
    
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