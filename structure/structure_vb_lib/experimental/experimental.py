def set_init_vb_params(g_obs, k_approx, vb_params_dict, 
                       prior_params_dict,
                       gh_loc = None, gh_weights = None,
                       seed = 1,
                       n_iter = 5): 
    
    # get nmf init
    t0 = time.time()
    print('running NMF ...')
    vb_params_dict = \
        set_nmf_init_vb_params(g_obs, k_approx, vb_params_dict, seed)
    
    # update population frequency betas 
    # and implicity the ezs. 
    # these updates are closed form, and relatively fast
    
    print('running a few cavi steps for pop beta ...')
    # get initial moments from vb_params
    e_log_sticks, e_log_1m_sticks, \
        e_log_pop_freq, e_log_1m_pop_freq = \
            structure_model_lib.get_moments_from_vb_params_dict(
                vb_params_dict, gh_loc, gh_weights)

    e_log_cluster_probs = \
        modeling_lib.get_e_log_cluster_probabilities_from_e_log_stick(
            e_log_sticks, e_log_1m_sticks)
    
    # a few cavi updates
    for i in range(n_iter): 
        vb_params_dict['pop_freq_beta_params'],\
            e_log_pop_freq, e_log_1m_pop_freq = \
                update_pop_beta(g_obs,
                                e_log_pop_freq, e_log_1m_pop_freq,
                                e_log_cluster_probs,
                                prior_params_dict)
    print('done. Elapsed: {0:3g}'.format(time.time() - t0))
    
    return vb_params_dict


#################
# Functions to specificially optmize 
# the individual admixture stick parameters
#################
def get_ind_admix_params_psloss(g_obs, ind_admix_params, 
                                e_log_pop_freq, e_log_1m_pop_freq, 
                                prior_params_dict, 
                                gh_loc, gh_weights,
                                detach_ez = True): 

    # returns the terms of the KL that depend on the 
    # individual admixture parameters
    # Hence the KL is not correct, but its derivatives 
    # wrt to ind_admix_params are correct
    
    # TODO handle log-phi's
    
    # data parameters
    n_obs = g_obs.shape[0]
    k_approx = e_log_pop_freq.shape[1]
    
    # get expecations
    e_log_sticks, e_log_1m_sticks = \
            ef.get_e_log_logitnormal(
                lognorm_means = ind_admix_params['stick_means'],
                lognorm_infos = ind_admix_params['stick_infos'],
                gh_loc = gh_loc,
                gh_weights = gh_weights)
    
    e_log_cluster_probs = \
        modeling_lib.get_e_log_cluster_probabilities_from_e_log_stick(
            e_log_sticks, e_log_1m_sticks)
    
    # sum the e_z's over loci
    body_fun = lambda val, x :\
                    structure_model_lib.get_optimal_ezl(x[0], x[1], x[2],
                                        e_log_cluster_probs)[1].sum(-1) + val
    
    scan_fun = lambda val, x : (body_fun(val, x), None)
    
    init_val = np.zeros((n_obs, k_approx))
    ez_nxk = jax.lax.scan(scan_fun, init_val,
                        xs = (g_obs.transpose((1, 0, 2)),
                                e_log_pop_freq, e_log_1m_pop_freq))[0]
    if detach_ez:
        ez_nxk = jax.lax.stop_gradient(ez_nxk)

    # log-likelihood term 
    loglik_ind = (ez_nxk * e_log_cluster_probs).sum()
    
    # entropy term
    stick_entropy = \
            modeling_lib.get_stick_breaking_entropy(
                                    ind_admix_params['stick_means'],
                                    ind_admix_params['stick_infos'],
                                    gh_loc, gh_weights)
    
    # prior term
    ind_mix_dp_prior =  (prior_params_dict['dp_prior_alpha'] - 1) * np.sum(e_log_1m_sticks)
    
    return - (loglik_ind + ind_mix_dp_prior + stick_entropy).squeeze()

def get_ind_admix_params_loss(g_obs, 
                            ind_admix_params, 
                            pop_freq_beta_params, 
                            prior_params_dict, 
                            gh_loc, gh_weights, 
                            detach_ez = True):
    
    # used for testing the pseudo-loss above 
        
    vb_params_dict = dict({'pop_freq_beta_params':pop_freq_beta_params,
                           'ind_admix_params': ind_admix_params})

    return structure_model_lib.get_kl(g_obs,
                                        vb_params_dict,
                                        prior_params_dict,
                                        gh_loc, gh_weights,
                                        detach_ez = detach_ez)

class StickObjective():
    def __init__(self, g_obs, vb_params_paragami, prior_params_dict, 
                          gh_loc, gh_weights, 
                          compute_hess = False): 

        self.vb_params_paragami = vb_params_paragami
        
        self.prior_params_dict = prior_params_dict
        self.gh_loc = gh_loc
        self.gh_weights = gh_weights
        
        # objective and gradients
        self.stick_objective_fun = \
            paragami.FlattenFunctionInput(
                original_fun = get_ind_admix_params_psloss,
                patterns = vb_params_paragami['ind_admix_params'],
                free = True,
                argnums = 1)
        
        self._grad_tmp = jax.grad(self.stick_objective_fun, argnums = 1)  
        self._hess_tmp = jax.hessian(self.stick_objective_fun, argnums = 1)
        
        self.compute_hess = compute_hess 
        
        self._compile_functions(g_obs)
    
    def _flatten_ind_admix_params(self, ind_admix_params): 
        return self.vb_params_paragami['ind_admix_params'].flatten(ind_admix_params, 
                                                                   free = True)
        
    def _f(self, 
          g_obs, 
          ind_admix_params_free, 
          e_log_pop_freq, 
          e_log_1m_pop_freq): 
        
        return self.stick_objective_fun(g_obs, 
                                        ind_admix_params_free, 
                                        e_log_pop_freq, e_log_1m_pop_freq, 
                                        self.prior_params_dict, 
                                        self.gh_loc, self.gh_weights)
    def _grad(self, 
             g_obs, 
             ind_admix_params_free, 
              e_log_pop_freq, 
              e_log_1m_pop_freq): 
        
        return self._grad_tmp(g_obs, 
                            ind_admix_params_free, 
                            e_log_pop_freq, e_log_1m_pop_freq, 
                            self.prior_params_dict, 
                            self.gh_loc, self.gh_weights)

    def _hvp(self, 
            g_obs, 
            ind_admix_params_free, 
            e_log_pop_freq, 
            e_log_1m_pop_freq, 
            v): 

        loss = lambda x : self.stick_objective_fun(g_obs, 
                                    x, 
                                    e_log_pop_freq, e_log_1m_pop_freq, 
                                    self.prior_params_dict, 
                                    self.gh_loc, self.gh_weights)
    
        return jax.jvp(jax.grad(loss), (ind_admix_params_free, ), (v, ))[1]
    
    def _hess(self, 
                g_obs, 
                ind_admix_params_free, 
                e_log_pop_freq, 
                e_log_1m_pop_freq): 
        
        return self._hess_tmp(g_obs, 
                            ind_admix_params_free, 
                            e_log_pop_freq, e_log_1m_pop_freq, 
                            self.prior_params_dict, 
                            self.gh_loc, self.gh_weights)
    
    def _compile_functions(self, g_obs): 
        
        self.f = jax.jit(self._f)
        self.grad = jax.jit(self._grad)
        self.hvp = jax.jit(self._hvp)
        self.flatten_ind_admix_params = jax.jit(self._flatten_ind_admix_params)
        
        if self.compute_hess: 
            self.hess = jax.jit(self._hess)
        else: 
            self.hess = None
        
        # compile 
        print('compiling stick objective and gradients ...')
        t0 = time.time()
        
        # draw random parameters        
        param_dict = self.vb_params_paragami.random()
        stick_free_params = self.flatten_ind_admix_params(param_dict['ind_admix_params'])
        
        e_log_pop_freq, e_log_1m_pop_freq = \
            modeling_lib.get_e_log_beta(param_dict['pop_freq_beta_params'])
        
        # compile functions
        _ = self.f(g_obs, stick_free_params, 
                   e_log_pop_freq, e_log_1m_pop_freq).block_until_ready()
        _ = self.grad(g_obs, stick_free_params, 
                      e_log_pop_freq, e_log_1m_pop_freq).block_until_ready()
        _ = self.hvp(g_obs, stick_free_params, 
                     e_log_pop_freq, e_log_1m_pop_freq, 
                     stick_free_params).block_until_ready()
        
        if self.compute_hess: 
            _ = self.hess(g_obs, stick_free_params, 
                      e_log_pop_freq, e_log_1m_pop_freq).block_until_ready()
                
        print('sticks compile time: {0:.3g}sec'.format(time.time() - t0))
    
    def optimize_sticks(self, 
                        g_obs, 
                        ind_admix_params, 
                        e_log_pop_freq, 
                        e_log_1m_pop_freq, 
                        maxiter = 1): 
        
        if self.hess is None: 
            raise NotImplementedError()
            
        x0 = self.flatten_ind_admix_params(ind_admix_params)
        fun = lambda x : onp.array(self.f(g_obs, x, e_log_pop_freq, e_log_1m_pop_freq))
        jac = lambda x : onp.array(self.grad(g_obs, x, e_log_pop_freq, e_log_1m_pop_freq))
        hess = lambda x : onp.array(self.hess(g_obs, x, e_log_pop_freq, e_log_1m_pop_freq))
                                   
        out = optimize.minimize(fun = fun, 
                              jac = jac, 
                              hess = hess, 
                              x0 = x0, 
                              method = 'trust-exact', 
                              options = {'maxiter': maxiter})
                                   
        ind_admix_params_opt = self.vb_params_paragami['ind_admix_params'].fold(out.x, free = True)
                                   
        return ind_admix_params_opt, out

########################
# A helper function to get
# objective functions and gradients
########################
def define_structure_objective(g_obs, vb_params_dict,
                                vb_params_paragami,
                                prior_params_dict,
                                gh_loc = None, gh_weights = None,
                                e_log_phi = None, 
                                compile_hvp = False):

    # set up loss
    _kl_fun_free = paragami.FlattenFunctionInput(
                                original_fun=structure_model_lib.get_kl,
                                patterns = vb_params_paragami,
                                free = True,
                                argnums = 1)

    kl_fun_free = lambda x : _kl_fun_free(g_obs, x, prior_params_dict,
                                                     gh_loc, gh_weights,
                                                     e_log_phi = e_log_phi)

    # initial free parameters
    init_vb_free = vb_params_paragami.flatten(vb_params_dict, free = True)
    
    # define objective
    optim_objective = OptimizationObjectiveJaxtoNumpy(kl_fun_free, 
                                                     init_vb_free, 
                                                      compile_hvp = compile_hvp, 
                                                      print_every = 1,
                                                      log_every = 0)
    
    return optim_objective, init_vb_free

