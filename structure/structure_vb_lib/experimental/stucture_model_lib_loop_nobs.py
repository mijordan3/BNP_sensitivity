# functions for the loglikelihood

def get_e_loglik_gene_nl(g_obs_nl, e_log_pop_freq_l, e_log_1m_pop_freq_l):

    g_obs_nl0 = g_obs_nl[0]
    g_obs_nl1 = g_obs_nl[1]
    g_obs_nl2 = g_obs_nl[2]

    loglik_a = \
        g_obs_nl0 * e_log_1m_pop_freq_l + \
            (g_obs_nl1 + g_obs_nl2) * e_log_pop_freq_l

    loglik_b = \
        (g_obs_nl0 + g_obs_nl1) * e_log_1m_pop_freq_l + \
            g_obs_nl2 * e_log_pop_freq_l

    # returns k_approx x 2 array
    return np.stack((loglik_a, loglik_b), axis = -1)

def get_optimal_ez_nl(g_obs_nl, e_log_pop_freq_l, e_log_1m_pop_freq_l,
                        e_log_cluster_probs_n, detach_ez): 
        
    # get loglikelihood of observations at loci l
    loglik_gene_nl = get_e_loglik_gene_nl(g_obs_nl, e_log_pop_freq_l, e_log_1m_pop_freq_l)

    # add individual belongings
    loglik_cond_z_nl = np.expand_dims(e_log_cluster_probs_n, axis = 1) + loglik_gene_nl

    # individal x chromosome belongings
    if detach_ez: 
        e_z_free = jax.lax.stop_gradient(loglik_cond_z_nl)
    else: 
        e_z_free = loglik_cond_z_nl
        
    e_z_nl = jax.nn.softmax(e_z_free, axis = 0)

    
    return loglik_cond_z_nl, e_z_nl
    
def get_e_loglik_nl(g_obs_nl, e_log_pop_freq_l, e_log_1m_pop_freq_l,
                    e_log_cluster_probs_n, detach_ez):
    
    # returns z-optimized log-likelihood for 
    # individual-n and locus-l
    
    loglik_cond_z_nl, e_z_nl = \
        get_optimal_ez_nl(g_obs_nl, e_log_pop_freq_l, e_log_1m_pop_freq_l,
                            e_log_cluster_probs_n, detach_ez)
    
    # log likelihood
    loglik_nl = np.sum(loglik_cond_z_nl * e_z_nl)

    # entropy term: save this because the z's won't be available later
    # compute the entropy
    z_entropy_nl = (sp.special.entr(e_z_nl)).sum()

    return loglik_nl + z_entropy_nl


def get_e_loglik_n(g_obs_n, e_log_pop_freq, e_log_1m_pop_freq,
                    e_log_cluster_probs_n, detach_ez):
    
    # inner loop over loci
    
    body_fun = lambda val, x : get_e_loglik_nl(x[0], x[1], x[2],
                                             e_log_cluster_probs_n,
                                             detach_ez) + val
    
    scan_fun = lambda val, x : (body_fun(val, x), None)
    
    return jax.lax.scan(scan_fun,
                        init = 0.,
                        xs = (g_obs_n,
                              e_log_pop_freq, 
                              e_log_1m_pop_freq))[0]

    
def get_e_loglik(g_obs, e_log_pop_freq, e_log_1m_pop_freq, \
                    e_log_sticks, e_log_1m_sticks,
                    detach_ez):
    
    # outer loop over individuals 
    
    e_log_cluster_probs = \
        modeling_lib.get_e_log_cluster_probabilities_from_e_log_stick(
                            e_log_sticks, e_log_1m_sticks)
    
    body_fun = lambda val, x : get_e_loglik_n(x[0], 
                                              e_log_pop_freq, 
                                              e_log_1m_pop_freq,
                                              x[1],
                                              detach_ez) + val
    
    scan_fun = lambda val, x : (body_fun(val, x), None)
    
    return jax.lax.scan(scan_fun,
                        init = 0.,
                        xs = (g_obs,
                              e_log_cluster_probs))[0]




# methods in the objective optimizer

    def _kl_zz(self, vb_free_params, v): 
        
        moments_tuple = \
            self._get_moments_from_vb_free_params(vb_free_params)
        
        moments_jvp = jax.jvp(self._get_moments_from_vb_free_params, \
                                      (vb_free_params, ), (v, ))[1]
        
        moments_vjp = jax.vjp(self._get_moments_from_vb_free_params, 
                             vb_free_params)[1]
        
        def inner_loop_over_l(g_obs_n, 
                               e_log_cluster_probs_n, 
                               e_log_cluster_probs_jvp_n):
            
            def scan_fun_inner_loop(val, x_n): 
                # x[0] is g_obs_n[l]
                # x[1] is e_log_pop[l]
                # x[2] is e_log_pop[l] jvp

                fun = lambda clust_probs_n, pop_freq_l: \
                        self._ps_loss_zl(x_n[0], clust_probs_n, pop_freq_l)

                jvp1 = jax.jvp(fun, 
                                (e_log_cluster_probs_n, x_n[1]), 
                                (e_log_cluster_probs_jvp_n, x_n[2]))[1]

                vjp1 = jax.vjp(fun, *(e_log_cluster_probs_n, x_n[1]))[1](jvp1)
                
                # sum over cluster probs, stack pop_freq
                return vjp1[0] + val, vjp1[1]
            
            vjp_cluster_probs_n, vjp_pop_freq = \
                jax.lax.scan(scan_fun_inner_loop,
                             init = np.zeros(self.k_approx), 
                             xs = (g_obs_n, 
                                     moments_tuple[1], 
                                     moments_jvp[1]))
            
            return vjp_pop_freq, vjp_cluster_probs_n
        
        def scan_fun_outer_loop(val, x): 
            # x[0] is g_obs 
            # x[1] is e_log_cluster_probs 
            # x[2] is e_log_cluster_probs jvp
            
            vjp_pop_freq, vjp_cluster_probs_n = \
                inner_loop_over_l(x[0], x[1], x[2])
            
            # sum population frequencies, stack cluster probs
            return vjp_pop_freq + val, vjp_cluster_probs_n
        
        vjp = jax.lax.scan(scan_fun_outer_loop,
                           init = np.zeros(moments_tuple[1].shape), 
                           xs = (self.g_obs, 
                                 moments_tuple[0], 
                                 moments_jvp[0]))
            
        return moments_vjp((vjp[1], vjp[0]))[0]
        
    
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
        
        return e_log_cluster_probs, \
                np.dstack((e_log_pop_freq, e_log_1m_pop_freq))
    
    @staticmethod
    def _ps_loss_zl(g_obs_nl, 
                    e_log_cluster_probs_n, 
                    e_log_pop_freq_l): 
                
        return 2 * np.sqrt(structure_model_lib.\
                           get_optimal_ez_nl(g_obs_nl, 
                                               e_log_pop_freq_l[:, 0],
                                               e_log_pop_freq_l[:, 1],
                                               e_log_cluster_probs_n,
                                               detach_ez = False)[1]).flatten()
    
