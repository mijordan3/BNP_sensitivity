import jax

import jax.numpy as np
import jax.scipy as sp

import scipy as osp

from vb_lib import structure_model_lib, preconditioner_lib, testutils

import unittest 

###############
# functions to compute preconditioner using autograd 
###############
def get_natural_params(vb_params_free, vb_params_paragami): 
    vb_params_dict = vb_params_paragami.fold(vb_params_free, free = True)
    
    pop_freq_beta_params = vb_params_dict['pop_freq_beta_params']
    
    means = vb_params_dict['ind_admix_params']['stick_means']
    infos = vb_params_dict['ind_admix_params']['stick_infos']
    ind_admix_stick_natparam1 = means * infos 
    ind_admix_stick_natparam2 = -0.5 * infos
    
    return dict({'pop_freq_beta_params': pop_freq_beta_params, 
                 'ind_admix_stick_natparam1': ind_admix_stick_natparam1,
                 'ind_admix_stick_natparam2': ind_admix_stick_natparam2})

def get_log_partition(nat_params_dict): 
    
    pop_freq_beta_params = nat_params_dict['pop_freq_beta_params']
    ind_admix_stick_natparam1 = nat_params_dict['ind_admix_stick_natparam1']
    ind_admix_stick_natparam2 = nat_params_dict['ind_admix_stick_natparam2']
    
    pop_freq_terms = (sp.special.gammaln(pop_freq_beta_params[:, :, 0]) + 
                         sp.special.gammaln(pop_freq_beta_params[:, :, 1]) -
                         sp.special.gammaln(pop_freq_beta_params.sum(-1))).sum()
    
    ind_admix_terms = (- ind_admix_stick_natparam1**2 / \
                       (4 * ind_admix_stick_natparam2) - \
                        0.5 * np.log(-2 * ind_admix_stick_natparam2)).sum()
    
    return pop_freq_terms + ind_admix_terms

def autodiff_preconditioner_v(v, vb_params_free, vb_params_paragami): 
    
    _get_natural_params = lambda x : get_natural_params(x, vb_params_paragami)
    
    jvp = jax.jvp(_get_natural_params, (vb_params_free, ), (v, ))
    hvp = jax.jvp(jax.grad(get_log_partition), (jvp[0], ), (jvp[1], ))[1]
    vjp = jax.vjp(_get_natural_params, vb_params_free)[1](hvp)[0]
    
    return vjp

def get_autodiff_preconditioner(vb_params_free, vb_params_paragami): 
    return jax.lax.map(lambda v : autodiff_preconditioner_v(v, vb_params_free, vb_params_paragami), 
                        np.eye(len(vb_params_free)))


class TestStructureObjective(unittest.TestCase):
    def test_preconditioner_against_autodiff(self): 
        
        # get model 
        g_obs, vb_params_dict, vb_params_paragami, prior_params_dict, gh_loc, gh_weights = \
            testutils.draw_data_and_construct_model()

        vb_params_free = vb_params_paragami.flatten(vb_params_dict, 
                                                   free = True)
        
        # get autograd preconditioner (the covariance)
        mfvb_cov_ag = get_autodiff_preconditioner(vb_params_free, vb_params_paragami)
        
        # check my covariance 
        identity_matr = np.eye(len(vb_params_free))
        mfvb_cov = jax.lax.map(lambda v : preconditioner_lib.get_mfvb_cov_matmul(v, 
                                    vb_params_dict,
                                    vb_params_paragami,
                                    return_info = False), 
                       identity_matr)

        assert np.abs(mfvb_cov - mfvb_cov_ag).max() < 1e-12
        
        # check covariance square roots 
        sqrt_cov_ag = osp.linalg.sqrtm(mfvb_cov_ag)
        sqrt_cov = jax.lax.map(lambda v : preconditioner_lib.get_mfvb_cov_matmul(v, 
                                            vb_params_dict,
                                            vb_params_paragami,
                                            return_info = False, 
                                            return_sqrt = True), 
                               identity_matr)
        assert np.abs(sqrt_cov - sqrt_cov_ag).max() < 1e-12
        
        
        # check infos
        mfvb_info_ag = np.linalg.inv(mfvb_cov_ag)
        mfvb_info = jax.lax.map(lambda v : preconditioner_lib.get_mfvb_cov_matmul(v, 
                                            vb_params_dict,
                                            vb_params_paragami,
                                            return_info = True), 
                               identity_matr)

        assert np.abs(mfvb_info - mfvb_info_ag).max() < 1e-12
        
        # check info sqrt
        sqrt_info_ag = osp.linalg.sqrtm(mfvb_info_ag)
        sqrt_info = jax.lax.map(lambda v : preconditioner_lib.get_mfvb_cov_matmul(v, 
                                            vb_params_dict,
                                            vb_params_paragami,
                                            return_info = True, 
                                            return_sqrt = True), 
                               identity_matr)
        assert np.abs(sqrt_info - sqrt_info_ag).max() < 1e-12