import jax

import jax.numpy as np

from bnpmodeling_runjingdev import modeling_lib

from structure_vb_lib import structure_model_lib, cavi_lib, testutils

import unittest


class TestCaviUpdate(unittest.TestCase):
    
    @staticmethod
    def get_moments(vb_params_dict): 
        
        # get moments 
        e_log_sticks, e_log_1m_sticks, \
            e_log_pop_freq, e_log_1m_pop_freq = \
                structure_model_lib.get_moments_from_vb_params_dict( \
                                        vb_params_dict)
        
        e_log_cluster_probs = \
            modeling_lib.get_e_log_cluster_probabilities_from_e_log_stick(
                e_log_sticks, e_log_1m_sticks)
        
        return e_log_sticks, e_log_1m_sticks, \
                e_log_pop_freq, e_log_1m_pop_freq, \
                    e_log_cluster_probs

    def test_admixture_stick_update(self):
        
        # define model
        g_obs, vb_params_dict, _, prior_params_dict, gh_loc, gh_weights = \
            testutils.draw_data_and_construct_model(use_logitnormal_sticks = False)
        
        e_log_sticks, e_log_1m_sticks, \
            e_log_pop_freq, e_log_1m_pop_freq, \
                e_log_cluster_probs = self.get_moments(vb_params_dict)
        
        dp_prior_alpha = prior_params_dict['dp_prior_alpha']
        allele_prior_alpha = prior_params_dict['allele_prior_alpha']
        allele_prior_beta = prior_params_dict['allele_prior_beta']

        # closed-form update
        ind_admix_update = cavi_lib.update_ind_admix_beta(g_obs,
                                                          e_log_pop_freq,
                                                          e_log_1m_pop_freq, 
                                                          e_log_cluster_probs, 
                                                          prior_params_dict)[0]
        
        # autodiff updates
        ind_admix_update1_ag = cavi_lib.get_stick_update1_ag(g_obs,
                        e_log_pop_freq, e_log_1m_pop_freq,
                        e_log_sticks, e_log_1m_sticks,
                        dp_prior_alpha, allele_prior_alpha,
                        allele_prior_beta) + 1

        ind_admix_update2_ag = cavi_lib.get_stick_update2_ag(g_obs,
                                e_log_pop_freq, e_log_1m_pop_freq,
                                e_log_sticks, e_log_1m_sticks,
                                dp_prior_alpha, allele_prior_alpha,
                                allele_prior_beta) + 1
        
        # should be close
        assert np.abs(ind_admix_update1_ag -  \
                      ind_admix_update[:, :, 0]).max() < 1e-8
        assert np.abs(ind_admix_update2_ag -  \
                      ind_admix_update[:, :, 1]).max() < 1e-8

        
    def test_population_stick_update(self):
        
        # define model
        g_obs, vb_params_dict, _, prior_params_dict, _, _ = \
            testutils.draw_data_and_construct_model(use_logitnormal_sticks = False)
        
        e_log_sticks, e_log_1m_sticks, \
            e_log_pop_freq, e_log_1m_pop_freq, \
                e_log_cluster_probs = self.get_moments(vb_params_dict)
        
        dp_prior_alpha = prior_params_dict['dp_prior_alpha']
        allele_prior_alpha = prior_params_dict['allele_prior_alpha']
        allele_prior_beta = prior_params_dict['allele_prior_beta']
        
        # closed-form update
        pop_beta_update = cavi_lib.update_pop_beta(g_obs, 
                                                   e_log_pop_freq, 
                                                   e_log_1m_pop_freq, 
                                                   e_log_cluster_probs, 
                                                   prior_params_dict)[0]
        
        # autodiff updates
        pop_beta_update1_ag = cavi_lib.get_pop_beta_update1_ag(g_obs,
                        e_log_pop_freq, e_log_1m_pop_freq,
                        e_log_sticks, e_log_1m_sticks,
                        dp_prior_alpha, allele_prior_alpha,
                        allele_prior_beta) + 1

        pop_beta_update2_ag = cavi_lib.get_pop_beta_update2_ag(g_obs,
                                e_log_pop_freq, e_log_1m_pop_freq,
                                e_log_sticks, e_log_1m_sticks,
                                dp_prior_alpha, allele_prior_alpha,
                                allele_prior_beta) + 1
        
        # should be close
        assert np.abs(pop_beta_update1_ag -  
                      pop_beta_update[:, :, 0]).max() < 1e-8
        assert np.abs(pop_beta_update2_ag -  
                      pop_beta_update[:, :, 1]).max() < 1e-8

    def test_cavi(self):
        # run cavi in full, with the debugger on
        
        # define model
        g_obs, vb_params_dict, vb_params_paragami, prior_params_dict, _, _ = \
            testutils.draw_data_and_construct_model(use_logitnormal_sticks = False)
        
        # run cavi
        _ = cavi_lib.run_cavi(g_obs,
                              vb_params_dict,
                              vb_params_paragami,
                              prior_params_dict, 
                              debug = True)

if __name__ == '__main__':
    unittest.main()
