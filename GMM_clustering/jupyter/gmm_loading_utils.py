# a wrapper to load results from GMM fits
# useful for my functional sensitivity notebooks ... 

import matplotlib.pyplot as plt

import jax.numpy as np

import paragami

from bnpmodeling_runjingdev import result_loading_utils

class GMMResultsLoader(object):
    
    # The method get_free_param_results_from_perturbation
    # loads all fits and lr predictions 
    # for a given perturbation (and delta). 
    
    def __init__(self,
                 alpha0 = 4.0,
                 out_folder = '../fits/',
                 out_filename = 'iris_fit'): 
        
        # file paths
        self.out_filename = out_filename 
        self.out_folder = out_folder
        
        # inital alpha 
        self.alpha0 = alpha0
                        
        self._set_init_fit()
        self._set_lr_data()
        
    def _set_init_fit(self): 
        
        # initial fit file
        init_fit_file = self.out_folder + self.out_filename + \
                                '_alpha' + str(self.alpha0) + '.npz'

        print('loading initial fit from: ', init_fit_file)

        vb_init_dict, self.vb_params_paragami, self.init_fit_meta_data = \
            paragami.load_folded(init_fit_file)
        
        self.vb_init_free = self.vb_params_paragami.flatten(vb_init_dict,
                                                       free = True)    
    
    def _set_lr_data(self): 
        
        lr_file = self.out_folder + self.out_filename + \
                    '_alpha' + str(self.alpha0) + '_lrderivatives.npz'
        
        print('loading lr derivatives from: ', lr_file)
        
        lr_data = np.load(lr_file)
        assert lr_data['alpha0'] == self.alpha0
        assert np.abs(lr_data['vb_opt'] - self.vb_init_free).max() < 1e-12
        assert np.abs(lr_data['kl'] - self.init_fit_meta_data['final_kl']) < 1e-8
        self.lr_data = lr_data
        
    def _load_refit_files(self, perturbation, delta): 

        # get all files for that particular perturbation
        match_crit = self.out_filename + '_' + perturbation + \
                        '_delta{}_eps'.format(delta) + '\d+.npz'
        
        # load refit results
        vb_refit_list, epsilon_vec, meta_data_list = \
            result_loading_utils.load_refit_files_epsilon(self.out_folder,
                                                          match_crit)
        
        # check some model parameters 
        assert np.all(result_loading_utils.\
                  _load_meta_data_from_list(meta_data_list, 
                                            'alpha') == \
                  self.alpha0)
        assert np.all(result_loading_utils.\
                          _load_meta_data_from_list(meta_data_list, 
                                                    'delta') == \
                          delta)
        
        # append epsilon = 0.
        vb_refit_list = np.vstack((self.vb_init_free, vb_refit_list))
        epsilon_vec = np.hstack((0., epsilon_vec))
        
        # print optimization time
        optim_time = meta_data_list[-1]['optim_time']
        
        
        print('Optim time at epsilon = 1: {:.3f}secs'.format(optim_time))

        return vb_refit_list, epsilon_vec
    
    def _get_lr_free_params(self, perturbation, epsilon_vec, delta): 
        # Function to load linear response derivatives and their predicted free parameters
        
        der_time = self.lr_data['lr_time_' + perturbation]
        dinput_hyper = self.lr_data['dinput_dfun_' + perturbation]
        
        print('Derivative time: {:.3f}secs'.format(der_time))
        
        def predict_opt_par_from_hyper_par(epsilon): 
            return self.vb_init_free + dinput_hyper * epsilon * delta

        lr_list = []
        for epsilon in epsilon_vec: 
            # get linear response
            lr_list.append(predict_opt_par_from_hyper_par(epsilon))

        return np.array(lr_list)
    
    def get_free_param_results_from_perturbation(self, perturbation, delta):
        # wrapper function to load everything

        vb_refit_list, epsilon_vec = \
            self._load_refit_files(perturbation, delta)

        lr_list = self._get_lr_free_params(perturbation,
                                          epsilon_vec,
                                          delta)

        return vb_refit_list, lr_list, epsilon_vec
