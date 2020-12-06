import jax.numpy as np


class FunctionalPerturbationObjectives(): 
    def __init__(self, 
                 e_log_phi,
                 vb_params_paragami): 

        # e_log_phi takes input mean and info
        # and returns the **additve** perturbation 
        # to the **ELBO** 

        self.e_log_phi = e_log_phi
        self.vb_params_paragami = vb_params_paragami
    
    def e_log_phi_epsilon(self, means, infos, epsilon): 
        # with epsilon fixed this is the input to the optimizer
        # (this is added to the ELBO)
        
        return epsilon * self.e_log_phi(means, infos)
    
    def hyper_par_objective_fun(self,
                                vb_params_free, 
                                epsilon): 
        # NOTE THE NEGATIVE SIGN
        # this is passed into the HyperparameterSensitivity class
        # and is added to the **KL** 

        vb_params_dict = self.vb_params_paragami.fold(vb_params_free, 
                                                free = True)
    
        # get means and infos 
        means = vb_params_dict['ind_admix_params']['stick_means']
        infos = vb_params_dict['ind_admix_params']['stick_infos']
        
        return - self.e_log_phi_epsilon(means, infos, epsilon)

