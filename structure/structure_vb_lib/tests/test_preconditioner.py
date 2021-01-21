import jax

import jax.numpy as np
import jax.scipy as sp

import scipy as osp

from structure_vb_lib import structure_model_lib, preconditioner_lib, testutils

from bnpmodeling_runjingdev.sensitivity_lib import HyperparameterSensitivityLinearApproximation

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


###############
# test preconditioner 
###############
class TestStructurePreconditioner(unittest.TestCase):        
    
    @classmethod
    def setUpClass(cls):
        # get model at optimum
        cls.vb_opt, cls.vb_params_paragami, cls.precond_objective = \
            testutils.construct_model_and_optimize(debug_cavi = False, 
                                                   seed = 3421)
    
    def test_preconditioner_against_autodiff(self): 
        
        # get model 
        g_obs, vb_params_dict, vb_params_paragami, \
            prior_params_dict, gh_loc, gh_weights = \
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
    
    def test_hessian_eigenvals_at_optimum(self): 
        
        # the hessian without preconditioning
        kl_hess = jax.lax.map(lambda v : self.precond_objective.hvp(self.vb_opt, v), 
                              np.eye(len(self.vb_opt)))
        
        # the preconditioned hessian 
        x_c = self.precond_objective.precondition(self.vb_opt, self.vb_opt)
        kl_hess_precond = jax.lax.map(lambda v : \
                                      self.precond_objective.hvp_precond(x_c, self.vb_opt, v), 
                                      np.eye(len(self.vb_opt)))
        
        def evaluate_condition_number(kl_hess): 
            # get eigenvalues
            kl_hess_evals = np.linalg.eigvals(kl_hess)
            
            # all real
            assert np.all(np.imag(kl_hess_evals) == 0.)
            kl_hess_evals = np.real(kl_hess_evals)
            
            # all positive 
            assert np.all(kl_hess_evals) > 0
            
            print((kl_hess_evals.max(), 
                   kl_hess_evals.min()))
            
            cn_hess = kl_hess_evals.max() / \
                        kl_hess_evals.min()
                
            return cn_hess
        
        print('Hessian eigenvalues: ')
        cn_hess = evaluate_condition_number(kl_hess)
        print('Precond Hessian eigenvalues: ')
        cn_hess_precond = evaluate_condition_number(kl_hess_precond)
        assert cn_hess_precond < cn_hess

    def test_hessian_solver(self): 
        # now check solver
        vb_opt_dict = self.vb_params_paragami.fold(self.vb_opt, free = True)
        
        cg_precond = lambda v : preconditioner_lib.get_mfvb_cov_matmul(v, vb_opt_dict,
                                            self.vb_params_paragami,
                                            return_sqrt = False, 
                                            return_info = True)

        # define sensitivity class 
        kwargs = dict(objective_fun = lambda x, y : 0., 
                      opt_par_value = self.vb_opt, 
                      hyper_par_value0 = np.array([0.]), 
                      obj_fun_hvp = self.precond_objective.hvp, 
                      # a null perturbation ... will set later
                      hyper_par_objective_fun = lambda x, y : 0.,
                      # this lets ust track the progress of the solver
                      use_scipy_cgsolve = True, 
                      cg_precond = None, 
		      cg_tol = 1e-8)
        
        # solver without preconditioner
        vb_sens = HyperparameterSensitivityLinearApproximation(**kwargs)
        
        # solver with preconditioner
        kwargs.update({'cg_precond': cg_precond})
        vb_sens_precond = HyperparameterSensitivityLinearApproximation(**kwargs)
        
        # the "cross-hessian"
        b = jax.random.normal(key = jax.random.PRNGKey(443), shape = (len(self.vb_opt), ))
        
        # compare solvers
        out1 = vb_sens.hessian_solver(b)
        out2 = vb_sens_precond.hessian_solver(b)
        
        # outputs should be the same
        assert np.abs(out1 - out2).max() < 1e-3
        
        # preconditioned solver should converge faster
        assert vb_sens.cg_solver.iter > vb_sens_precond.cg_solver.iter
        
