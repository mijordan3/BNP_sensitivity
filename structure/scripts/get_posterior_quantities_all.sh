#!/bin/bash

sbatch --array 0-18 --export=perturbation='worst_case' get_posterior_quantities_epsilon.sh

sbatch --array 0-18 --export=perturbation='worst_case' get_posterior_quantities_epsilon.sh
sbatch --array 0-18 --export=perturbation='sigmoidal' get_posterior_quantities_epsilon.sh
sbatch --array 0-18 --export=perturbation='sigmoidal_neg' get_posterior_quantities_epsilon.sh

sbatch --array 0-18 --export=perturbation='alpha_pert_pos' get_posterior_quantities_epsilon.sh
sbatch --array 0-18 --export=perturbation='alpha_pert_neg' get_posterior_quantities_epsilon.sh

sbatch --array 0-18 --export=perturbation='alpha_pert_pos_xflip' get_posterior_quantities_epsilon.sh
sbatch --array 0-18 --export=perturbation='alpha_pert_neg_xflip' get_posterior_quantities_epsilon.sh

sbatch --array 0-18 --export=perturbation='gauss_pert1_pos' get_posterior_quantities_epsilon.sh
sbatch --array 0-18 --export=perturbation='gauss_pert1_neg' get_posterior_quantities_epsilon.sh

sbatch --array 0-18 --export=perturbation='gauss_pert2_pos' get_posterior_quantities_epsilon.sh
sbatch --array 0-18 --export=perturbation='gauss_pert2_neg' get_posterior_quantities_epsilon.sh
