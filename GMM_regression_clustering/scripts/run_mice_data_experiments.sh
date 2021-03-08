#!/bin/bash

seed=3453
alpha=6.0

out_folder='../fits/'
out_filename='mice_genomics_fit'

# #################
# # Initial fit
# #################
# python fit_gmm_regression.py \
#   --seed ${seed} \
#   --alpha ${alpha} \
#   --out_folder ${out_folder} \
#   --out_filename ${out_filename}_alpha${alpha} \


# #################
# # compute derivative
# #################
# python get_lr_derivative.py \
#   --out_folder ${out_folder} \
#   --fit_file ${out_filename}_alpha${alpha}.npz \

# #################
# # Refits
# #################

# # sigmoidal perturbations
# sbatch \
#     --array 0-18 \
#     --export=alpha=$alpha,out_folder=$out_folder,out_filename=$out_filename,perturbation='sigmoidal',delta=5 \
#     fit_gmmreg_perturbed.sh

# sbatch \
#     --array 0-18 \
#     --export=alpha=$alpha,out_folder=$out_folder,out_filename=$out_filename,perturbation='sigmoidal',delta=-5 \
#     fit_gmmreg_perturbed.sh

# # alpha perturbation
# sbatch \
#     --array 0-18 \
#     --export=alpha=$alpha,out_folder=$out_folder,out_filename=$out_filename,perturbation='alpha_pert_pos',delta=1 \
#     fit_gmmreg_perturbed.sh
# sbatch \
#     --array 0-18 \
#     --export=alpha=$alpha,out_folder=$out_folder,out_filename=$out_filename,perturbation='alpha_pert_neg',delta=1 \
#     fit_gmmreg_perturbed.sh

# # gaussian bumps
# sbatch \
#     --array 0-18 \
#     --export=alpha=$alpha,out_folder=$out_folder,out_filename=$out_filename,perturbation='gauss_pert1',delta=2 \
#     fit_gmmreg_perturbed.sh
# sbatch \
#     --array 0-18 \
#     --export=alpha=$alpha,out_folder=$out_folder,out_filename=$out_filename,perturbation='gauss_pert1',delta=-2 \
#     fit_gmmreg_perturbed.sh
# sbatch \
#     --array 0-18 \
#     --export=alpha=$alpha,out_folder=$out_folder,out_filename=$out_filename,perturbation='gauss_pert2',delta=2 \
#     fit_gmmreg_perturbed.sh
# sbatch \
#     --array 0-18 \
#     --export=alpha=$alpha,out_folder=$out_folder,out_filename=$out_filename,perturbation='gauss_pert2',delta=-2 \
#     fit_gmmreg_perturbed.sh

#################
# compute influence functions
#################
python get_influence_functions.py \
  --out_folder ${out_folder} \
  --fit_file ${out_filename}_alpha${alpha}.npz \

# refit at step functions
# for mu_indx in {0..12}
# do
#    sbatch --array 0-8 --export=mu_indx=$mu_indx,out_folder=$out_folder,out_filename=$out_filename fit_gmmreg_inflbump.sh
# done

# refit for worst-case
# sbatch --array 0-8 --export=g_name='num_clust',out_folder=$out_folder,out_filename=$out_filename,alpha=$alpha fit_gmmreg_worst_case.sh
# sbatch --array 0-8 --export=g_name='num_clust_pred',out_folder=$out_folder,out_filename=$out_filename,alpha=$alpha fit_gmmreg_worst_case.sh