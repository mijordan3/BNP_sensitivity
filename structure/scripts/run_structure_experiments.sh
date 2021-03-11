#!/bin/bash

source activate bnp_sensitivity_jax

seed=234234

############################
# initial value of alpha
############################
alpha=3.0 

############################
# file paths and file names 
############################

data_file=../data/thrush_data/thrush-data.str
out_filename=thrush_fit
out_folder=../fits/thrush_fits/

############################
# get initial fit
############################
# python fit_structure.py \
#   --seed ${seed} \
#   --alpha ${alpha} \
#   --data_file ${data_file} \
#   --out_folder ${out_folder} \
#   --out_filename ${out_filename}_alpha${alpha} \
#   --k_approx 20 
  
############################
# get linear response derivatives
############################
# python get_lr_derivative.py \
#   --data_file ${data_file} \
#   --out_folder ${out_folder} \
#   --fit_file ${out_filename}_alpha${alpha}.npz \
#   --cg_tol 1e-8
    
############################
# get functional sensitivity refits
############################
# sbatch \
#     -p high \
#     --array 0-9 \
#     --export=perturbation='sigmoidal',delta=5,data_file=$data_file,alpha=$alpha,out_folder=$out_folder,out_filename=$out_filename \
#     fit_structure_on_epsilon.sh
    
# sbatch \
#     -p high \
#     --array 0-9 \
#     --export=perturbation='sigmoidal',delta=-5,data_file=$data_file,alpha=$alpha,out_folder=$out_folder,out_filename=$out_filename \
#     fit_structure_on_epsilon.sh

# sbatch \
#     -p high \
#     --array 0-9 \
#     --export=perturbation='alpha_pert_pos',delta=1,data_file=$data_file,alpha=$alpha,out_folder=$out_folder,out_filename=$out_filename \
#     fit_structure_on_epsilon.sh

# sbatch \
#     -p high \
#     --array 0-9 \
#     --export=perturbation='gauss_pert1',delta=1,data_file=$data_file,alpha=$alpha,out_folder=$out_folder,out_filename=$out_filename \
#     fit_structure_on_epsilon.sh
# sbatch \
#     -p high \
#     --array 0-9 \
#     --export=perturbation='gauss_pert1',delta=-1,data_file=$data_file,alpha=$alpha,out_folder=$out_folder,out_filename=$out_filename \
#     fit_structure_on_epsilon.sh

# sbatch \
#     -p high \
#     --array 0-9 \
#     --export=perturbation='gauss_pert2',delta=1,data_file=$data_file,alpha=$alpha,out_folder=$out_folder,out_filename=$out_filename \
#     fit_structure_on_epsilon.sh
# sbatch \
#     -p high \
#     --array 0-9 \
#     --export=perturbation='gauss_pert2',delta=-1,data_file=$data_file,alpha=$alpha,out_folder=$out_folder,out_filename=$out_filename \
#     fit_structure_on_epsilon.sh

#####################
# Refit at step functions
#####################
# for mu_indx in {0..9}
# do
# sbatch \
#     -p high \
#     --array 0-9 \
#     --export=data_file=$data_file,mu_indx=$mu_indx,out_folder=$out_folder,out_filename=$out_filename,alpha=$alpha \
#     fit_structure_bump.sh
# done


############################
# compute influence functions
############################
# python get_influence_functions.py \
#   --seed 4353453 \
#   --data_file ${data_file} \
#   --out_folder ${out_folder} \
#   --fit_file ${out_filename}_alpha${alpha}.npz \
#   --cg_tol 1e-8

#####################
# Refit at worst-case
#####################
sbatch \
    -p high \
    --array 0-9 \
    --export=data_file=$data_file,mu_indx=$mu_indx,out_folder=$out_folder,out_filename=$out_filename,alpha=$alpha,g_name='num_clust' \
    fit_structure_worst_case.sh
    
sbatch \
    -p high \
    --array 0-9 \
    --export=data_file=$data_file,mu_indx=$mu_indx,out_folder=$out_folder,out_filename=$out_filename,alpha=$alpha,g_name='num_clust_pred' \
    fit_structure_worst_case.sh