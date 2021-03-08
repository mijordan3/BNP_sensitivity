#!/bin/bash

############################
# initial value of alpha
############################
alpha=6.0 

############################
# file paths and file names 
############################

# data_file=../data/phased_HGDP+India+Africa_2810SNPs-regions1to36.npz 
# out_filename=huang2011_fit
# out_folder=../fits/hgdp_fits/

data_file=../data/huang2011_sub_nobs25_nloci75.npz
out_filename=huang2011_fit_sub
out_folder=../fits/tmp/

############################
# get initial fit
############################
# ./fit_structure.sh $data_file $alpha $out_folder $out_filename  

############################
# get linear response derivatives
############################
# ./get_lr_derivative.sh $data_file $alpha $out_folder $out_filename  

############################
# get parametric sensitivity refits
############################
# sbatch \
#     --array 0-20 \
#     --export=data_file=$data_file,alpha=$alpha,out_folder=$out_folder,out_filename=$out_filename\
#     -p high \
#     fit_structure_on_alpha.sh
    
############################
# get functional sensitivity refits
############################
sbatch \
    -p high \
    --array 0-9 \
    --export=perturbation='sigmoidal',delta=5,data_file=$data_file,alpha=$alpha,out_folder=$out_folder,out_filename=$out_filename \
    fit_structure_on_epsilon.sh
    
sbatch \
    -p high \
    --array 0-9 \
    --export=perturbation='sigmoidal',delta=-5,data_file=$data_file,alpha=$alpha,out_folder=$out_folder,out_filename=$out_filename \
    fit_structure_on_epsilon.sh

sbatch \
    -p high \
    --array 0-9 \
    --export=perturbation='alpha_pert_pos',delta=1,data_file=$data_file,alpha=$alpha,out_folder=$out_folder,out_filename=$out_filename \
    fit_structure_on_epsilon.sh
sbatch \
    -p high \
    --array 0-9 \
    --export=perturbation='alpha_pert_neg',delta=1,data_file=$data_file,alpha=$alpha,out_folder=$out_folder,out_filename=$out_filename \
    fit_structure_on_epsilon.sh

sbatch \
    -p high \
    --array 0-9 \
    --export=perturbation='gauss_pert1',delta=1,data_file=$data_file,alpha=$alpha,out_folder=$out_folder,out_filename=$out_filename \
    fit_structure_on_epsilon.sh
sbatch \
    -p high \
    --array 0-9 \
    --export=perturbation='gauss_pert1',delta=-1,data_file=$data_file,alpha=$alpha,out_folder=$out_folder,out_filename=$out_filename \
    fit_structure_on_epsilon.sh

sbatch \
    -p high \
    --array 0-9 \
    --export=perturbation='gauss_pert2',delta=1,data_file=$data_file,alpha=$alpha,out_folder=$out_folder,out_filename=$out_filename \
    fit_structure_on_epsilon.sh
sbatch \
    -p high \
    --array 0-9 \
    --export=perturbation='gauss_pert2',delta=-1,data_file=$data_file,alpha=$alpha,out_folder=$out_folder,out_filename=$out_filename \
    fit_structure_on_epsilon.sh
