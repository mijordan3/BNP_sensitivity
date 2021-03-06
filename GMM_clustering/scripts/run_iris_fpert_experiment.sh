#!/bin/bash

seed=789970
alpha=6.0

out_folder='../fits/'
out_filename='iris_fit'

#################
# Initial fit
#################
python fit_iris.py \
  --seed ${seed} \
  --alpha ${alpha} \
  --out_folder ${out_folder} \
  --out_filename ${out_filename}_alpha${alpha} \


#################
# compute derivative
#################
python get_lr_derivative.py \
  --out_folder ${out_folder} \
  --fit_file ${out_filename}_alpha${alpha}.npz \

#################
# Refits for functional perturbations
################# 

# sigmoidal perturbations
sbatch \
    --array 0-18 \
    --export=alpha=$alpha,out_folder=$out_folder,out_filename=$out_filename,perturbation='sigmoidal',delta=5 \
    fit_iris_perturbed.sh

sbatch \
    --array 0-18 \
    --export=alpha=$alpha,out_folder=$out_folder,out_filename=$out_filename,perturbation='sigmoidal',delta=-5 \
    fit_iris_perturbed.sh

# alpha perturbation
sbatch \
    --array 0-18 \
    --export=alpha=$alpha,out_folder=$out_folder,out_filename=$out_filename,perturbation='alpha_pert_pos',delta=1 \
    fit_iris_perturbed.sh
sbatch \
    --array 0-18 \
    --export=alpha=$alpha,out_folder=$out_folder,out_filename=$out_filename,perturbation='alpha_pert_neg',delta=1 \
    fit_iris_perturbed.sh

# gaussian bumps
sbatch \
    --array 0-18 \
    --export=alpha=$alpha,out_folder=$out_folder,out_filename=$out_filename,perturbation='gauss_pert1',delta=2 \
    fit_iris_perturbed.sh
sbatch \
    --array 0-18 \
    --export=alpha=$alpha,out_folder=$out_folder,out_filename=$out_filename,perturbation='gauss_pert1',delta=-2 \
    fit_iris_perturbed.sh
sbatch \
    --array 0-18 \
    --export=alpha=$alpha,out_folder=$out_folder,out_filename=$out_filename,perturbation='gauss_pert2',delta=2 \
    fit_iris_perturbed.sh
sbatch \
    --array 0-18 \
    --export=alpha=$alpha,out_folder=$out_folder,out_filename=$out_filename,perturbation='gauss_pert2',delta=-2 \
    fit_iris_perturbed.sh

#################
# Refits for step function perturbations
################# 
./fit_iris_inflbump_vec.sh
