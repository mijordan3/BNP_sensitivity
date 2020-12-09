#!/bin/bash

source activate bnp_sensitivity_jax

epsilon_indx=$SLURM_ARRAY_TASK_ID

# data_file=../data/phased_HGDP+India+Africa_2810SNPs-regions1to36.npz 
# out_filename=huang2011_fit
# out_folder=../fits/hgdp_fits/

data_file=../simulated_data/simulated_structure_data_nobs20_nloci50_npop4.npz
out_filename=simulated_fit
out_folder=../fits/tmp/

python fit_structure_perturbed.py \
  --perturbation ${perturbation} \
  --epsilon_indx ${epsilon_indx} \
  --data_file ${data_file} \
  --out_folder ${out_folder} \
  --out_filename ${out_filename}_${perturbation}${epsilon_indx} \
  --init_fit ${out_folder}${out_filename}_alpha6.0.npz \
  --influence_file ${out_folder}${out_filename}_alpha6.0_lrderivatives.npz
