#!/bin/bash

sbatch --array 0-12 --export=mu_indx=0 fit_structure_gauss_bump.sh
sbatch --array 0-12 --export=mu_indx=1 fit_structure_gauss_bump.sh
sbatch --array 0-12 --export=mu_indx=2 fit_structure_gauss_bump.sh
sbatch --array 0-12 --export=mu_indx=3 fit_structure_gauss_bump.sh
sbatch --array 0-12 --export=mu_indx=4 fit_structure_gauss_bump.sh
sbatch --array 0-12 --export=mu_indx=5 fit_structure_gauss_bump.sh
