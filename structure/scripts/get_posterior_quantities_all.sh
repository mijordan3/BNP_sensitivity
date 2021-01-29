#!/bin/bash

sbatch --array 0-18 --export=perturbation='gauss_pert1_pos' fit_structure_on_epsilon.sh
