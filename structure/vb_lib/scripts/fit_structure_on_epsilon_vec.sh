sbatch --array 0-10 --export=perturbation='worst-case' fit_structure_on_epsilon.sh
sbatch --array 0-10 --export=perturbation='sigmoidal' fit_structure_on_epsilon.sh
