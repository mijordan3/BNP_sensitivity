# sbatch --array 0-18 --export=perturbation='worst_case',delta=1 fit_structure_on_epsilon.sh

sbatch --array 0-18 --export=perturbation='sigmoidal',delta=5 fit_structure_on_epsilon.sh
sbatch --array 0-18 --export=perturbation='sigmoidal',delta=-5 fit_structure_on_epsilon.sh

sbatch --array 0-18 --export=perturbation='alpha_pert_pos',delta=1 fit_structure_on_epsilon.sh
sbatch --array 0-18 --export=perturbation='alpha_pert_neg',delta=1 fit_structure_on_epsilon.sh

sbatch --array 0-18 --export=perturbation='alpha_pert_pos_xflip',delta=1 fit_structure_on_epsilon.sh
sbatch --array 0-18 --export=perturbation='alpha_pert_neg_xflip',delta=1 fit_structure_on_epsilon.sh

sbatch --array 0-18 --export=perturbation='gauss_pert1',delta=1 fit_structure_on_epsilon.sh
sbatch --array 0-18 --export=perturbation='gauss_pert1',delta=-1 fit_structure_on_epsilon.sh

sbatch --array 0-18 --export=perturbation='gauss_pert2',delta=1 fit_structure_on_epsilon.sh
sbatch --array 0-18 --export=perturbation='gauss_pert2',delta=-1 fit_structure_on_epsilon.sh
