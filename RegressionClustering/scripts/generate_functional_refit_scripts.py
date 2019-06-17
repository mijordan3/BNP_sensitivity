#!/usr/bin/env python3
"""Create shell scripts to run ``refit.py`` for a range of parameters.
Optionally submit the shell scripts to slurm.

./generate_functional_refit_scripts.py \
    --fit_dir '/home/rgiordan/Documents/git_repos/BNP_sensitivity/RegressionClustering/fits' \
    --no-submit

./generate_functional_refit_scripts.py  --fit_dir '/accounts/grad/rgiordano/Documents/git_repos/BNP_sensitivity/RegressionClustering/fits'  --no-submit

"""

import argparse
import numpy as np
import os
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument('--submit', dest='submit', action='store_true',
                    help='Submit to slurm.')
parser.add_argument('--no-submit', dest='submit', action='store_false',
                    help='Do not submit to slurm.')
parser.add_argument('--fit_dir', required=True, type=str)
parser.set_defaults(submit=False)

args = parser.parse_args()

# Set parameters for refitting.
script_dir = './slurm_scripts'
if not os.path.isdir(script_dir):
    raise ValueError('Script directory {} does not exist.'.format(script_dir))

small = True
if small:
    # Use strings to avoid formatting problems.
    initial_alpha = '2.0'
    num_components = '40'
    inflate = '0.0'
    #inflate = '1.0'
    genes = '700'
    alpha_scales = np.linspace(0.001, 1.0, 10)
    alpha_scales = [np.round(alpha, 5) for alpha in alpha_scales]
else:
    # Use strings to avoid formatting problems.
    initial_alpha = '2.0'
    num_components = '60'
    inflate = '0.0'
    #inflate = '1.0'
    genes = '7000'
    alpha_scales = np.linspace(0.001, 1.0, 10)
    alpha_scales = [np.round(alpha, 5) for alpha in alpha_scales]

initial_fit_template = \
    ('{fit_dir}/transformed_gene_regression_df4_degree3_genes{genes}_' +
     'num_components{num_components}_inflate{inflate}_' +
     'shrunkTrue_alpha{alpha}_fit.npz')

initial_fitfile = initial_fit_template.format(
    fit_dir=args.fit_dir,
    alpha=initial_alpha, num_components=num_components,
    inflate=inflate, genes=genes)

if not os.path.isfile(initial_fitfile):
    raise ValueError('Input file {} does not exist.'.format(initial_fitfile))

activate_venv_cmd = 'source ../../venv/bin/activate'

for alpha_scale in alpha_scales:
    script_name = ('refit_script_functionalpert_' +
        'genes{genes}_inflate{inflate}_epsilon{alpha_scale}.sh').format(
        genes=genes, inflate=inflate, alpha_scale=alpha_scale)
    full_script_name = os.path.join(script_dir, script_name)
    with open(full_script_name, 'w') as slurm_script:
        slurm_script.write('#!/bin/bash\n')
        slurm_script.write(activate_venv_cmd + '\n')
        cmd = ('../refit.py ' +
               '--fit_directory {fit_directory} ' +
               '--input_filename {input_filename} ' +
               '--alpha_scale {alpha_scale} ' +
               '--functional'
               '\n').format(
            fit_directory=args.fit_dir,
            alpha_scale=alpha_scale,
            input_filename=initial_fitfile
            )
        slurm_script.write(cmd)
    if args.submit:
        print('Submitting {}'.format(full_script_name))
        command = ['sbatch', full_script_name]
        subprocess.run(command)
    else:
        print('Generating (but not submitting) shell script {}'.format(
            full_script_name))
