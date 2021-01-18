#!/bin/bash

source activate bnp_sensitivity_jax

seed=432452101

# simulate a tiny dataset for testing and debugging
# python simulate_structure_data.py \
#             --seed ${seed} \
#             --n_obs 20 \
#             --n_loci 50 \
#             --n_pop 4
            

# simulate hgdp data
python simulate_structure_data.py \
            --seed ${seed} \
            --n_obs 1107 \
            --n_loci 2810 \
            --n_pop 6
            