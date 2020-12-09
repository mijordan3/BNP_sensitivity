#!/bin/bash

source activate bnp_sensitivity_jax

seed=432452101

# simulate a tiny dataset for testing and debugging
python simulate_structure_data.py \
            --seed ${seed} \
            --n_obs 20 \
            --n_loci 50 \
            --n_pop 4
            

# # simulate a small dataset
# python simulate_structure_data.py \
#             --seed ${seed} \
#             --n_obs 200 \
#             --n_loci 500 \
#             --n_pop 4
            
# # simulate a medium dataset
# python simulate_structure_data.py \
#             --seed ${seed} \
#             --n_obs 1000 \
#             --n_loci 70000 \
#             --n_pop 4 \
#             --mem_saver True