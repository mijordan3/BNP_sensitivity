#!/bin/bash
python ../structure.py -K 3 \
       --input=testdata_small \
       --output=testdata_small_beta_pert_out \
       --full --seed=42 --tol=1e-9 \
       --starting_values_file=testdata_small_out.3 \
       --prior_beta=1.1 \
       --full --seed=42 --tol=1e-9
