#!/bin/bash
python ../structure.py -K 3 \
       --input=testdata_small \
       --output=testdata_small_restart_out \
       --full --seed=42 --tol=1e-9 \
       --starting_values_file=testdata_small_out.3 \
       --accelerated=False


