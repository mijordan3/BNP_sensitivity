#!/bin/bash
python ../structure.py -K 3 \
       --input=testdata_small_no37 \
       --output=testdata_small_no37_out \
       --starting_values_file=testdata_small_out.3_no37_start \
       --full --seed=42 --tol=1e-9

