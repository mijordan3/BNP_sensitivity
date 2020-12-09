#!/bin/bash
python ../structure.py -K 3 \
       --input=testdata_small \
       --output=testdata_small_out \
       --full --seed=42 --tol=1e-6

