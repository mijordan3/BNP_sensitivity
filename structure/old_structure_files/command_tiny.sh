#!/bin/bash

# Input format is bed by default.
python structure.py -K 2 --input=test_input_data/testdata_tiny \
    --output=test_input_data/testdata_tiny_out \
    --full --tol=1e-10


#    --starting_values_file=test_input_data/testdata_tiny_start.2
