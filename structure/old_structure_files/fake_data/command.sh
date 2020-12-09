#!/bin/bash
./filter_data.sh
python ../structure.py -K 2 --input=testdata_fake --output=out/testdata_fake_out --full --seed=42 --tol=1e-9

