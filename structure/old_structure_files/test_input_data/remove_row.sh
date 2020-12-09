#!/bin/bash

# Remove a single row and re-run the analysis.
cat testdata_plain.ped | sed '165d' > testdata_plain_no165.ped
cp testdata_plain.map testdata_plain_no165.map
cp testdata_plain.fam testdata_plain_no165.fam

plink --file testdata_plain_no165 --make-bed --out testdata_plain_no165 --noweb
python ../structure.py -K 3 --input=testdata_plain_no165 --output=testdata_plain_no165_out --full --seed=42 --tol=1e-9

