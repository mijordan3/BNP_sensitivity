#!/bin/bash

# The first six columns are ids and there is no header.  Recall that there are two
# columns for each allele.
cat testdata_plain.ped | head -n 5 | cut -d' ' -f1-26 > testdata_tiny.ped

# There is no header row.
cat testdata_plain.map | head -n 10 > testdata_tiny.map
cat testdata.fam | head -n 10 > testdata_tiny.fam

plink --file testdata_tiny --make-bed --out testdata_tiny --noweb
