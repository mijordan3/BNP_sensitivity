#!/bin/bash

# cp tiny_data/testdata_tiny.ped testdata_missing.ped
# cp tiny_data/testdata_tiny.map testdata_missing.map
# cp tiny_data/testdata_tiny.fam testdata_missing.fam

plink --file testdata_missing --make-bed --out testdata_missing --noweb
