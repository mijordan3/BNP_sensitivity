#!/bin/bash

# Individual 37 has been manually removed from testdata_small.ped.
cp testdata_small.map testdata_small_no37.map
cp testdata_small.fam testdata_small_no37.fam
plink --file testdata_small_no37 --make-bed --out testdata_small_no37 --noweb

cp testdata_small_out.3.varP testdata_small_out.3_no37_start.varP
cp testdata_small_out.3.varQ testdata_small_out.3_no37_start.varQ
sed -i.bak -e '37d' testdata_small_out.3_no37_start.varQ
