#!/bin/bash

filename=phased_HGDP+India+Africa_2810SNPs-regions1to36

# download data
wget http://rosenberglab.stanford.edu/data/huangEtAl2011/${filename}.stru

# convert structure to plink 
python stru_to_plink.py \
        --input_file=$filename.stru \
        --output_file=$filename

# convert from .ped to .bed
echo 'ped to bed'
./plink --file ${filename} \
        --make-bed \
        --out ${filename}

# convert .bed to .npz
echo 'bed to np array'
python plink_to_numpy.py \
    --input ${filename} \
    --output ${filename}
