#!/bin/bash



seed=345345

alpha=3.5

out_folder='../fits'
out_filename='testing'

data_dir='/home/rliu/Documents/BNP/fastStructure/hgdp_data/huang2011_plink_files/'
data_file=${data_dir}'phased_HGDP+India+Africa_2810SNPs-regions1to36.npz'

# get fit
python get_structure_fit.py \
  --seed ${seed} \
  --data_file ${data_file} \
  --alpha ${alpha} \
  --out_folder ${out_folder} \
  --out_filename ${out_filename}

# compute alpha sensitivity derivatives
# python get_alpha_derivative.py \
#     --data_file ${data_file} \
#     --fit_file ${out_folder}${out_filename}.npz \
#     --out_folder ${out_folder} \
#     --out_file alpha_sens_nobs${nobs}_nloci${nloci}_npop${npop}_alpha${alpha}
