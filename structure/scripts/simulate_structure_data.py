import numpy as np
import os

from vb_lib import data_utils

import argparse
import distutils.util

parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, default=4143122)
parser.add_argument('--n_obs', type=int)
parser.add_argument('--n_loci', type=int)
parser.add_argument('--n_pop', type=int)

# where to save the structure fit
parser.add_argument('--outfolder', type=str,
                    default='../simulated_data/')
args = parser.parse_args()

np.random.seed(args.seed)

# parameters
n_obs = args.n_obs
n_loci = args.n_loci
n_pop = args.n_pop

# directories
data_file = 'simulated_structure_data_nobs' + str(n_obs) + \
                '_nloci' + str(n_loci) + \
                '_npop' + str(n_pop) + '.npz'

outfile = os.path.join(args.outfolder, data_file)
print('outfile: ', outfile)

print('generating data ...')
g_obs, true_z, true_pop_allele_freq, true_ind_admix_propn = \
        data_utils.draw_data(n_obs, n_loci, n_pop)

np.savez(outfile, 
            g_obs = g_obs,
            true_z = true_z,
            true_pop_allele_freq = true_pop_allele_freq,
            true_ind_admix_propn = true_ind_admix_propn)

print('done. ')