import jax.numpy np

import numpy as onp
import os

from vb_lib import data_utils

# parameters
seed = 2387372

n_obs = 1000
n_loci = 700000
n_pop = 4

# directories
scratch_folder = '/scratch/users/genomic_times_series_bnp/structure/data/'
data_file = 'simulated_structure_data_nobs' + nobs + \
                '_nloci' + nloci + \
                '_npop' + npop + '.npz'

outfile = os.path.join(scratch_folder, data_file)
print('outfile: ', outfile)

onp.random.seed(seed)

print('generating data ...')
g_obs, true_pop_allele_freq, true_ind_admix_propn = \
        data_utils.draw_data(n_obs, n_loci, n_pop, mem_saver=True)

np.savez(args.data_file,
        g_obs = g_obs,
        true_pop_allele_freq = true_pop_allele_freq,
        true_ind_admix_propn = true_ind_admix_propn)
