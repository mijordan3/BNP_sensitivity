import autograd

import autograd.numpy as np
import autograd.scipy as sp
from numpy.polynomial.hermite import hermgauss

import sys
sys.path.insert(0, '../')

import structure_model_lib
import data_utils

import argparse
import distutils.util

import os

parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, default=45426543)
parser.add_argument('--n_obs', type=int, default=50)
parser.add_argument('--n_loci', type=int, default=40)
parser.add_argument('--n_pop', type=int, default=4)

parser.add_argument('--outfolder', default='../data/')

args = parser.parse_args()

assert os.path.exists(args.outfolder)

n_obs = args.n_obs
n_loci = args.n_loci
n_pop = args.n_pop
seed = args.seed

np.random.seed(seed)

# population allele frequencies
true_pop_allele_freq = np.random.random((n_loci, n_pop))

# individual admixtures
true_ind_admix_propn = np.random.dirichlet(np.ones(n_pop) / n_pop, size = (n_obs))

# cluster the individuals
clustering_indx = data_utils.cluster_admix_get_indx(true_ind_admix_propn)
true_ind_admix_propn = true_ind_admix_propn[clustering_indx, :]

# draw data
g_obs = data_utils.draw_data(true_pop_allele_freq, true_ind_admix_propn)

# save data
outfilename = 'simulated_structure_data_nobs{}_nloci{}_npop{}'.format(
                    n_obs, n_loci, n_pop)
outfile = os.path.join(args.outfolder, outfilename)
print('saving data to ', outfile)

np.savez(outfile, g_obs = g_obs,
                true_pop_allele_freq = true_pop_allele_freq,
                true_ind_admix_propn = true_ind_admix_propn)
