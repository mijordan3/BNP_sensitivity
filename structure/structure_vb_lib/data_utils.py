import numpy as np

import jax.numpy as jnp

from scipy import spatial
import scipy.cluster.hierarchy as sch

from itertools import permutations


###################
# Functions to simulate data
###################

def get_one_hot(targets, nb_classes):
    res = np.eye(nb_classes)[np.array(targets).reshape(-1)]
    return res.reshape(list(targets.shape)+[nb_classes])

def draw_data_from_popfreq_and_admix(pop_allele_freq, ind_admix_propn):
    """
    Draws data for structure

    Parameters
    ----------
    true_pop_allele_freq : ndarray
        The true population frequencies from which to draw g_obs,
        in a (n_loci x n_population) array.
    true_ind_admix_propn : ndarray
        The true individual admixtures from which to draw g_obs,
        in a (n_obs x n_population) array.

    Returns
    -------
    g_obs : ndarray
        Array of size (n_obs x n_loci x 3), giving a one-hot encoding of
        genotypes

    """

    n_obs = ind_admix_propn.shape[0]
    n_pop = ind_admix_propn.shape[1]

    n_loci = pop_allele_freq.shape[0]
    assert pop_allele_freq.shape[1] == n_pop

    # population belongings for each loci
    z_a = np.array([np.random.choice(n_pop, p=row, size = (n_loci))
              for row in ind_admix_propn])
    z_b = np.array([np.random.choice(n_pop, p=row, size = (n_loci))
              for row in ind_admix_propn])
    z_a_onehot = get_one_hot(z_a, nb_classes=n_pop)
    z_b_onehot = get_one_hot(z_b, nb_classes=n_pop)


    # allele frequencies for each individual at each loci
    ind_allele_freq_a = np.einsum('nlk, lk -> nl', z_a_onehot, pop_allele_freq)
    ind_allele_freq_b = np.einsum('nlk, lk -> nl', z_b_onehot, pop_allele_freq)

    # draw genotypes at each chromosome
    genotype_a = (np.random.random((n_obs, n_loci)) < ind_allele_freq_a).astype(int)
    genotype_b = (np.random.random((n_obs, n_loci)) < ind_allele_freq_b).astype(int)

    # we only observe their sum
    g_obs = genotype_a + genotype_b
    g_obs = get_one_hot(g_obs, nb_classes=3)

    return g_obs, np.stack((z_a_onehot, z_b_onehot), axis = -1)

def draw_data(n_obs, n_loci, n_pop, 
              save_as_jnp = False):
    """
    Draws data for structure

    Parameters
    ----------
    n_obs : integer
        The number of observations
    n_loci : integer
        The number of loci per observation
    n_pop : integer
        The number of populations in the model

    Returns
    -------
    g_obs : ndarray
        Array of size n_obs x n_loci x 3, giving a one-hot encoding of
        genotypes
    true_pop_allele_freq : ndarray
        The true population frequencies from which g_obs was drawn,
        in a (n_loci x n_population) array.
    true_ind_admix_propn : ndarray
        The true individual admixtures from which g_obs was drawn,
        in a (n_obs x n_population) array.
    
    """


    # draw population allele frequencies
    true_pop_allele_freq = np.random.random((n_loci, n_pop))

    # individual admixtures
    true_ind_admix_propn = \
        np.random.dirichlet(np.ones(n_pop) / n_pop, size = (n_obs))

    # cluster the individuals (just for better plotting)
    clustering_indx = cluster_admix_get_indx(true_ind_admix_propn)
    true_ind_admix_propn = true_ind_admix_propn[clustering_indx, :]
    
    # draw data
    g_obs, true_z = draw_data_from_popfreq_and_admix(true_pop_allele_freq,
                                                     true_ind_admix_propn)
    
    if save_as_jnp: 
        # save as jax numpy object
        g_obs = jnp.array(g_obs) 
        true_pop_allele_freq = jnp.array(true_pop_allele_freq)
        true_ind_admix_propn = jnp.array(true_ind_admix_propn)
        
    return g_obs, true_z, true_pop_allele_freq, true_ind_admix_propn


####################
# Other utils for
# permuting / clustering matrices
####################
def find_min_perm(x, y, axis = 0):
    # perumutes array x along `axis' to find closest
    # match to y

    perms = list(permutations(np.arange(x.shape[axis])))

    i = 0
    diff_best = np.Inf
    for perm in perms:

        x_perm = x.take(perm, axis)

        diff = np.sum((x_perm - y)**2)

        if diff < diff_best:
            diff_best = diff
            i_best = i

        i += 1

    return perms[i_best]

def cluster_admix_get_indx(ind_admix_propn):
    # clusters the individual admixtures for better plotting
    y = sch.linkage(ind_admix_propn, method='average')
    indx = sch.dendrogram(y, no_plot=True)["leaves"]

    return indx


################
# Function to load thrush data
################
def load_thrush_data(data_filename = '../data/thrush_data/thrush-data.str'): 
    data_raw = np.loadtxt(data_filename)
    
    # rows are individuals x chromosome: 
    # so number of rows should be divisible by 2
    assert (data_raw.shape[0] % 2) == 0
    
    # number of individuals
    n_ind = int(data_raw.shape[0] / 2)

    # this n ind x n_columns x 2
    data_raw2d = np.array([data_raw[(2*i):(2*i+2)].transpose() for i in range(n_ind)])
    
    # the population
    labels = data_raw2d[:, 1, :]
    assert np.all(labels[:, 0] == labels[:, 1])
    labels = labels[:, 0]
    
    # the genotypes
    genotypes = data_raw2d[:, 2:, :]
    n_loci = genotypes.shape[1]

    # sort by population
    perm = np.argsort(labels)
    labels = labels[perm]
    genotypes = genotypes[perm]
    
    # get one-hot encoding
    unique_alleles = np.unique(genotypes)
    unique_alleles = unique_alleles[unique_alleles != -9]
    
    genotypes_one_hot = np.zeros((n_ind, n_loci, 2, len(unique_alleles)))
    
    for i in range(len(unique_alleles)): 
        genotypes_one_hot[:, :, :, i] = (genotypes == unique_alleles[i])
        
    return jnp.array(genotypes_one_hot), jnp.array(genotypes), labels, unique_alleles