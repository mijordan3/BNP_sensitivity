import numpy as np

import jax.numpy as jnp

from scipy import spatial
import scipy.cluster.hierarchy as sch

from itertools import permutations

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

    return g_obs

def draw_data(n_obs, n_loci, n_pop, 
              mem_saver = False, 
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
    mem_saver : boolean
        Whether to compute g_obs in a for-loop, to save memory.
        Needed for simulating particularly large datasets ...
    """


    # draw population allele frequencies
    true_pop_allele_freq = np.random.random((n_loci, n_pop))

    # individual admixtures
    true_ind_admix_propn = \
        np.random.dirichlet(np.ones(n_pop) / n_pop, size = (n_obs))

    # cluster the individuals (just for better plotting)
    clustering_indx = cluster_admix_get_indx(true_ind_admix_propn)
    true_ind_admix_propn = true_ind_admix_propn[clustering_indx, :]

    # draw data in batches (useful for simulating large datasets)
    if mem_saver:
        batchsize = 10
    else:
        batchsize = n_obs

    n_batches = int(np.ceil(n_obs / batchsize))
    g_obs = np.zeros((n_batches * batchsize, n_loci, 3))
    for i in range(n_batches):
        indx0 = batchsize * i
        indx1 = batchsize * (i+1) 
                
        g_obs[indx0:indx1] = \
            draw_data_from_popfreq_and_admix(true_pop_allele_freq,
                                             true_ind_admix_propn[indx0:indx1])
    g_obs = g_obs[0:n_obs]
    
    if save_as_jnp: 
        g_obs = jnp.array(g_obs) 
        true_pop_allele_freq = jnp.array(true_pop_allele_freq)
        true_ind_admix_propn = jnp.array(true_ind_admix_propn)
        
    return g_obs, true_pop_allele_freq, true_ind_admix_propn


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
