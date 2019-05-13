import autograd

import autograd.numpy as np
import autograd.scipy as sp
from numpy.polynomial.hermite import hermgauss

from vb_lib import structure_model_lib, data_utils

import paragami

import unittest

np.random.seed(25465)

# set some trivial parameters, and check that
# a draw of genotypes makes sense from the underlying
# population allele frequencies and individual admixtures
class TestDrawData(unittest.TestCase):
    def test_draw_data(self):
        # draw data
        n_obs = 10
        n_loci = 5
        n_pop = n_loci

        # population allele frequencies: all or nothing, so easy to identify
        p1 = 1.0
        p0 = 0.0
        pop_allele_freq = np.maximum(np.eye(n_loci, n_pop) * p1, p0)

        # individual admixtures: everyone belongs to exactly one population
        ind_admix_propn = np.random.choice(n_pop, n_obs)
        ind_admix_propn = data_utils.get_one_hot(ind_admix_propn, nb_classes = n_pop)

        # draw genotypes
        g_obs = data_utils.draw_data_from_popfreq_and_admix(pop_allele_freq,
                                            ind_admix_propn)
        _g_obs = (g_obs.argmax(axis = 2) == 2) # places where its both AA

        assert np.all(_g_obs == ind_admix_propn)
