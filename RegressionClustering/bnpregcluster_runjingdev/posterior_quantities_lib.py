# TODO: put the posterior quantites stuff here
import numpy as np
from bnpmodeling_runjingdev.cluster_quantities_lib import \
    get_e_num_large_clusters_from_ez
from bnpmodeling_runjingdev.cluster_quantities_lib import \
    get_e_number_clusters_from_logit_sticks

def get_posterior_quantity_function(predictive, gmm, n_samples, threshold):
    samples = None
    if not predictive:
        samples = np.random.random((gmm.num_obs, n_samples))
    else:
        samples = np.random.normal(
            0, 1, size = (n_samples, gmm.num_components - 1))

    def get_posterior_quantity(gmm_params):
        if not predictive:
            e_z = gmm.get_e_z(gmm_params)
            e_num, var_num = get_e_num_large_clusters_from_ez(
                e_z,
                threshold = threshold,
                n_samples = None,
                unif_samples = samples)
        else:
            e_num = \
                get_e_number_clusters_from_logit_sticks(
                    gmm_params['stick_propn_mean'],
                    gmm_params['stick_propn_info'],
                    n_obs = gmm.num_obs,
                    threshold = threshold,
                    n_samples = None,
                    unv_norm_samples = samples)
        return e_num

    return get_posterior_quantity
