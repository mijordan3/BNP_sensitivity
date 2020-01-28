import autograd
import autograd.numpy as np

from bnpmodeling_runjingdev.modeling_lib import my_slogdet3d

import paragami

# def get_mvn_log_partition(mean, info):
#     assert mean.shape[0] == info.shape[0]
#     assert mean.shape[1] == info.shape[1]
#
#     info_mean = np.einsum('kij, kj -> ki', info, mean)
#     mean_info_mean = np.einsum('ki, ki -> k', mean, info_mean)
#
#     return (0.5 * mean_info_mean - 0.5 * my_slogdet3d(info)[1]).sum()
#
# def get_log_partition_free(vb_params_free, vb_params_paragami,
#                             use_logitnormal_sticks = True):
#     vb_params_dict = vb_params_paragami.fold(vb_params_free, free = True)
#
#     cluster_log_part = get_mvn_log_partition(\
#                     vb_params_dict['cluster_params']['centroids'].transpose(),
#                     vb_params_dict['cluster_params']['cluster_info'])
#
#     if use_logitnormal_sticks:
#         stick_log_part = get_mvn_log_partition(\
#             vb_params_dict['stick_params']['stick_propn_mean'][:, None],
#             vb_params_dict['stick_params']['stick_propn_info'][:, None, None])
#
#     else:
#         raise NotImplementedError()
#
#     return cluster_log_part + stick_log_part
#
# get_neg_fishers_info = autograd.hessian(get_log_partition_free, 0)

def get_nat_vec(param_vec, mvn_params_paragami, mvn_nat_params_paragami):

    nat_params_dict = {}

    mvn_param_dict = mvn_params_paragami.fold(param_vec, free = True)

    mean = mvn_param_dict['mean']
    info = mvn_param_dict['info']

    nat_params_dict['nat1'] = np.einsum('kij, kj -> ki', info, mean)
    nat_params_dict['neg_nat2'] = 0.5 * info

    return mvn_nat_params_paragami.flatten(nat_params_dict, free = False)

def get_mvn_log_partition(nat_vec, mvn_nat_params_paragami):

    nat_params_dict = mvn_nat_params_paragami.fold(nat_vec, free = False)

    nat1 = nat_params_dict['nat1']
    neg_nat2 = nat_params_dict['neg_nat2']

    nat2_inv = np.linalg.inv(-neg_nat2)

    nat2_inv_nat1 = np.einsum('kij, kj -> ki', nat2_inv, nat1)
    squared_term = np.einsum('ki, ki -> k', nat1, nat2_inv_nat1)

    return (- 0.25 * squared_term - 0.5 * my_slogdet3d(2 * neg_nat2)[1]).sum()

get_jac_term = autograd.jacobian(get_nat_vec, 0)
get_log_part_hess = autograd.hessian(get_mvn_log_partition, 0)

def get_mvn_paragami_objects(k_approx, dim):
    mvn_nat_params_paragami = paragami.PatternDict()
    mvn_nat_params_paragami['nat1'] = \
        paragami.NumericArrayPattern(shape=(k_approx, dim))
    mvn_nat_params_paragami['neg_nat2'] = \
        paragami.pattern_containers.PatternArray(array_shape = (k_approx, ), \
                    base_pattern = paragami.PSDSymmetricMatrixPattern(size=dim))

    mvn_params_paragami = paragami.PatternDict()
    mvn_params_paragami['mean'] = \
        paragami.NumericArrayPattern(shape=(k_approx, dim))
    mvn_params_paragami['info'] = \
        paragami.pattern_containers.PatternArray(array_shape = (k_approx, ), \
                    base_pattern = paragami.PSDSymmetricMatrixPattern(size=dim))

    return mvn_params_paragami, mvn_nat_params_paragami


def get_fishers_info(mean, info):
    assert mean.shape[0] == info.shape[0]
    assert mean.shape[1] == info.shape[1]

    k_approx = mean.shape[0]
    dim = mean.shape[1]

    # get paragami objects
    mvn_params_paragami, mvn_nat_params_paragami = \
        get_mvn_paragami_objects(k_approx, dim)

    # dictionary of parameters
    mvn_params_dict = mvn_params_paragami.random()
    mvn_params_dict['mean'] = mean
    mvn_params_dict['info'] = info

    # vector of parameters
    param_vec = mvn_params_paragami.flatten(mvn_params_dict, free = True)
    nat_vec = get_nat_vec(param_vec, mvn_params_paragami, mvn_nat_params_paragami)

    fishers_info = get_log_part_hess(nat_vec, mvn_nat_params_paragami)

    jac_term = get_jac_term(param_vec, mvn_params_paragami, mvn_nat_params_paragami)

    return np.dot(jac_term.transpose(), np.dot(fishers_info, jac_term))
