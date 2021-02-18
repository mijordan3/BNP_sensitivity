# copied over from 
# https://github.com/rgiordan/AISTATS2019SwissArmyIJ/blob/master/genomics/aistats2019_ij_paper/transform_regression_lib.py

import numpy as np
from copy import deepcopy
import paragami

############################################################
# Transform regression parameters by matrix multiplication

def make_matrix_full_row_rank(x, min_ev=1e-8):
    """Return a matrix with full row rank such that
    x.T @ x = new_x.T @ new_x
    """
    u, s, vh = np.linalg.svd(x, full_matrices=False)

    s_nonzero = np.argwhere(s > min_ev)[:,0]
    new_x = np.diag(s[s_nonzero]) @ vh[s_nonzero, :]
    return new_x


def multiply_regression_by_matrix(old_beta, old_beta_info, transform_mat):
    """Means and infos of regression parameters after matrix multiplication.
    
    """
    n_obs = old_beta.shape[0]

    beta_transformed = old_beta @ transform_mat.T

    beta_cov = np.array([
        np.linalg.inv(old_beta_info[n, :, :])
        for n in range(n_obs) ])
    new_beta_cov = np.einsum(
        'ia,nab,bj->nij', transform_mat, beta_cov, transform_mat.T)
    beta_info_transformed = np.array([
        np.linalg.inv(new_beta_cov[n, :, :])
        for n in range(n_obs) ])

    return beta_transformed, beta_info_transformed


def make_matrix_full_row_rank_with_unrotation(x, min_ev=1e-8):
    """Return a full row rank representation of x and an un-rotating matrix.

    Note that different systems may produce different matrices from
    the same input since the svd is not unique.

    Parameters
    -----------
    x: `numpy.ndarray` (N, M)
        A numeric matrix, with N >= M.

    Returns
    ----------
    new_x: `numpy.ndarray` (M, M)
        A full-rank matrix such that x.T @ x = new_x.T @ new_x
    new_x_unrotate: `numpy.ndarray` (N, M)
        A matrix such that, for any (M,) vector ``v``,
        ``x @ v == x_unrotate @ x_new @ v``.
    """
    u, s, vh = np.linalg.svd(x, full_matrices=False)
    s_nonzero = np.argwhere(s > min_ev)[:,0]
    new_x = np.diag(s[s_nonzero]) @ vh[s_nonzero, :]
    new_x_unrotate = u[:, s_nonzero]
    return new_x, new_x_unrotate


def get_reversible_predict_and_demean_matrix(x):
    """Map regressors to and from a de-meaned observation space.

    The matrix ``transform_mat`` maps a regression coefficient
    ``beta`` into a full-rank space where the L2 norm is equivalent
    to the L2 norm of a demeaned version of ``y``.

    That is, if
    ``pred_y = x @ beta - np.mean(x @ beta)`` and
    ``gamma = transform_mat @ beta``,
    then ``np.linalg.norm(gamma) == np.linalg.norm(pred_y)``.

    The matrix ``unrotate_mat`` undoes the transformation in the
    sense that
    ``unrotate_mat @ gamma = pred_y``.

    Parameters
    -----------
    x: `numpy.ndarray` (N, M)
        A ``regression_lib.Regressions`` object.

    Returns
    ---------
    transform_mat: `numpy.ndarray` (M, M)
        A matrix mapping to the full-rank demeaned observation space.
    unrotate_mat: `numpy.ndarray` (N, M)
        A matrix mapping back to the original space without the zero
        eigenvalues.

    """
    y_obs_dim = x.shape[0]
    beta_dim = x.shape[1]
    demean_mat = (np.eye(y_obs_dim) -
                  np.ones((y_obs_dim, y_obs_dim)) / y_obs_dim)

    base_transform_mat = demean_mat @ x
    transform_mat, unrotate_mat = \
        make_matrix_full_row_rank_with_unrotation(base_transform_mat)

    return transform_mat, unrotate_mat