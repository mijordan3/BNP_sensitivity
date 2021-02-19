# copied from https://github.com/rgiordan/AISTATS2019SwissArmyIJ/blob/master/genomics/aistats2019_ij_paper/regression_lib.py

# replaced autograd.numpy with numpy
import numpy as np


def run_regressions(y, regressors, w=None):
    """Get the optimal regression lines in closed form.
    Parameters
    ----------------
    y : `numpy.ndarray` (N, M)
        A matrix containing the outcomes of ``N`` regressions with
        ``M`` observations each.
    regressors : `numpy.ndarray` (M, D)
        A matrix of regressors.  The regression coefficient will be
        a ``D``-length vector.
    w : `numpy.ndarray` (M,), optional
        A vector of weights on the columns of ``y``.  If not set,
        the vector of ones is used.
    Returns
    ---------
    beta : `numpy.ndarray` (N, D)
        An array of the ``N`` regression coefficients.
    beta_infos : `numpy.ndarray` (N, D, D)
        An array of the "information" matrices, i.e. the inverse
        covariance matrices, of ``beta``.
    y_infos : `numpy.ndarray` (N,)
        An array of the inverse residual variances for each regression.
    """
    if w is None:
        w = np.ones(y.shape[1])
    assert y.shape[1] == regressors.shape[0]
    num_obs = y.shape[0]
    x_obs_dim = regressors.shape[1]
    y_obs_dim = y.shape[1]
    assert y_obs_dim > x_obs_dim

    beta = np.full((num_obs, x_obs_dim), float('nan'))
    beta_infos = np.full((num_obs, x_obs_dim, x_obs_dim), float('nan'))
    y_infos = np.full(num_obs, float('nan'))

    rtr = np.matmul(regressors.T, w[:, np.newaxis] * regressors)
    evs = np.linalg.eigvals(rtr)
    if np.min(evs < 1e-6):
        raise ValueError('Regressors are approximately singular.')
    rtr_inv_rt = np.linalg.solve(rtr, regressors.T)
    for n in range(num_obs):
        beta_reg = np.matmul(rtr_inv_rt, w * y[n, :])
        beta[n, :] = beta_reg
        resid = y[n, :] - np.matmul(regressors, beta_reg)

        # The extra -x_obs_dim comes from the beta variance -- see notes.
        #y_info = (y_obs_dim - x_obs_dim) / np.sum(resid ** 2)
        y_info = (sum(w) - x_obs_dim) / np.sum(w * (resid ** 2))

        beta_infos[n, :, :] = rtr * y_info
        y_infos[n] = y_info

    return beta, beta_infos, y_infos

