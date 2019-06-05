import autograd
from autograd import numpy as np

# import numpy as onp
import scipy as sp

def get_hess_inv_sqrt(block_hessian, ev_min=1.0, ev_max=1e5):
    # Get the matrix square root.  Sparse linear algebra doesn't seem to
    # be necessary.
    hessian_sym = 0.5 * (block_hessian + block_hessian.T)
    eig_val, eig_vec = np.linalg.eigh(hessian_sym)

    #ev_thresh = 1 + np.abs(np.max(eig_val)) * ev_tol
    #eig_val[eig_val <= ev_thresh] = ev_thresh
    eig_val[eig_val <= ev_min] = ev_min
    eig_val[eig_val >= ev_max] = ev_max

    hess_corrected = np.matmul(eig_vec,
                               np.matmul(np.diag(eig_val), eig_vec.T))

    hess_inv_sqrt = \
        np.matmul(eig_vec, np.matmul(np.diag(1 / np.sqrt(eig_val)), eig_vec.T))
    return np.array(hess_inv_sqrt), np.array(hess_corrected)
