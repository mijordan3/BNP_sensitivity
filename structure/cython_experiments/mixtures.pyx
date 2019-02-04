#from cython.view cimport array
from cpython.array cimport array, clone

import numpy as np
from libc.math cimport exp, log, pow


def myprod(double x, double y):
    return x * pow(y, 2)


def mypow(float x, int p):
    return pow(x, p)


def square(float x):
    return mypow(x, 2)


# e_z is overwritten.
def get_row_e_z(double[:] a_row, double[:] e_z):
    cdef int k;
    cdef int num_k = a_row.shape[0];
    assert num_k > 1;

    # Subtract off the maximum to avoid overflows.
    cdef float row_max = a_row[0];
    for k in range(1, num_k):
        if a_row[k] > row_max:
            row_max = a_row[k];

    # Calcualate the log normalizing constant.
    cdef float log_norm = 0;
    for k in range(num_k):
        # This value of e_z is a placeholder to avoid recalcualting exp.
        e_z[k] = a_row[k] - row_max
        log_norm += exp(e_z[k]);

    log_norm = log(log_norm)
    for k in range(num_k):
        e_z[k] = exp(e_z[k] - log_norm);

    return e_z;


# Get e_z to a power in place.
def get_e_z(double[:, :] a, double[:, :] e_z, int p):
    cdef int n_num = a.shape[0]
    cdef int k_num = a.shape[1]
    assert n_num == e_z.shape[0]
    assert k_num == e_z.shape[1]

    # Allocate memory for e_z.  See this StackOverflow post:
    # https://tinyurl.com/y9x8dv8s
    cdef array[double] arr, template = array('d')
    e_z_row = clone(template, k_num, False)

    for n in range(n_num):
        get_row_e_z(a[n, :], e_z_row)
        for k in range(k_num):
            e_z[n, k] = pow(e_z_row[k], p)



# All JVPs of the indicator sum can be expressed as sums of this function.
#
# f\left(a,g,\alpha,\beta,p\right) =
#   \sum_{n}\sum_{k}g_{nk}\left(\alpha+\beta a_{nk}\right)m_{nk}^{p}
#
# If g.shape == (0, 0), then it is not used (i.e., it is taken to be
# identically 1).
def get_mixture_sum(double[:,:] a,
                    double[:,:] g,
                    double alpha,
                    double beta,
                    int p,
                    double[:,:] result):

    cdef int n_num = a.shape[0]
    cdef int k_num = a.shape[1]

    # Check whether g is identically one
    cdef bint use_g;
    if g.shape[0] == 0:
        assert g.shape[1] == 0
        use_g = False
    else:
        assert n_num == g.shape[0]
        assert k_num == g.shape[1]
        use_g = True

    # Check whether to save the matrix of results
    cdef bint save_result;
    if result.shape[0] == 0:
        assert result.shape[1] == 0
        save_result = False
    else:
        assert n_num == result.shape[0]
        assert k_num == result.shape[1]
        save_result = True

    # Allocate memory for e_z.  See this StackOverflow post:
    # https://tinyurl.com/y9x8dv8s
    cdef array[double] arr, template = array('d')
    e_z = clone(template, k_num, False)

    cdef double total = 0
    cdef int n
    cdef double g_n_k = 1;
    cdef double e_z_k_p;
    cdef double term;
    for n in range(n_num):
        get_row_e_z(a[n, :], e_z)
        for k in range(k_num):
            if use_g:
                g_n_k = g[n, k]
            if p == 1:
                e_z_k_p = e_z[k]
            else:
                e_z_k_p = pow(e_z[k], p)

            term = e_z_k_p * g_n_k * (alpha + beta * a[n, k])
            if save_result:
                result[n, k] = term
            total += term

    return total
