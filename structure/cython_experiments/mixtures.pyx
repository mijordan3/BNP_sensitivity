#from cython.view cimport array
from cpython.array cimport array, clone

import numpy as np
from libc.math cimport exp, log

print("Hello World")


def square(float x):
    return x ** 2


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


# f\left(a,g,\alpha,\beta,p\right) =
#   \sum_{n}\sum_{k}g_{nk}\left(\alpha+\beta a_{nk}\right)m_{nk}^{p}
def mixture_sum(double[:,:] a):
    cdef int n, k

    cdef int n_num = a.shape[0]
    cdef int k_num = a.shape[1]

    # Allocate memory for e_z.  See this StackOverflow post:
    # https://tinyurl.com/y9x8dv8s
    cdef array[double] arr, template = array('d')
    e_z = clone(template, k_num, False)

    cdef float total = 0
    for n in range(n_num):
        get_row_e_z(a[n, :], e_z)
        print(e_z)
        for k in range(k_num):
            total += e_z[k] * a[n, k]

    return total