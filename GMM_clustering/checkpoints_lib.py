# Utilites to save and load analysis checkpoints.

from datetime import datetime

import numpy as np
from scipy import sparse

import json_tricks  # https://github.com/mverleg/pyjson_tricks

sp_string = '_sp_packed'
np_string = '_np_packed'


def get_timestamp():
    return datetime.today().timestamp()


# Pack a sparse csr_matrix in a json-seralizable format.
def pack_csr_matrix(sp_mat):
    assert sparse.isspmatrix_csr(sp_mat)
    sp_mat = sparse.csr_matrix(sp_mat)
    return { 'data': json_tricks.dumps(sp_mat.data),
             'indices': json_tricks.dumps(sp_mat.indices),
             'indptr': json_tricks.dumps(sp_mat.indptr),
             'shape': sp_mat.shape,
             'type': 'csr_matrix' }


# Convert the output of pack_csr_matrix back into a csr_matrix.
def unpack_csr_matrix(sp_mat_dict):
    assert sp_mat_dict['type'] == 'csr_matrix'
    data = json_tricks.loads(sp_mat_dict['data'])
    indices = json_tricks.loads(sp_mat_dict['indices'])
    indptr = json_tricks.loads(sp_mat_dict['indptr'])
    return sparse.csr_matrix(
        ( data, indices, indptr), shape = sp_mat_dict['shape'])


# Populate a dictionary with the minimum necessary fields to describe
# a pre-processing step.
def get_preprocessing_dict(method, p_values=None, log_fold_change=None):
    """
    Returns a pre-processing dict, population with defaults.

    Parameters
    ----------
    method : string

    p_values : ndarray, optional, default: None

    log_fold_change : ndarray, optional, default: None

    Returns
    -------
    dictionary
    """

    if p_values is None:
        p_values = np.array([])

    if log_fold_change is None:
        log_fold_change = np.array([])

    return {
        'timestamp': get_timestamp(),
        'method': method,
        'p_values' + np_string: json_tricks.dumps(p_values),
        'log_fold_change' + sp_string: json_tricks.dumps(log_fold_change)}


def get_fit_dict(method, initialization_method, seed, centroids,
                 labels=None,
                 cluster_assignments=None,
                 preprocessing_dict=None, basis_mat=None,
                 cluster_weights=None):
    """
    returns a "fit" dictionary

    Parameters
    ----------
    method : string

    initialization_method : string

    seed : int
        random seed

    centroids : ndarray

    labels : ndarray (n, ), optional, default: None
        1D-array containing the cluster labels.
        Either labels or cluster_assignments needs to be provided.

    cluster_assignments : {ndarray, csr_matrix}, optional, default: None
        n by k sparse or dense matrix contraining the cluster assignments or
        probability.
        Either labels or cluster_assignments needs to be provided.

    preprocessing_dict : dictionary
        dictionary containing a `timestamp` and `method` key

    basis_mat : ndarray, optional, default: None

    cluster_weights : ndarray, optional, default: None
    """
    if labels is None and cluster_assignments is None:
        raise ValueError(
            "In order to save the results, either provide labels or cluster"
            " assignment")

    if labels is not None and len(labels.shape) > 1:
        raise ValueError(
            "Labels should be a 1D array. "
            "Provided a %d-d array" % len(labels.shape))

    if cluster_assignments is not None and not sparse.issparse(cluster_assignments):
        cluster_assignments = sparse.csr_matrix(cluster_assignments)

    if labels is not None:
        if labels.max() >= centroids.shape[1]:
            raise ValueError(
                "There are %d centroids, but the labels contain up to %d "
                "element" % (centroids.shape[1], labels.max()))

        # The user provided labels and not a sparse matrix
        if cluster_assignments is not None:
            if np.any(labels != cluster_assignments.argmax(axis=1).A.flatten()):
                raise ValueError(
                    "Incoherence between the labels provided and the cluster "
                    "assignments.")
        else:
            cluster_assignments = sparse.csr_matrix(
                (np.ones(len(labels)), (np.arange(len(labels)), labels)),
                shape=(len(labels), centroids.shape[1]))

    if preprocessing_dict is None:
        preprocessing_dict = get_preprocessing_dict("NoPreprocessing")

    if basis_mat is None:
        basis_mat = np.array([])

    if cluster_weights is None:
        cluster_weights = cluster_assignments.sum(axis=0).A.flatten()
        cluster_weights /= cluster_weights.sum()

    return {
        'timestamp': get_timestamp(),
        'method': method,
        'initialization_method': initialization_method,
        'seed': seed,
        'preprocessing_method': preprocessing_dict['method'],
        'preprocessing_timestamp': preprocessing_dict['timestamp'],
        'centroids' + np_string: json_tricks.dumps(centroids),
        'basis_mat' + np_string: json_tricks.dumps(basis_mat),
        'cluster_weights' + np_string: json_tricks.dumps(cluster_weights),
        'cluster_assignments' + sp_string: pack_csr_matrix(cluster_assignments)
         }
