import logging

import numpy as np
from sklearn.preprocessing import normalize
from scipy.sparse import csr_matrix

logger = logging.getLogger(__name__)


def robustnorm(arr, qr=(0.1, 0.9)):
    """ Normalize the array so each column is between (-1, 1) in the specified quantile range (qr). 
    Parameters
    ----------
    arr: 2d array
        array to be normalized
    qr: tuple
        quantile range to be normalized to (0, 1)
    """
    ql = np.quantile(arr, qr[0], axis=0)
    qu = np.quantile(arr, qr[1], axis=0)

    arr_trans = (arr - ql) / (qu - ql) * 2 - 1

    sparse_genes_colids = (ql == qu)

    if np.any(sparse_genes_colids):
        arr_trans[:, sparse_genes_colids] = arr[:, sparse_genes_colids] - qu[sparse_genes_colids]

    return arr_trans, ql, qu


def inverse_robustnorm(arr_trans, _range):
    """ Inverse operation to :py:func:`robustnorm`.
    """
    ql, qu = _range
    arr = 0.5 * (arr_trans + 1) * (qu - ql) + ql
    return arr


def row_normalize(arr):
    """
    Each row sums to 1
    """
    arr_normalized = normalize(arr, norm='l1', axis=1)
    return arr_normalized


def col_normalize(arr):
    """
    Each col sums to 1
    """
    arr_normalized = normalize(arr, norm='l1', axis=0)
    return arr_normalized


def row_normalize_sparse(csr):
    """
    Each row sums to 1
    """
    csr_normalized = normalize(csr, norm='l1', axis=1)
    return csr_normalized


def col_normalize_sparse(csr):
    """
    Each col sums to 1
    """
    csr_normalized = normalize(csr, norm='l1', axis=0)
    return csr_normalized


def is_symmetric_csr_matrix(A):
    D = len((A - A.T).data)
    return not bool(D)


def get_sparsity(x):
    return 1 - x.size/np.multiply(*x.shape)


def mask_large_sparse_matrix(sm, mask):
    """Mask a sparse matrix by setting the values of the masked elements to 0.
    
    Parameters
    ----------
    sm: scipy.sparse.csr_matrix
        sparse matrix to be masked
    mask: 1d array
        boolean array to mask the sparse matrix [indices to remove]

    Returns
    -------
    sm: scipy.sparse.csr_matrix
        masked sparse matrix
    """

    rows = sm.nonzero()[0][mask]
    cols = sm.nonzero()[1][mask]
    sm[rows, cols] = 0
    sm.eliminate_zeros()
    return sm


def arr_to_csr(arr, dtype=None):
    """Convert a numpy array to a csr_matrix.

    Parameters
    ----------
    arr: numpy array
        numpy array to be converted

    Returns
    -------
    csr_arr: csr_matrix
        converted csr_matrix
    """

    csr_arr = csr_matrix(arr, dtype=dtype)
    csr_arr.eliminate_zeros()
    return csr_arr  


def var_cos(pred, tru):
    """ Compute cosine similarity between vars(genes) in pred and tru

    Parameters
    ----------
    pred: 2d array
        predicted gene expression matrix
    tru: 2d array
        true gene expression matrix
    
    Returns
    -------
    1d array
        cosine similarity between vars(genes) in pred and tru
    """
    p_norm = normalize(pred, norm="l2", axis=0)
    t_norm = normalize(tru, norm="l2", axis=0)
    return (p_norm * t_norm).sum(axis=0)


def sparse_elementwise_divide_nonzero(a, b):
    """
    Parameters
    ---------
    a: csr_matrix 
    b: csr_matrix 

    Returns
    -------
    a/b elementwise division (ignore 0 elements in b)
    csr_matrix
    """
    inv_b = b.copy()
    inv_b.data = 1 / inv_b.data
    return a.multiply(inv_b)


def flatten_nested_lists_to_array(L):
    return np.array(list(set([j for i in L for j in i])))


def v1_v2(v1, v2):
    """ Get the unique elements in v1 but not in v2, preserving the order of the elements in v1.
    Parameters
    ----------
    v1: 1d numpy array
    v2: 1d numpy array

    Returns
    -------
    1d numpy array
    """
    return v1[~np.in1d(v1, v2)]
