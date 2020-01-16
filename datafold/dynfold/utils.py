"""Miscellaneous utilities.

"""

from typing import Union

import numpy as np
import scipy.sparse
from scipy.sparse import coo_matrix, csr_matrix


def downsample(data: np.ndarray, num_samples: int) -> np.ndarray:
    """Randomly sample a subset of a data set while preserving order.

    The sampling is done without replacement.

    Parameters
    ----------
    data : np.ndarray
        Array whose 0-th axis indexes the data points.
    num_samples : int
        Number of items to randomly (uniformly) sample from the data.  This is typically
        less than the total number of elements in the data set.

    Returns
    -------
    sampled_data : np.ndarray
        A total of `num_samples` uniformly randomly sampled data points from `data`.
    """
    if 0 > num_samples >= data.shape[0]:
        raise ValueError(
            f"The parameter 'num_samples' has to be larger than zero and smaller or "
            f"equal to data.shape[0]={data.shape[0]}."
        )

    indices = np.random.permutation(data.shape[0])[:num_samples]
    return data[indices, :]


@DeprecationWarning
def coo_tocsr(matrix: scipy.sparse.coo_matrix) -> scipy.sparse.csr_matrix:
    """Convert matrix to Compressed Sparse Row format, fast.

    This function is derived from the corresponding SciPy code but it avoids the sanity
    checks that slow `scipy.sparse.coo_matrix.to_csr down`. In particular,
    by not summing duplicates we can attain important speed-ups for large matrices.
    """
    from scipy.sparse import csr_matrix

    if matrix.nnz == 0:
        return csr_matrix(matrix.shape, dtype=matrix.dtype)

    m, n = matrix.shape
    # Using 32-bit integer indices allows for matrices of up to 2,147,483,647
    # non-zero entries.
    idx_dtype = np.int32
    row = matrix.row.astype(idx_dtype, copy=True)
    col = matrix.col.astype(idx_dtype, copy=True)

    indptr = np.empty(n + 1, dtype=idx_dtype)
    indices = np.empty_like(row, dtype=idx_dtype)
    data = np.empty_like(matrix.data, dtype=matrix.dtype)

    scipy.sparse._sparsetools.coo_tocsr(
        m, n, matrix.nnz, row, col, matrix.data, indptr, indices, data
    )

    return csr_matrix((data, indices, indptr), shape=matrix.shape)


@DeprecationWarning
def get_row(matrix: Union[scipy.sparse.coo_matrix, np.ndarray], row: int) -> np.ndarray:
    """Get a row from a matrix as an array in row shape.

    Parameters
    ----------
    matrix : Union[scipy.sparse.coo_matrix, np.ndarray]
        A matrix.
    row : int
        The number of the row to return.

    """

    if isinstance(matrix, coo_matrix):
        return np.squeeze(matrix.getrow(row).toarray())
    elif isinstance(matrix, np.ndarray):
        return matrix[row, :]
    elif isinstance(matrix, csr_matrix):
        return np.squeeze(matrix[row, :].toarray())
    else:
        raise TypeError(f"Instance {type(matrix)} not supported")


def to_ndarray(matrix: Union[coo_matrix, csr_matrix, np.ndarray]) -> np.ndarray:
    """Convert a coo_matrix or an ndarray to an ndarray

    Parameters
    ----------
    matrix : Union[scipy.sparse.coo_matrix, np.ndarray]
        A matrix.

    """

    if isinstance(matrix, coo_matrix):
        return np.squeeze(matrix.toarray())
    elif isinstance(matrix, csr_matrix):
        return matrix.toarray()
    elif isinstance(matrix, np.ndarray):
        return matrix
    else:
        raise TypeError(f"Instance {type(matrix)} not supported")
