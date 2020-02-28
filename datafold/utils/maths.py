from typing import Tuple, Union

import numpy as np
import scipy.sparse


def sort_eigenpairs(eigenvalues, eigenvectors, ascending=False):
    """
    eigenvectors are assumed to be column wise
    """
    if eigenvalues.ndim != 1 or eigenvectors.ndim != 2:
        raise ValueError(
            "eigenvalues have to be 1-dim and eigenvectors "
            "2-dim np.ndarray respectively"
        )

    # Sort eigenvectors according to absolute value of eigenvalue:
    idx = np.abs(eigenvalues).argsort()

    if not ascending:
        # creates a view on array and is most efficient way for reversing order
        # see: https://stackoverflow.com/q/6771428
        idx = idx[::-1]

    return eigenvalues[idx], eigenvectors[:, idx]


def mat_dot_diagmat(matrix, diag_elements, out=None):
    """Fast element-wise and row-wise multiplication.
    This computes :code:`matrix @ np.diag(diag_elements)` much more efficiently.
    """
    assert diag_elements.ndim == 1 and matrix.ndim == 2
    return np.multiply(diag_elements, matrix, out=out)


def diagmat_dot_mat(diag_elements, matrix, out=None):
    """Fast element-wise and column-wise multiplication.
    This computes :code:'np.diag(diag_elements) @ matrix' much more efficiently.
    """
    assert diag_elements.ndim == 1 and matrix.ndim == 2
    return np.multiply(matrix, diag_elements[:, np.newaxis], out=out)


def is_symmetric_matrix(matrix: np.ndarray, tol=0):
    if matrix.ndim != 2:
        raise ValueError("matrix has to be 2-dim.")
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("matrix has to be square")

    max_abs_deviation = np.max(np.abs(matrix - matrix.T))
    return max_abs_deviation <= tol


def remove_numeric_noise_symmetric_matrix(
    matrix: Union[np.ndarray, scipy.sparse.spmatrix]
) -> Union[np.ndarray, scipy.sparse.spmatrix]:
    """
    Removes numerical noise from a symmetric matrix, that can get sometimes can get
    introduced for certain operations. Matrices that should be perfectly symmetric they
    show differences like: :code:`np.max(np.abs(matrix - matrix.T)) #
    1.1102230246251565e-16`

    Parameters
    ----------
    matrix
        matrix to remove numerical noise from
    Returns
    -------
    Union[np.ndarray, scipy.sparse.spmatrix]
        same matrix perfectly symmetric
    """

    # A faster way would be to truncate the floating values of the matrix to a certain
    # precision, but numpy does not seem to provide anything for it?

    if scipy.sparse.issparse(matrix):
        matrix = (matrix + matrix.T) / 2
    else:
        matrix = np.add(matrix, matrix.T, out=matrix)
        matrix = np.divide(matrix, 2, out=matrix)

    return matrix


def random_subsample(data: np.ndarray, n_samples: int) -> Tuple[np.ndarray, np.ndarray]:
    """Randomly sample a subset of a data set while preserving order.

    The sampling is done without replacement.

    Parameters
    ----------
    data : np.ndarray
        Array whose 0-th axis indexes the data points.
    n_samples : int
        Number of items to randomly (uniformly) sample from the data.  This is typically
        less than the total number of elements in the data set.

    Returns
    -------
    sampled_data : np.ndarray
        A total of `num_samples` uniformly randomly sampled data points from `data`.
    """

    from sklearn.utils.validation import check_scalar, check_array

    data = check_array(data, force_all_finite=False, ensure_min_samples=2,)
    assert isinstance(data, np.ndarray)

    n_samples_data = data.shape[0]

    check_scalar(
        n_samples,
        name="n_samples",
        target_type=(int, np.integer),
        min_val=1,
        max_val=n_samples_data - 1,
    )

    indices = np.random.permutation(n_samples_data)[:n_samples]
    return data[indices, :], indices
