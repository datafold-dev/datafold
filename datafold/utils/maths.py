from typing import Union

import numpy as np
import scipy.sparse


def sort_eigenpairs(
    eigenvalues, eigenvectors, ascending=False, eigenvector_orientation="column"
):
    eigenvector_orientation = eigenvector_orientation.lower()

    if eigenvector_orientation not in ["row", "column"]:
        raise ValueError(f"Invalid eigenvector orientation '{eigenvector_orientation}'")

    if eigenvalues.ndim != 1 or eigenvectors.ndim != 2:
        raise ValueError("eigenvalues has to be 1-dim. and eigenvectors 2-dim.")

    # Sort eigenvectors accordingly:
    idx = np.abs(eigenvalues).argsort()
    if not ascending:
        idx = idx[::-1]

    eigenvalues = eigenvalues[idx]

    if eigenvector_orientation == "column":
        eigenvectors = eigenvectors[:, idx]
    else:
        eigenvectors = eigenvectors[idx, :]

    return eigenvalues, eigenvectors


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
