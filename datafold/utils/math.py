
import numpy as np


def sort_eigenpairs(eigenvalues, eigenvectors, ascending=False, eigenvector_orientation="column"):
    eigenvector_orientation = eigenvector_orientation.lower()

    if eigenvector_orientation not in ["row", "column"]:
        raise ValueError(f"Invalid eigenvector orientation '{eigenvector_orientation}'")

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


def mat_dot_diagmat(matrix, diag_elements):
    """Fast element-wise and row-wise multiplication. This solves 'matrix @ np.diag(diag_elements)' much more
    efficiently.
    """
    assert diag_elements.ndim == 1 and matrix.ndim == 2
    return diag_elements * matrix


def diagmat_dot_mat(diag_elements, matrix):
    """Fast element-wise and column-wise multiplication. This solves 'np.diag(diag_elements) @ matrix' much more
    efficiently.
    """
    assert diag_elements.ndim == 1 and matrix.ndim == 2
    return matrix * diag_elements[:, np.newaxis]
