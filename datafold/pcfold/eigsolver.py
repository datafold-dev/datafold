import sys
from typing import Callable, Optional, Union

import numpy as np
import scipy.sparse
import scipy.sparse.linalg

from datafold.utils.general import is_symmetric_matrix, sort_eigenpairs


class NumericalMathError(Exception):
    """Use for numerical problems/issues, such as singular matrices or too large
    imaginary part."""

    def __init__(self, message):
        super(NumericalMathError, self).__init__(message)


def scipy_eigsolver(kernel_matrix, n_eigenpairs, is_symmetric, is_stochastic):
    """Selects the parametrization for scipy eigensolver depending on the
    properties of the kernel.


    """

    n_samples, n_features = kernel_matrix.shape

    # check only for n_eigenpairs == n_features and n_eigenpairs < n_features
    # wrong parametrized n_eigenpairs are catched in scipy functions
    if n_eigenpairs == n_features:
        if is_symmetric:
            scipy_eigvec_solver = scipy.linalg.eigh
        else:
            scipy_eigvec_solver = scipy.linalg.eig

        # symmetric case:
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.eigh.html#scipy.linalg.eigh
        # non-symmetric case:
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.eig.html#scipy.linalg.eig
        solver_kwargs = {"check_finite": False}  # should be already checked

    else:  # n_eigenpairs < matrix.shape[1]
        if is_symmetric:
            scipy_eigvec_solver = scipy.sparse.linalg.eigsh
        else:
            scipy_eigvec_solver = scipy.sparse.linalg.eigs

        # symmetric case:
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.eigsh.html
        # non-symmetric case:
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.eigs.html#scipy.sparse.linalg.eigs
        solver_kwargs = {
            "k": n_eigenpairs,
            "which": "LM",
            "v0": np.ones(n_samples),
            "tol": 1e-14,
        }

        # The selection of sigma is a result of the microbenchmark_kernel_eigvect.py
        # which checks solution quality and convergence speed.
        if is_symmetric and is_stochastic:
            # NOTE: it turned out that for self.kernel_.is_symmetric=False (-> eigs),
            # setting sigma=1 resulted into a slower computation.
            NUMERICAL_EXACT_BREAKER = 0.1
            solver_kwargs["sigma"] = 1.0 + NUMERICAL_EXACT_BREAKER
            solver_kwargs["mode"] = "normal"
        else:
            solver_kwargs["sigma"] = None

    eigvals, eigvects = scipy_eigvec_solver(kernel_matrix, **solver_kwargs)

    return eigvals, eigvects


_valid_backends = ["scipy"]


def compute_kernel_eigenpairs(
    kernel_matrix: Union[np.ndarray, scipy.sparse.spmatrix],
    n_eigenpairs: int,
    is_symmetric: bool = False,
    is_stochastic: bool = False,
    backend: str = "scipy",
):
    """Computing eigenpairs of kernel matrices by exploiting.

    Parameters
    ----------
    kernel_matrix
        Two dimensional square matrix, dense or sparse.

    n_eigenpairs
        Number of eigenpairs to compute.

    is_symmetric
        If True this allows for specialized algorithms exploiting symmetry and enables
        an additional check if all eigenvalues are real valued.

    is_stochastic
        If True this allows to improve convergence because the trivial first eigenvalue
        is known.

    backend
        * "scipy" - selects between\
           `scipy.eigs <https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.eigs.html#scipy.sparse.linalg.eigs>`\
            and\
          `scipy.eigsh <https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.eigsh.html>`

    Returns
    -------
    Tuple[numpy.ndarray, numpy.ndarray]
        Eigenpairs sorted by largest (magnitude) eigenvalue, decreasing. The
        eigenvectors are not necessarily normalized.
    """

    if kernel_matrix.ndim != 2 or kernel_matrix.shape[0] != kernel_matrix.shape[1]:
        raise ValueError(
            f"kernel matrix must be a square. "
            f"Got kernel_matrix.shape={kernel_matrix.shape}"
        )

    if (
        isinstance(kernel_matrix, scipy.sparse.spmatrix)
        and not np.isfinite(kernel_matrix.data).all()
    ) or (
        isinstance(kernel_matrix, np.ndarray) and not np.isfinite(kernel_matrix).all()
    ):
        raise ValueError(
            "kernel_matrix must only contain finite values (no np.nan or np.inf)"
        )

    if is_symmetric and not is_symmetric_matrix(kernel_matrix):
        raise ValueError("matrix is not symmetric")

    # BEGIN experimental code
    test_sparsify_experimental = False
    if test_sparsify_experimental:

        SPARSIFY_CUTOFF = 1e-14

        if scipy.sparse.issparse(kernel_matrix):
            kernel_matrix.data[np.abs(kernel_matrix.data) < SPARSIFY_CUTOFF] = 0
            kernel_matrix.eliminate_zeros()
        else:
            kernel_matrix[np.abs(kernel_matrix) < SPARSIFY_CUTOFF] = 0
            kernel_matrix = scipy.sparse.csr_matrix(kernel_matrix)
    # END experimental

    if backend == "scipy":
        eigvals, eigvects = scipy_eigsolver(
            kernel_matrix=kernel_matrix,
            n_eigenpairs=n_eigenpairs,
            is_symmetric=is_symmetric,
            is_stochastic=is_stochastic,
        )
    else:
        raise ValueError(f"backend {backend} not known.")

    if not np.isfinite(eigvals).all() or not np.isfinite(eigvects).all():
        raise NumericalMathError(
            "eigenvalues or eigenvector contain 'NaN' or 'inf' values."
        )

    if is_symmetric:
        if np.any(eigvals.imag > 1e2 * sys.float_info.epsilon):
            raise NumericalMathError(
                "Eigenvalues have non-negligible imaginary part (larger than "
                f"{1e2 * sys.float_info.epsilon})."
            )

        # can include zero or numerical noise imaginary part
        eigvals = np.real(eigvals)
        eigvects = np.real(eigvects)

    return sort_eigenpairs(eigvals, eigvects)
