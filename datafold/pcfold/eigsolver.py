import sys
from typing import Callable, Optional

import numpy as np
import scipy

from datafold.utils.maths import is_symmetric_matrix, sort_eigenpairs

try:
    from datafold.pcfold._gpu_eigensolver import eigensolver as gpu_eigsolve
except ImportError:
    gpu_eigsolve: Optional[Callable] = None  # type: ignore

    # variable is used to warn user when requesting GPU eigensolver
    SUCCESS_GPU_IMPORT = False
else:
    SUCCESS_GPU_IMPORT = True


class NumericalMathError(Exception):
    """Use for numerical problems/issues, such as singular matrices or too large
    imaginary part."""

    def __init__(self, message):
        super(NumericalMathError, self).__init__(message)


def scipy_eigsolver(matrix, n_eigenpairs, is_symmetric, is_stochastic):
    """Selects the parametrization for scipy eigensolver depending on the
    properties of the kernel.


    """

    n_samples, n_features = matrix.shape

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

    eigvals, eigvects = scipy_eigvec_solver(matrix, **solver_kwargs)

    return eigvals, eigvects


VALID_BACKEND = ["scipy", "gpu"]


def compute_kernel_eigenpairs(
    matrix, n_eigenpairs=None, is_symmetric=False, is_stochastic=False, backend="scipy",
):
    """
    TODO: Docu note
       -- the eigenvectors are not necessarily normalized,
       -- the eigenvectors are sorted w.r.t. abs(eigenvalue), descending
    Parameters
    ----------
    matrix
    n_eigenpairs
    is_symmetric
    is_stochastic
    backend

    Returns
    -------

    """

    if matrix.dtype != np.dtype(np.float64):
        raise TypeError(
            "only real-valued floating points (double precision) are supported"
        )

    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(
            f"kernel matrix must be a square. Got matrix.shape={matrix.shape}"
        )

    if is_symmetric:
        assert is_symmetric_matrix(matrix)

    # BEGIN experimental
    test_sparsify = False
    if test_sparsify:

        SPARSIFY_CUTOFF = 1e-14

        if scipy.sparse.issparse(matrix):
            matrix.data[np.abs(matrix.data) < SPARSIFY_CUTOFF] = 0
            matrix.eliminate_zeros()
        else:
            matrix[np.abs(matrix) < SPARSIFY_CUTOFF] = 0
            matrix = scipy.sparse.csr_matrix(matrix)
    # END experimental

    if backend == "scipy":
        eigvals, eigvects = scipy_eigsolver(
            matrix=matrix,
            n_eigenpairs=n_eigenpairs,
            is_symmetric=is_symmetric,
            is_stochastic=is_stochastic,
        )
    elif backend == "gpu":
        if not SUCCESS_GPU_IMPORT:
            raise ValueError(
                "backend 'gpu' not available because import failed "
                "(cusparse and/or cuda not available)"
            )
        eigvals, eigvects = gpu_eigsolve(
            matrix=matrix, num_eigenpairs=n_eigenpairs, sigma=None, initial_vector=None,
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
