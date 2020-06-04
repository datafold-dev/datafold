import sys
from typing import Callable, Dict, Optional, Tuple, Union

import numpy as np
import scipy.sparse
import scipy.sparse.linalg

from datafold.utils.general import is_symmetric_matrix, sort_eigenpairs


class NumericalMathError(Exception):
    """Use for numerical problems/issues, such as singular matrices or too large
    imaginary part."""

    def __init__(self, message):
        super(NumericalMathError, self).__init__(message)


def scipy_eigsolver(
    kernel_matrix: Union[np.ndarray, scipy.sparse.csr_matrix],
    n_eigenpairs: int,
    is_symmetric: bool,
    is_stochastic: bool,
):
    """Compute eigenpairs of kernel matrix with scipy backend.

    The scipy solver is selected based on the number of eigenpairs to compute. Note
    that also for dense matrix cases a sparse solver is selected. There are two reasons
    for this decsision:

    1. General dense matrix eigensolver only allow *all* eigenpairs to be computed. This
       is computational more costly than handling a dense matrix to a sparse solver
       which can also solve for `k` eigenvectors.
    2. The hermitian (symmetric) `eigh` solver would also allow a partial computation of
       eigenpairs, but it showed to be slower in microbenchmark tests than the sparse
       solvers for dense matrices.

    Internal selection of backend:

    * If :code:`n_eigenpairs == n_samples` (for dense / sparse):
      * symmetric `eigh <https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.eigh.html#scipy.linalg.eigh>`_
      * non-symmetric `eig  <https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.eig.html#scipy.linalg.eig>`_

    * If :code:`n_eigenpairs < n_samples` (for dense / sparse):
      * symmetric `eigsh <https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.eigsh.html>`_
      * non-symmetric `eigs <https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.eigs.html#scipy.sparse.linalg.eigs>`_

    Parameters
    ----------

    kernel_matrix
        Matrix of shape `(n_samples, n_samples)`.

    n_eigenpairs
        Number of eigenpairs to compute.

    is_symmetric
        True if matrix is symmetric. Note that there is no check and also numerical
        noise that breaks the symmetry can lead to instabilities.

     is_stochastic
        If True, the kernel matrix is assumed to be row-stochastic. This enables
        setting a `sigma` close to 1 to accelerate convergence.

    Returns
    -------
    numpy.ndarray
        eigenvalues of shape `(n_eigenpairs,)`

    numpy.ndarray
        eigenvectors of shape `(n_samples, n_eigenpairs)`
    """

    n_samples, n_features = kernel_matrix.shape

    # check only for n_eigenpairs == n_features and n_eigenpairs < n_features
    # wrong parametrized n_eigenpairs are catched in scipy functions
    if n_eigenpairs == n_features:
        if is_symmetric:
            scipy_eigvec_solver = scipy.linalg.eigh
        else:
            scipy_eigvec_solver = scipy.linalg.eig

        solver_kwargs: Dict[str, object] = {
            "check_finite": False
        }  # should be already checked

    else:  # n_eigenpairs < matrix.shape[1]
        if is_symmetric:
            scipy_eigvec_solver = scipy.sparse.linalg.eigsh
        else:
            scipy_eigvec_solver = scipy.sparse.linalg.eigs

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
    kernel_matrix: Union[np.ndarray, scipy.sparse.csr_matrix],
    n_eigenpairs: int,
    is_symmetric: bool = False,
    is_stochastic: bool = False,
    normalize_eigenvectors: bool = False,
    backend: str = "scipy",
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute eigenvalues and -vectors from kernel matrix with consideration of matrix
    properties.

    Parameters
    ----------
    kernel_matrix
        Kernel matrix of shape `(n_samples, n_samples)`.

    n_eigenpairs
        Number of eigenpairs to compute.

    is_symmetric
        If True, this allows using specialized algorithms for symmetric matrices and
        enables an additional numerical sanity check that all eigenvalues are real-valued.

    is_stochastic
        If True, this allows convergence to be improved because the trivial first
        eigenvalue is known and all following eigenvalues are smaller.

    normalize_eigenvectors
        If True, all eigenvectors are normalized to Eucledian norm 1.

    backend
        Valid backends:
            * "scipy"

    Returns
    -------
    numpy.ndarray
        Eigenvalues in ascending order (absolute value).

    numpy.ndarray
        Eigenvectors (not necessarily normalized) in the same order to eigenvalues.
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

    assert not is_symmetric or (is_symmetric and is_symmetric_matrix(kernel_matrix))

    # BEGIN experimental code
    # test_sparsify_experimental = False
    # if test_sparsify_experimental:
    #
    #     SPARSIFY_CUTOFF = 1e-14
    #
    #     if scipy.sparse.issparse(kernel_matrix):
    #         kernel_matrix.data[np.abs(kernel_matrix.data) < SPARSIFY_CUTOFF] = 0
    #         kernel_matrix.eliminate_zeros()
    #     else:
    #         kernel_matrix[np.abs(kernel_matrix) < SPARSIFY_CUTOFF] = 0
    #         kernel_matrix = scipy.sparse.csr_matrix(kernel_matrix)
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
            "eigenvalues or eigenvectors contain 'NaN' or 'inf' values."
        )

    if is_symmetric:
        if np.any(eigvals.imag > 1e2 * sys.float_info.epsilon):
            raise NumericalMathError(
                "Eigenvalues have non-negligible imaginary part (larger than "
                f"{1e2 * sys.float_info.epsilon})."
            )

        # algorithm can include numerical noise in imaginary part
        eigvals = np.real(eigvals)
        eigvects = np.real(eigvects)

    if normalize_eigenvectors:
        eigvects /= np.linalg.norm(eigvects, axis=0)[np.newaxis, :]

    return sort_eigenpairs(eigvals, eigvects)
