import sys
from typing import Union

import numpy as np
import scipy.sparse
import scipy.sparse.linalg

from datafold.utils.general import is_matrix, is_symmetric_matrix, sort_eigenpairs

_valid_eigsolver_backends = ["scipy"]
_valid_svd_backends = ["scipy"]


class NumericalMathError(Exception):
    """Use for numerical problems/issues, such as singular matrices or too large
    imaginary part.
    """

    def __init__(self, message):
        super().__init__(message)


def scipy_eigsolver(
    kernel, kernel_matrix: Union[np.ndarray, scipy.sparse.csr_matrix], n_eigenpairs: int
):
    """Compute eigenpairs of kernel matrix with scipy backend.

    The scipy solver is selected based on the number of eigenpairs to compute. Note
    that also for dense matrix cases a sparse solver is selected. There are two reasons
    for this decision:

    1. General dense matrix eigensolver only allow *all* eigenpairs to be computed. This
       is computational more costly than handling a dense matrix to a sparse solver
       which can also solve for `k` eigenvectors.
    2. The hermitian (symmetric) `eigh` solver would also allow a partial computation of
       eigenpairs, but it showed to be slower in microbenchmark tests than the sparse
       solvers for dense matrices.

    Internal selection of backend:

    * If :code:`n_eigenpairs == n_samples` (for dense / sparse):

      * symmetric `eigh <https://docs.scipy.org/doc/scipy/reference/generated/scipy.
        linalg.eigh.html#scipy.linalg.eigh>`__
      * non-symmetric `eig <https://docs.scipy.org/doc/scipy/reference/generated/scipy.
        linalg.eig.html#scipy.linalg.eig>`__

    * If :code:`n_eigenpairs < n_samples` (for dense / sparse):

      * symmetric `eigsh <https://docs.scipy.org/doc/scipy/reference/generated/scipy.
        sparse.linalg.eigsh.html>`__
      * non-symmetric `eigs <https://docs.scipy.org/doc/scipy/reference/generated/scipy.
        sparse.linalg.eigs.html#scipy.sparse.linalg.eigs>`__

    Parameters
    ----------
    kernel_matrix
        Matrix of shape `(n_samples, n_samples)`.

    n_eigenpairs
        Number of eigenpairs to compute.

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
        if kernel._is_symmetric_kernel:
            scipy_eigvec_solver = scipy.linalg.eigh
        else:
            scipy_eigvec_solver = scipy.linalg.eig

        solver_kwargs: dict[str, object] = {
            "check_finite": False
        }  # should be checked already

    else:  # n_eigenpairs < matrix.shape[1]
        if kernel.is_symmetric:
            scipy_eigvec_solver = scipy.sparse.linalg.eigsh
        else:
            scipy_eigvec_solver = scipy.sparse.linalg.eigs

        solver_kwargs = {
            "k": n_eigenpairs,
            "which": "LM",
            "v0": np.ones(n_samples),
            "tol": 1e-14,
        }

        # The selection of sigma is a result of a microbenchmark
        if kernel.is_symmetric and kernel.is_stochastic:
            # NOTE: it turned out that for self.kernel_.is_symmetric=False (-> eigs),
            # setting sigma=1 resulted into a slower computation.
            NUMERICAL_EXACT_BREAKER = 0.1
            solver_kwargs["sigma"] = 1.0 + NUMERICAL_EXACT_BREAKER
            solver_kwargs["mode"] = "normal"
        else:
            # NOTE: it turned out that for self.kernel_.is_symmetric=False (-> eigs),
            # setting sigma=1 resulted into a slower computation.
            solver_kwargs["sigma"] = None

        # the scipy solvers only work on floating points
        if scipy.sparse.issparse(
            kernel_matrix
        ) and kernel_matrix.data.dtype.kind not in ["fdFD"]:
            kernel_matrix = kernel_matrix.asfptype()
        elif isinstance(kernel_matrix, np.ndarray) and kernel_matrix.dtype.kind != "f":
            kernel_matrix = kernel_matrix.astype(float)

    eigvals, eigvects = scipy_eigvec_solver(kernel_matrix, **solver_kwargs)

    return eigvals, eigvects


def scipy_svdsolver(
    kernel_matrix, n_svdvtriplets, **kwargs
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decompose a (possibly rectangular) kernel matrix into singular value components.

    Compute

    .. math::

        K = U S V^T

    where, :math:`K` is the kernel matrix, :math:`U` and :math:`V` are the left and right
    singular vectors and :math:`S` the diagonal singular matrix.

    Parameters
    ----------
    kernel_matrix
        Kernel matrix of shape `(n_samples_data, n_samples_reference)`, where
        `n_samples_reference` can be the size of a reference/landmark set.

    n_svdvtriplets
        The number of SVD vectors and values to compute. Must be smaller than
        `min(n_samples_data, n_samples_reference)`. If all triplets are computed, then a
        sparse kernel may be cast to a dense matrix.

    Returns
    -------
        svdvec_left
            matrix :math:`U`
        svdvals
            array of singular values :math:`S`
        svdvec_right
            matrix :math:`V`
    """
    max_n_triplets = np.min(kernel_matrix.shape)

    if n_svdvtriplets < max_n_triplets:
        random_state = kwargs.pop("random_state", 3)
        which = kwargs.pop("which", "LM")

        svdvec_left, svdvals, svdvec_right = scipy.sparse.linalg.svds(
            kernel_matrix,
            k=n_svdvtriplets,
            which=which,
            random_state=random_state,
            **kwargs,
        )
    else:  # n_svdvtriplets == max_n_triplets:
        if scipy.sparse.isspmatrix(kernel_matrix):
            # must be a dense matrix for the solver -- TODO: maybe raise warning?
            kernel_matrix = kernel_matrix.toarray()
        svdvec_left, svdvals, svdvec_right = scipy.linalg.svd(
            kernel_matrix, full_matrices=False
        )

    return svdvec_left, svdvals, svdvec_right


def compute_kernel_eigenpairs(
    kernel,
    kernel_matrix: Union[np.ndarray, scipy.sparse.csr_matrix],
    n_eigenpairs: int,
    normalize_eigenvectors: bool = False,
    backend: str = "scipy",
    validate_matrix: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute eigenvalues and -vectors from kernel matrix with consideration of matrix
    properties.

    Parameters
    ----------
    kernel_matrix
        Kernel matrix of shape `(n_samples, n_samples)`.

    n_eigenpairs
        Number of eigenpairs to compute.

    normalize_eigenvectors
        If True, all eigenvectors are normalized to length 1.

    backend
        Valid backends:
            * "scipy"

    validate_matrix
        Validate matrix that it fulfills the specified properties (primarily used for
        testing).

    Returns
    -------
    numpy.ndarray
        Eigenvalues in ascending order (absolute value).

    numpy.ndarray
        Eigenvectors (not necessarily normalized) in the same order to eigenvalues.
    """
    is_matrix(kernel_matrix, "kernel_matrix", square=True, allow_sparse=True)

    err_nonfinite = ValueError(
        "kernel_matrix must only contain finite values (no np.nan or np.inf)"
    )
    if (
        scipy.sparse.issparse(kernel_matrix)
        and not np.isfinite(kernel_matrix.data).all()
    ):
        raise err_nonfinite
    elif isinstance(kernel_matrix, np.ndarray) and not np.isfinite(kernel_matrix).all():
        raise err_nonfinite

    if kernel_matrix.dtype == bool:
        # cast bools to float64, otherwise the resulting vectors are in float32
        kernel_matrix = kernel_matrix.astype(np.float64)

    if validate_matrix:
        if kernel.is_symmetric and not is_symmetric_matrix(kernel_matrix):
            raise ValueError("kernel_matrix is not symmetric")

        # TODO: include this after kernel refactor is carried out in #149
        # if kernel_content.is_symmetric and not is_stochastic_matrix(kernel_matrix, axis=1):
        #      raise ValueError("kernel_matrix is not stochastic")

    if backend == "scipy":
        eigvals, eigvects = scipy_eigsolver(
            kernel=kernel,
            kernel_matrix=kernel_matrix,
            n_eigenpairs=n_eigenpairs,
        )
    else:
        raise ValueError(f"backend {backend} not known.")

    if not np.isfinite(eigvals).all() or not np.isfinite(eigvects).all():
        raise NumericalMathError(
            "eigenvalues or eigenvectors contain 'NaN' or 'inf' values."
        )

    if kernel._is_symmetric_kernel:
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


def compute_kernel_svd(
    kernel_matrix: Union[np.ndarray, scipy.sparse.csr_matrix],
    n_svdtriplet: int,
    backend: str = "scipy",
    **backend_kwargs,
):
    if n_svdtriplet > np.min(kernel_matrix.shape):
        raise ValueError(
            f"{n_svdtriplet} is larger than the maximum number of SVD triplets available "
            f"(={np.min(kernel_matrix.shape)})"
        )

    if backend == "scipy":
        svdvec_left, svdvals, svdvec_right = scipy_svdsolver(
            kernel_matrix, n_svdvtriplets=n_svdtriplet, **backend_kwargs
        )
    else:
        raise ValueError(
            f"SVD backend {backend} is not available. "
            f"Choose from {_valid_svd_backends}"
        )

    if np.iscomplexobj(svdvec_left) or np.iscomplexobj(svdvec_right):
        # Note that the singular values are guaranteed to be real-valued

        max_imag_entry = max(
            np.abs(np.imag(svdvec_left)).max(), np.abs(np.imag(svdvec_right)).max()
        )

        if max_imag_entry > 1e2 * sys.float_info.epsilon:
            raise NumericalMathError(
                "SVD eigenvectors have non-negligible imaginary part (larger than "
                f"{1e2 * sys.float_info.epsilon})."
            )
        else:
            svdvec_left = np.real(svdvec_left)
            svdvec_right = np.real(svdvec_right)

    # Note: it is correct that left/right is opposite here
    svdvals, svdvec_left, svdvec_right = sort_eigenpairs(
        svdvals, right_eigenvectors=svdvec_left, left_eigenvectors=svdvec_right
    )

    return svdvec_left, svdvals, svdvec_right
