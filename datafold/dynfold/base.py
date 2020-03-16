#!/usr/bin/env python3

import sys
from typing import Tuple

import numpy as np
import scipy
import scipy.sparse
import scipy.sparse.linalg
from sklearn.base import BaseEstimator

from datafold.pcfold.eigsolver import NumericalMathError, compute_kernel_eigenpairs


class DmapKernelMethod(BaseEstimator):
    def __init__(
        self,
        epsilon: float,
        n_eigenpairs: int,
        cut_off,
        is_stochastic: bool,
        alpha: float,
        symmetrize_kernel,
        use_cuda,
        dist_backend,
        dist_backend_kwargs,
    ):
        self.epsilon = epsilon
        self.n_eigenpairs = n_eigenpairs
        self.cut_off = cut_off
        self.is_stochastic = is_stochastic
        self.alpha = alpha
        self.symmetrize_kernel = symmetrize_kernel
        self.use_cuda = use_cuda
        self.dist_backend = dist_backend
        self.dist_backend_kwargs = dist_backend_kwargs

    @property
    def kernel_(self):
        if not hasattr(self, "_kernel"):
            raise AttributeError(
                f"Subclass {type(self)} is not properly implemented. "
                f"No attribute '_kernel' in KernelMethod subclass. Please report bug."
            )

        return getattr(self, "_kernel")

    def _unsymmetric_kernel_matrix(self, kernel_matrix, basis_change_matrix):
        inv_basis_change_matrix = scipy.sparse.diags(
            np.reciprocal(basis_change_matrix.data.ravel())
        )

        return inv_basis_change_matrix @ kernel_matrix @ inv_basis_change_matrix

    def _solve_eigenproblem(
        self, kernel_matrix, basis_change_matrix, use_cuda
    ) -> Tuple[np.ndarray, np.ndarray]:

        if not use_cuda:
            backend = "scipy"
        else:
            backend = "gpu"

        try:
            eigvals, eigvect = compute_kernel_eigenpairs(
                matrix=kernel_matrix,
                n_eigenpairs=self.n_eigenpairs,
                is_symmetric=self.kernel_.is_symmetric,
                is_stochastic=self.is_stochastic,
                backend=backend,
            )
        except NumericalMathError:
            # re-raise with more details for the DMAP
            raise NumericalMathError(
                "Eigenvalues have non-negligible imaginary part (larger than "
                f"{1e2 * sys.float_info.epsilon}. First try to use "
                f"parameter 'symmetrize_kernel=True' (improves numerical stability) and "
                f"only if this is not working adjust epsilon."
            )

        if basis_change_matrix is not None:
            eigvect = basis_change_matrix @ eigvect

        eigvect /= np.linalg.norm(eigvect, axis=0)[np.newaxis, :]

        return np.real(eigvals), np.real(eigvect)
