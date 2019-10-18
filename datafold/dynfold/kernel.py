#!/usr/bin/env python3 

import logging
import sys
from typing import Callable, Optional, Union

import numpy as np
import numbers
import scipy
import scipy.sparse
import scipy.sparse.linalg
from sklearn.base import BaseEstimator
from sklearn.gaussian_process.kernels import Kernel, StationaryKernelMixin
from sklearn.preprocessing import normalize

from pcmanifold.distance import compute_distance_matrix
from pcmanifold.kernels import PCManifoldKernelMixin, RadialBasisKernel

gpu_eigsolve: Optional[Callable]  # optional because GPU/CUDA code may not be available

try:
    from pydmap.gpu_eigensolver import eigensolver as gpu_eigsolve
except ImportError:
    gpu_eigsolve = None
    SUCCESS_GPU_IMPORT = False   # variable is used to warn user when requesting GPU eigensolver
else:
    SUCCESS_GPU_IMPORT = True


class NumericalMathError(Exception):
    """Use for numerical problems/issues, such as singular matrices or too large imaginary part."""


class KernelMethod(BaseEstimator):

    def __init__(self, epsilon: float, num_eigenpairs: int, cut_off, is_stochastic: bool, alpha: float,
                 symmetrize_kernel, use_cuda, dist_backend, dist_backend_kwargs):

        super(KernelMethod, self).__init__()

        self.epsilon = epsilon
        self.num_eigenpairs = num_eigenpairs
        self.cut_off = cut_off
        self.is_stochastic = is_stochastic
        self.alpha = alpha
        self.symmetrize_kernel = symmetrize_kernel
        self.use_cuda = use_cuda
        self.dist_backend = dist_backend

        if dist_backend_kwargs is None:
            self.dist_backend_kwargs = {}  # empty to use **kwargs
        else:
            self.dist_backend_kwargs = dist_backend_kwargs

        # variables that need to be set by subclasses
        self._kernel = None
        self._kernel_matrix = None
        self._basis_change_matrix = None

    @property
    def kernel_(self):
        if self._kernel is None:
            raise AttributeError("Subclass is not properly implemented. No attribute self._kernel found.")
        return self._kernel

    @property
    def kernel_matrix_(self):
        # TODO: could compute transform back the true kernel (corrected by conjugate transform)
        # if self._basis_change_matrix is not None:
        #     assert isinstance(self._basis_change_matrix, scipy.sparse.dia_matrix)  # assumption required for inverse
        #     inv_basis_change_matrix = scipy.sparse.diags(1 / self._basis_change_matrix.data.ravel())
        #     return_kernel = inv_basis_change_matrix @ self._kernel_matrix @ inv_basis_change_matrix
        # else:
        return_kernel = self._kernel_matrix
        return return_kernel

    def solve_eigenproblem(self, kernel_matrix, basis_change_matrix, use_cuda):
        """Solve eigenproblem (eigenvalues and eigenvectors) using CPU or GPU solver.
        """

        # ----
        # TODO: insert this, this allows to have another "layer" to sparsify the matrix (note the cut_off is for
        #    distances, but after the kernel is applied, there may be  small numbers). --> only apply this, if there
        #    are values in the order of 1E-1 or greater (additional test required)
        #    also print a warning if there are many removed...)
        # cut_off = 1E-15  # TODO: for now cut-off approx. numerical double precision, parametrize if necessary
        # bool_idx = np.abs(kernel_matrix) < cut_off
        # print(f"The cut-off rate is {cut_off}. There are {np.sum(np.sum(bool_idx))} values set to zero.")
        # kernel_matrix[bool_idx] = 0
        # ----

        if use_cuda:
            assert gpu_eigsolve is not None
            if not SUCCESS_GPU_IMPORT:
                logging.warning("Importing GPU eigensolver failed, falling back to CPU eigensolver.")
            # Note: the GPU eigensolver is not maintained. It cannot handle the special case of is_symmetric=True
            eigvals, eigevects = gpu_eigsolve(kernel_matrix, self.num_eigenpairs)
        else:
            eigvals, eigevects = self.cpu_eigensolver(kernel_matrix,
                                                      is_symmetric=self.kernel_.is_symmetric,
                                                      k=self.num_eigenpairs,
                                                      v0=np.ones(kernel_matrix.shape[0]),
                                                      which="LM",  # largest magnitude
                                                      sigma=None,
                                                      tol=1E-13)

        if basis_change_matrix is not None:
            # NOTE: this the order has to be reverted here, when eigenvectors are column-wise (TODO: #44)
            eigevects = eigevects @ basis_change_matrix

        if np.any(eigvals.imag > 1e2 * sys.float_info.epsilon):
            raise NumericalMathError("Eigenvalues have non-negligible imaginary part. First try to use "
                                     "'symmetrize_kernel=True' (numerically more stable) and only if this is not "
                                     "working try to adjust epsilon.")

        return eigvals.real, eigevects.real

    @staticmethod
    def cpu_eigensolver(matrix: Union[np.ndarray, scipy.sparse.csr_matrix], is_symmetric: bool, **solver_kwargs):
        """Solve eigenvalue problem for sparse matrix.

        Parameters
        ----------
        matrix : np.ndarray (dense) or scipy.sparse.csr_matrix (sparse)
            Matrix to solve the eigenproblem for.
        is_symmetric : bool,
            If symmetric matrix scipy.eigsh solver is used, else scipy.eigs
        **solver_kwargs : kwargs
            All parameter handed to eigenproblem solver, see documentation:
            symmetric case:
            https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.eigsh.html
            non-symmetric case:
            https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.eigs.html#scipy.sparse.linalg.eigs

        Returns
        -------
        np.ndarray
            Eigenvalues in descending order of magnitude.
        np.ndarray
            Eigenvectors corresponding to the eigenvalues.
        """

        # TODO: currently the sparse implementation is used (also for dense kernels), possibly solve with
        #  the numpy.eig solver for dense case? The problem with the sparse solvers is, that they are not able to
        #  compute *all* eigenpairs (see doc), but the numpy/dense case can.

        if is_symmetric:
            assert _check_symmetric(matrix)
            eigvals, eigvects = scipy.sparse.linalg.eigsh(matrix, **solver_kwargs)
        else:
            eigvals, eigvects = scipy.sparse.linalg.eigs(matrix, **solver_kwargs)

        ii = np.argsort(np.abs(eigvals))[::-1]
        return eigvals[ii], eigvects[:, ii].T  # TODO: #44


class DmapKernelFixed(StationaryKernelMixin, PCManifoldKernelMixin, Kernel):
    """RBF kernel which with the following extension:
          * stochastic kernel (-> non-symmetric)
               ** can use the (symmetric) conjugate matrix
               ** renormalization (alpha)
    """

    # TODO: check whether this is correct: the kernel is stationary, if stochastic=True and alpha=1

    def __init__(self, epsilon=1.0, length_scale_bounds=(1e-5, 1e5), is_stochastic=True, alpha=1.0,
                 symmetrize_kernel=False):

        # TODO: not sure if I need this length_scale_bounds?! Check in the scikit learn doc what this is about

        if not (0 <= alpha <= 1):
            raise ValueError(f"alpha has to be between [0, 1]. Got alpha={alpha}")

        self.alpha = alpha
        self.is_stochastic = is_stochastic
        self._internal_rbf_kernel = RadialBasisKernel(epsilon, length_scale_bounds=length_scale_bounds)

        if not is_stochastic or symmetrize_kernel:
            # If not stochastic, the kernel is always symmetric
            # `symmetrize_kernel` indicates if the user wants the kernel to use similarity transformations to solve the
            # eigenproblem on a symmetric kernel (if required).

            # NOTE: a necessary condition to symmetrize the kernel is that the kernel is evaluated pairwise
            #  (i.e. is_pdist = True)
            self.is_symmetric = True
        else:
            self.is_symmetric = False

        # TODO: check if the super call is reaching the Kernel class, currently the output does not show the params
        super(DmapKernelFixed, self).__init__()

    def _normalize(self, rbf_kernel, is_pdist):

        basis_change_matrix = None  # only required for symmetric kernel, return None if not used

        if self.is_stochastic:

            if self.alpha > 0:
                rbf_kernel = normalize_kernel_matrix(rbf_kernel, self.alpha)  # here rbf_kernel is usually non-symmetric

            if self.is_symmetric_transform(is_pdist):
                # increases numerical stability when solving the eigenproblem
                # -- Note: when using the (symmetric) conjugate matrix, the eigenvectors have to be transformed back
                #          to match the original
                # -- Note: the similarity transform only works for the is_pdist case (for cdist, there is no symmetric
                #          kernel in the first place
                rbf_kernel, basis_change_matrix = conjugate_stochastic_kernel_matrix(rbf_kernel)
            else:
                rbf_kernel = stochastic_kernel_matrix(rbf_kernel)

            assert (self.is_symmetric_transform(is_pdist) and basis_change_matrix is not None) or \
                   (not self.is_symmetric_transform(is_pdist) and basis_change_matrix is None)

        if is_pdist and self.is_symmetric:
            assert _check_symmetric(rbf_kernel)

        return rbf_kernel, basis_change_matrix

    def is_symmetric_transform(self, is_pdist):
        # If the kernel is made stochastic, it looses the symmetry, if symmetric_kernel is set to True, then apply the
        # the symmetry transformation
        return is_pdist and self.is_stochastic and self.is_symmetric

    def __call__(self, X, Y=None, dist_cut_off=None, dist_backend="guess_optimal", **backend_options):

        if self.is_stochastic and Y is not None:  # cdist case
            raise NotImplementedError("see issue pydmap/#65")  # TODO

        rbf_kernel = self._internal_rbf_kernel(X, Y=Y,
                                               dist_cut_off=dist_cut_off,
                                               dist_backend=dist_backend,
                                               **backend_options)

        is_pdist = Y is None
        kernel_matrix, basis_change_matrix = self._normalize(rbf_kernel, is_pdist)

        return kernel_matrix, basis_change_matrix  # basis_change_matrix is None if not required for back-transformation

    def eval(self, distance_matrix):
        if _check_symmetric(distance_matrix) and (np.diag(distance_matrix) == 0).all():
            is_pdist = True
        else: # cdist
            is_pdist = False

        rbf_kernel = self._internal_rbf_kernel.eval(distance_matrix)
        kernel_matrix, basis_change_matrix = self._normalize(rbf_kernel, is_pdist)

        return kernel_matrix, basis_change_matrix

    def gradient(self, X, X_eval):
        if self.is_stochastic:
            raise NotImplementedError("Not implemented, also not sure how it works with stochastic and renormalization")
        else:
            return self._internal_rbf_kernel._gradient(X, X_eval)

    def diag(self, X):
        """Implementing abstract function, required by the Kernel interface."""
        if not self.is_stochastic:
            return self._internal_rbf_kernel.diag(X)
        else:
            # TODO: Most likely it is required to compute the full kernel self(X, X) and then take the diag.
            #  However, there are more options required now (such as cut_off, etc.)
            raise NotImplementedError("This case a bit more complicated. Also not sure if this is required.")


class DmapKernelVariable(StationaryKernelMixin, PCManifoldKernelMixin, Kernel):

    def __init__(self, epsilon, k, expected_dim, beta, symmetrize_kernel):
        if expected_dim <= 0 and not isinstance(expected_dim, numbers.Integral):
            raise ValueError("expected_dim has to be a non-negative integer.")

        if epsilon < 0 and not not isinstance(expected_dim, numbers.Real):
            raise ValueError("epsilon has to be positive float.")

        if k <= 0 and not isinstance(expected_dim, numbers.Integral):
            raise ValueError("k has to be a non-negative integer.")

        self.beta = beta
        self.epsilon = epsilon
        self.k = k
        self.expected_dim = expected_dim  # variable 'd' in paper

        if symmetrize_kernel:  # allows to later on include a stochastic option...
            self.is_symmetric = True
        else:
            self.is_symmetric = False

        self.alpha = -self.expected_dim / 4
        c2 = 1 / 2 - 2 * self.alpha + 2 * self.expected_dim * self.alpha + self.expected_dim * self.beta / 2 + self.beta

        if c2 >= 0:
            raise ValueError("Theory requires c2 to be negative:\n"
                             "c2 = 1/2 - 2 * alpha + 2 * expected_dim * alpha + expected_dim * beta/2 + beta \n"
                            f"but is {c2}")

    def _compute_rho0(self, distance_matrix):

        nr_samples = distance_matrix.shape[1]

        MODE = 1  # both modes are equivalent, MODE=1 allows to easier compare with ref3.
        if MODE == 1:  # according to Berry Code
            distance_matrix = np.sort(np.sqrt(distance_matrix), axis=1)[:, :self.k]  # keep only nearest neighbors
            rho0 = np.sqrt(np.mean(distance_matrix[:, 1:] ** 2, axis=1))  # like in berry code.
        elif MODE == 2:  # more performant if required
            if self.k < nr_samples:
                # TODO: have to revert setting the inf
                #   -> check that the diagonal is all zeros
                #   -> set to inf
                #   -> after computation set all infs back to zero
                #   -> this is also very similar to continous-nn in PCManifold
                np.fill_diagonal(distance_matrix, np.inf)  # this allows to ignore the trivial distance=0 to itself
                distance_matrix.partition(self.k, axis=1)
                distance_matrix = np.sort(distance_matrix[:, :self.k], axis=1)
                distance_matrix = distance_matrix * distance_matrix
            else:  # self.k == self.N
                np.fill_diagonal(distance_matrix, np.nan)
                bool_mask = ~np.diag(np.ones(nr_samples)).astype(np.bool)
                distance_matrix = distance_matrix[bool_mask].reshape(distance_matrix.shape[0], distance_matrix.shape[1] - 1)

            # experimental: ---------------------------------------------------------------------------------
            # paper: in var-bw paper (ref2) pdfp. 7
            # it is mentioned to IGNORE non-zero entries -- this is not detailed more.
            # a consequence is that the NN and kernel looses symmetry, so do (K+K^T) / 2
            # This is with a cut off rate:
            # val = 1E-2
            # distance_matrix[distance_matrix < val] = np.nan
            # experimental END ------------------------------------------------------------------------------

            # nanmean only for the experimental part, if leaving this out, np.mean suffices
            rho0 = np.sqrt(np.nanmean(distance_matrix, axis=1))
        return rho0

    def _compute_q0(self, distance_matrix, rho0):
        meanrho0 = np.mean(rho0)
        rho0tilde = rho0 / meanrho0

        eps0 = meanrho0 ** 2  # TODO: eps0 could also be optimized (see Berry Code + paper ref2)

        expon_matrix = _symmetric_matrix_division(matrix=-distance_matrix, vec=rho0tilde, scalar=2 * eps0)
        assert _check_symmetric(expon_matrix)

        nr_samples = distance_matrix.shape[0]

        # according to eq. (10) in ref1
        q0 = np.power(2 * np.pi * eps0, - self.expected_dim / 2) / (np.power(rho0, self.expected_dim) * nr_samples) \
             * np.sum(np.exp(expon_matrix), axis=1)

        return q0

    def _compute_rho(self, q0):
        rho = np.power(q0, self.beta)
        rho = rho / np.mean(rho)  # Division by mean is not in papers, but in berry code (ref3)
        return rho

    def _compute_kernel_eps_s(self, distance_matrix, rho):
        expon_matrix = _symmetric_matrix_division(matrix=-distance_matrix, vec=rho, scalar=4 * self.epsilon)
        kernel_eps_s = np.exp(expon_matrix)
        assert _check_symmetric(kernel_eps_s)
        return kernel_eps_s

    def _compute_q_eps_s(self, kernel_eps_s, rho):
        rho_power_dim = np.power(rho, self.expected_dim)[:, np.newaxis]
        q_eps_s = np.sum(kernel_eps_s / rho_power_dim, axis=1)
        return q_eps_s

    def _compute_kernel_eps_alpha_s(self, kernel_eps_s, q_eps_s):
        kernel_eps_alpha_s = _symmetric_matrix_division(matrix=kernel_eps_s, vec=np.power(q_eps_s, self.alpha))
        assert _check_symmetric(kernel_eps_alpha_s)

        return kernel_eps_alpha_s

    def _compute_matrix_l(self, kernel_eps_alpha_s, rho):
        rhosq = (rho ** 2)[:, np.newaxis]
        n_samples = rho.shape[0]
        matrix_l = (kernel_eps_alpha_s - np.eye(n_samples)) / (self.epsilon * rhosq)
        return matrix_l

    def _compute_matrix_s_inv(self, rho, q_eps_alpha_s):
        s_diag = 1 / (rho * np.sqrt(q_eps_alpha_s))
        # matrix_s_inv = scipy.sparse.spdiags(s_diag, diags=0, m=s_diag.shape[0], n=s_diag.shape[0])
        matrix_s_inv = np.diag(s_diag)
        return matrix_s_inv

    def _compute_matrix_l_conjugate(self, kernel_eps_alpha_s, rho, q_eps_alpha_s):

        basis_change_matrix = self._compute_matrix_s_inv(rho, q_eps_alpha_s)
        #p_sq_inv = scipy.sparse.spdiags(1 / (rho ** 2), diags=0, m=basis_change_matrix.shape[0], n=basis_change_matrix.shape[1])
        p_sq_inv = np.diag(np.reciprocal(np.square(rho)))

        # as described in paper, but more prone to numerical issues:
        # mat = 1/self.eps * (basis_change_matrix @ kernel_eps_alpha_s @ basis_change_matrix - p_sq_inv)
        # mat = (basis_change_matrix @ kernel_eps_alpha_s @ basis_change_matrix - p_sq_inv)

        matrix_l_hat = basis_change_matrix @ kernel_eps_alpha_s @ basis_change_matrix - \
                       (p_sq_inv - np.eye(kernel_eps_alpha_s.shape[0]))

        return matrix_l_hat, basis_change_matrix   # matrix_l_hat conjugate kernel matrix

    def __call__(self, X, Y=None, dist_cut_off=None, dist_backend="guess_optimal", **backend_options):

        if dist_cut_off is not None and not np.isinf(dist_cut_off):
            raise NotImplementedError("Handling sparsity is currently not implemented!")

        if Y is not None:
            raise NotImplementedError("cdist case is currently not implemented!")

        if self.k > X.shape[0]:
            raise ValueError(f"nr of nearest neighbors (self.k={self.k}) is larger than number of samples (={X.shape[0]})")

        distance_matrix = compute_distance_matrix(X, Y,
                                                  metric="sqeuclidean",
                                                  cut_off=dist_cut_off,
                                                  backend=dist_backend,
                                                  **backend_options)

        kernel_matrix, basis_change_matrix, rho0, rho, q0, q_eps_s = self.eval(distance_matrix)

        return kernel_matrix, basis_change_matrix, rho0, rho, q0, q_eps_s  # TODO: make a return_vectors option?

    def eval(self, distance_matrix):
        if scipy.sparse.issparse(distance_matrix):
            raise NotImplementedError("Currently the variable bandwidth kernel is only implemented for the dense "
                                      "distance matrix case.")

        assert _check_symmetric(distance_matrix) and (np.diag(distance_matrix) == 0).all(), \
            "only pdist case supported at the moment"

        rho0 = self._compute_rho0(distance_matrix)
        q0 = self._compute_q0(distance_matrix, rho0)
        rho = self._compute_rho(q0)
        kernel_eps_s = self._compute_kernel_eps_s(distance_matrix, rho)
        q_eps_s = self._compute_q_eps_s(kernel_eps_s, rho)
        kernel_eps_alpha_s = self._compute_kernel_eps_alpha_s(kernel_eps_s, q_eps_s)

        if self.is_symmetric:
            q_eps_alpha_s = kernel_eps_alpha_s.sum(axis=1)
            kernel_matrix, basis_change_matrix = self._compute_matrix_l_conjugate(kernel_eps_alpha_s, rho, q_eps_alpha_s)
        else:
            basis_change_matrix = None
            kernel_matrix = self._compute_matrix_l(kernel_eps_alpha_s, rho)

        return kernel_matrix, basis_change_matrix, rho0, rho, q0, q_eps_s  # TODO: make a return_vectors option?

    def diag(self, X):
        """Implementing abstract function, required by the Kernel interface."""
        raise NotImplementedError("This case a bit more complicated. Also not sure if this is required.")


def normalize_kernel_matrix(kernel_matrix, alpha):
    """Function to normalize (sparse/dense) kernels."""

    assert 0 <= alpha <= 1

    if scipy.sparse.issparse(kernel_matrix):
        row_sums = kernel_matrix.sum(axis=1).A1

        inv_diag = np.reciprocal(np.power(row_sums, alpha), out=row_sums)
        inv_diag = np.nan_to_num(inv_diag, copy=False)
        sparse_inv_diag = scipy.sparse.spdiags(inv_diag, 0, *kernel_matrix.shape)

        # TODO: not sure if scipy makes this efficiently? the sparse_inv_diag is in DIA format (sparse diagonal)
        #  alternative: apply the vector on it.
        #  the misc/microbenchmark_stochastic_matrix.py suggests that there are faster ways (increase of factor ~8, in
        #  this case probably more, bc. the @ operation is used twice
        normalized_kernel = sparse_inv_diag @ kernel_matrix @ sparse_inv_diag

    else:  # dense
        row_sums = np.ravel(kernel_matrix.sum(axis=1))
        inv_diag = np.reciprocal(np.power(row_sums, alpha), out=row_sums)
        inv_diag = np.nan_to_num(inv_diag, copy=False)

        normalized_kernel = np.multiply(inv_diag[:, np.newaxis], kernel_matrix)
        normalized_kernel = np.multiply(normalized_kernel, inv_diag, out=normalized_kernel)

    return normalized_kernel


def _check_symmetric(matrix, tol=1E-14):
    max_abs_deviation = np.max(np.abs(matrix - matrix.T))
    if max_abs_deviation > tol:
        print(f"WARNING: deviation is {max_abs_deviation}")
        return False
    else:
        return True


def _symmetric_matrix_division(matrix, vec, scalar=None):

    if scipy.sparse.issparse(matrix):
        raise NotImplementedError("no sparse version implemented yet")

    if vec.ndim == 1:
        vec = vec[:, np.newaxis]

    # TODO: check/try if there are more memory efficient ways, that also keep **perfectly** (numerically identical) the
    #  symmetry
    denom = np.ones_like(matrix) * vec * vec.T
    matrix = np.divide(matrix, denom, out=matrix)

    if scalar is not None:
        scalar = 1 / scalar
        matrix = matrix * scalar
    return matrix


def conjugate_stochastic_kernel_matrix(kernel_matrix):
    """Make the kernel matrix symmetric, the spectrum is kept because of a similarity transformation. This is only
       required if alpha>0 (i.e. renormalization) which makes the kernel matrix non-symmetric.

       NOTE: the conjugate-stochastic matrix is not stochastic in the strict sense (i.e. the row-sum is not equal to 1).
       The eigenvalues and eigenvectors (after transformation) of the conjugate_stochastic is the same to the
       stochastic.
       """

    if scipy.sparse.issparse(kernel_matrix):
        # Note: sparse_matrix.sum(axis=1) returns np.matrix -- 'A1' converts the object to a flattened ndarray.
        # basis_change_matrix = scipy.sparse.diags(kernel_matrix.sum(axis=1).A1 ** (-1/2))
        basis_change_matrix = kernel_matrix.sum(axis=1).A1
        basis_change_matrix = scipy.sparse.diags(np.reciprocal(np.sqrt(basis_change_matrix)))
        kernel_matrix = basis_change_matrix @ kernel_matrix @ basis_change_matrix
    else:  # dense
        division_vec = np.sqrt(kernel_matrix.sum(axis=1))
        kernel_matrix = _symmetric_matrix_division(np.copy(kernel_matrix), vec=division_vec)
        basis_change_matrix = scipy.sparse.diags(np.reciprocal(division_vec, out=division_vec))

    return kernel_matrix, basis_change_matrix


def stochastic_kernel_matrix(kernel_matrix):
    """Function to make (sparse/dense) kernel stochastic."""
    if scipy.sparse.issparse(kernel_matrix):
        # see microbenchmark_stochastic_matrix.py, for the sparse case this variant is faster) then the previous code
        kernel_matrix = normalize(kernel_matrix, copy=False, norm="l1")

        # Old code: (loop is expensive)
        # data = kernel_matrix.data
        # indptr = kernel_matrix.indptr
        # for i in range(kernel_matrix.shape[0]):
        #     a, b = indptr[i:i + 2]
        #     norm1 = np.sum(data[a:b])
        #     data[a:b] /= norm1
    else:  # dense
        normalize_diagonal = np.reciprocal(np.sum(kernel_matrix, axis=1))
        kernel_matrix = np.multiply(normalize_diagonal[:, np.newaxis], kernel_matrix)

    return kernel_matrix
