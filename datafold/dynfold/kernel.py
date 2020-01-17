#!/usr/bin/env python3

import logging
import sys
from typing import Callable, Optional, Tuple, Union

import numpy as np
import scipy
import scipy.sparse
import scipy.sparse.linalg
from sklearn.base import BaseEstimator
from sklearn.gaussian_process.kernels import Kernel, StationaryKernelMixin
from sklearn.preprocessing import normalize

from datafold.pcfold.distance import compute_distance_matrix
from datafold.pcfold.kernels import PCManifoldKernelMixin, RadialBasisKernel
from datafold.utils.datastructure import is_float, is_integer
from datafold.utils.maths import (
    diagmat_dot_mat,
    is_symmetric_matrix,
    mat_dot_diagmat,
    remove_numeric_noise_symmetric_matrix,
)

try:
    from pydmap.gpu_eigensolver import eigensolver as gpu_eigsolve
except ImportError:
    gpu_eigsolve: Optional[Callable] = None  # type: ignore

    # variable is used to warn user when requesting GPU eigensolver
    SUCCESS_GPU_IMPORT = False
else:
    SUCCESS_GPU_IMPORT = True


class NumericalMathError(Exception):
    """Use for numerical problems/issues, such as singular matrices or too large
    imaginary part."""


class KernelMethod(BaseEstimator):
    def __init__(
        self,
        epsilon: float,
        num_eigenpairs: int,
        cut_off,
        is_stochastic: bool,
        alpha: float,
        symmetrize_kernel,
        use_cuda,
        dist_backend,
        dist_backend_kwargs,
    ):

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
            self.dist_backend_kwargs: dict = {}  # empty to use **kwargs
        else:
            self.dist_backend_kwargs = dist_backend_kwargs

        # variables that need to be set by subclasses
        self._kernel: Kernel
        self.operator_matrix_ = None

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

    def solve_eigenproblem(self, kernel_matrix, basis_change_matrix, use_cuda):
        """Solve eigenproblem (eigenvalues and eigenvectors) using CPU or GPU solver.
        """

        # ----
        # TODO: insert this, this allows to have another "layer" to sparsify the matrix
        #  (note the cut_off is for distances, but after the kernel is applied,
        #  there may be  small numbers). --> only apply this, if there
        #    are values in the order of 1E-1 or greater (additional test required)
        #    also print a warning if there are many removed...)
        # TODO: for now cut-off approx. numerical double precision,parametrize if
        #  necessary
        # cut_off = 1E-15
        # bool_idx = np.abs(kernel_matrix) < cut_off
        # print(f"The cut-off rate is {cut_off}. There are {np.sum(np.sum(bool_idx))}
        # values set to zero.")
        # kernel_matrix[bool_idx] = 0
        # ----

        if use_cuda and SUCCESS_GPU_IMPORT:
            # Note: the GPU eigensolver is not maintained. It cannot handle the special
            # case of is_symmetric=True
            eigvals, eigvect = gpu_eigsolve(kernel_matrix, self.num_eigenpairs)
        else:
            if use_cuda and not SUCCESS_GPU_IMPORT:
                logging.warning(
                    "Importing GPU eigensolver failed, falling back to CPU eigensolver."
                )

            eigvals, eigvect = self.cpu_eigensolver(
                kernel_matrix,
                is_symmetric=self.kernel_.is_symmetric,
                k=self.num_eigenpairs,
                v0=np.ones(kernel_matrix.shape[0]),
                which="LM",  # largest magnitude
                sigma=None,
                tol=1e-13,
            )

        if basis_change_matrix is not None:
            # NOTE: this order has to be reverted, when eigenvectors are
            # column-wise (TODO: #44)
            eigvect = eigvect @ basis_change_matrix

        # normalize eigenvectors to 1 (change if required differently).
        eigvect /= np.linalg.norm(eigvect, axis=1)[:, np.newaxis]

        if np.any(eigvals.imag > 1e2 * sys.float_info.epsilon):
            raise NumericalMathError(
                "Eigenvalues have non-negligible imaginary part (larger than "
                f"{1e2 * sys.float_info.epsilon}. First try to use "
                f"the option 'symmetrize_kernel=True' (numerically more stable) and "
                f"only if this is not working try to adjust epsilon."
            )

        return eigvals.real, eigvect.real

    @staticmethod
    def cpu_eigensolver(
        matrix: Union[np.ndarray, scipy.sparse.csr_matrix],
        is_symmetric: bool,
        **solver_kwargs,
    ):
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
        eigenvalues : np.ndarray
            Eigenvalues in descending order of magnitude.
        eigenvectors : np.ndarray
            Eigenvectors corresponding to the eigenvalues.
        """

        # TODO: currently the sparse implementation is used (also for dense kernels),
        #  possibly solve with the numpy.eig solver for dense case? The problem with
        #  the sparse solvers is, that they are not able to compute *all* eigenpairs (
        #  see doc), but the numpy/dense case can.

        if is_symmetric:
            assert is_symmetric_matrix(matrix)
            eigvals, eigvects = scipy.sparse.linalg.eigsh(matrix, **solver_kwargs)
        else:
            eigvals, eigvects = scipy.sparse.linalg.eigs(matrix, **solver_kwargs)

        if np.isnan(eigvals).any() or np.isnan(eigvects).any():
            raise RuntimeError(
                "eigenvalues or eigenvector contains NaN values. Maybe try a larger "
                "epsilon value."
            )

        ii = np.argsort(np.abs(eigvals))[::-1]
        return eigvals[ii], eigvects[:, ii].T  # TODO: #44


class DmapKernelFixed(StationaryKernelMixin, PCManifoldKernelMixin, Kernel):
    """RBF kernel which with the following extension:
          * stochastic kernel (-> non-symmetric)
               ** can use the (symmetric) conjugate matrix
               ** renormalization (alpha)
    """

    # TODO: check whether this is correct: the kernel is stationary, if stochastic=True
    #  and alpha=1

    def __init__(
        self,
        epsilon=1.0,
        length_scale_bounds=(1e-5, 1e5),
        is_stochastic=True,
        alpha=1.0,
        symmetrize_kernel=False,
    ):

        self.row_sums_init = None
        # TODO: not sure if I need this length_scale_bounds?! Check in the scikit learn
        #  doc what this is about

        self.epsilon = epsilon

        if not (0 <= alpha <= 1):
            raise ValueError(f"alpha has to be between [0, 1]. Got alpha={alpha}")
        self.alpha = alpha

        self.is_stochastic = is_stochastic
        self._internal_rbf_kernel = RadialBasisKernel(
            self.epsilon, length_scale_bounds=length_scale_bounds
        )

        if not is_stochastic or symmetrize_kernel:
            # If not stochastic, the kernel is always symmetric
            # `symmetrize_kernel` indicates if the user wants the kernel to use
            # similarity transformations to solve the
            # eigenproblem on a symmetric kernel (if required).

            # NOTE: a necessary condition to symmetrize the kernel is that the kernel
            # is evaluated pairwise
            #  (i.e. is_pdist = True)
            self.is_symmetric = True
        else:
            self.is_symmetric = False

        # TODO: check if the super call is reaching the Kernel class, currently the
        #  output does not show the params
        super(DmapKernelFixed, self).__init__()

    def _normalize_sampling_density_kernel_matrix(
        self, kernel_matrix, row_sums_alpha_fit
    ):
        """Normalize (sparse/dense) kernels with positive `alpha` value. This is also
        referred to 'renormalization'. """

        if row_sums_alpha_fit is None:
            assert is_symmetric_matrix(kernel_matrix)
        else:
            assert row_sums_alpha_fit.shape[0] == kernel_matrix.shape[1]

        row_sums = kernel_matrix.sum(axis=1)

        if scipy.sparse.issparse(kernel_matrix):
            row_sums = row_sums.A1  # turns matrix (deprectated) into np.ndarray

        if self.alpha < 1:
            row_sums_alpha = np.power(row_sums, self.alpha, out=row_sums)
            # TODO: not sure why this line is here, maybe can remove:
            #  (comment out for now)
            # row_sums_alpha = np.nan_to_num(row_sums_alpha, copy=False)
        else:  # no need to power with 1
            row_sums_alpha = row_sums

        normalized_kernel = _symmetric_matrix_division(
            matrix=kernel_matrix, vec=row_sums_alpha, vec_right=row_sums_alpha_fit,
        )

        if row_sums_alpha_fit is not None:
            # Set row_sums_alpha to None for security, because in a cdist-case (if
            # row_sums_alpha_fit) there is no need to further process row_sums_alpha, yet.
            row_sums_alpha = None

        return normalized_kernel, row_sums_alpha

    def _normalize(self, rbf_kernel, row_sums_alpha_fit, is_pdist):

        # only required for symmetric kernel, return None if not used
        basis_change_matrix = None

        # required if alpha>0 and _normalize is called later for a cdist case
        # set in the pdist, alpha > 0 case
        row_sums_alpha = None

        if self.is_stochastic:

            if self.alpha > 0:
                # if pdist: kernel is still symmetric after this function call
                (
                    rbf_kernel,
                    row_sums_alpha,
                ) = self._normalize_sampling_density_kernel_matrix(
                    rbf_kernel, row_sums_alpha_fit
                )

            if self.is_symmetric_transform(is_pdist):
                # increases numerical stability when solving the eigenproblem
                # Note1: when using the (symmetric) conjugate matrix, the eigenvectors
                #        have to be transformed back to match the original
                # Note2: the similarity transform only works for the is_pdist case
                #        (for cdist, there is no symmetric kernel in the first place,
                #        because it is generally rectangular and does not include self
                #        points)
                rbf_kernel, basis_change_matrix = conjugate_stochastic_kernel_matrix(
                    rbf_kernel, None
                )
            else:
                rbf_kernel = stochastic_kernel_matrix(rbf_kernel)

            assert (
                self.is_symmetric_transform(is_pdist)
                and basis_change_matrix is not None
            ) or (
                not self.is_symmetric_transform(is_pdist)
                and basis_change_matrix is None
            )

        if is_pdist and self.is_symmetric:
            assert is_symmetric_matrix(rbf_kernel)

        return rbf_kernel, basis_change_matrix, row_sums_alpha

    def is_symmetric_transform(self, is_pdist):
        # If the kernel is made stochastic, it looses the symmetry, if symmetric_kernel
        # is set to True, then apply the the symmetry transformation
        return is_pdist and self.is_stochastic and self.is_symmetric

    def _read_kernel_kwargs(self, kernel_kwargs, is_pdist):
        row_sums_alpha_fit = None
        if self.is_stochastic and self.alpha > 0 and not is_pdist:
            row_sums_alpha_fit = kernel_kwargs.pop("row_sums_alpha_fit", None)

            if row_sums_alpha_fit is None:
                raise RuntimeError(
                    "cdist cannot be carried out, if no "
                    "kernel_kwargs['row_sums_alpha_fit'] is given. "
                    "Please report bug."
                )
        return row_sums_alpha_fit

    def __call__(
        self,
        X,
        Y=None,
        dist_cut_off=None,
        dist_backend="guess_optimal",
        kernel_kwargs=None,
        dist_backend_kwargs=None,
    ):

        if kernel_kwargs is None:
            kernel_kwargs = {}

        rbf_kernel_matrix = self._internal_rbf_kernel(
            X,
            Y=Y,
            dist_cut_off=dist_cut_off,
            dist_backend=dist_backend,
            kernel_kwargs=kernel_kwargs,
            dist_backend_kwargs=dist_backend_kwargs,
        )

        is_pdist = Y is None

        row_sums_alpha_fit = self._read_kernel_kwargs(
            kernel_kwargs=kernel_kwargs, is_pdist=is_pdist
        )

        kernel_matrix, basis_change_matrix, row_sums_alpha = self._normalize(
            rbf_kernel_matrix, row_sums_alpha_fit, is_pdist
        )

        return (
            kernel_matrix,
            basis_change_matrix,  # is None if not required for back-transformation
            row_sums_alpha,  # is None for cdist or (pdist and alpha==0)
        )

    def eval(self, distance_matrix, is_pdist=False, **kernel_kwargs):

        row_sums_alpha_fit = self._read_kernel_kwargs(kernel_kwargs, is_pdist)

        if is_pdist and row_sums_alpha_fit is not None:
            raise ValueError(
                "If is_pdist=True then no row_sum_alpha_fit should be " "provided"
            )

        rbf_kernel_matrix = self._internal_rbf_kernel.eval(distance_matrix)
        kernel_matrix, basis_change_matrix, row_sums_alpha = self._normalize(
            rbf_kernel_matrix, row_sums_alpha_fit=row_sums_alpha_fit, is_pdist=is_pdist
        )

        return kernel_matrix, basis_change_matrix, row_sums_alpha

    def gradient(self, X, X_eval):
        if self.is_stochastic:
            raise NotImplementedError(
                "Not implemented, also not sure how it works with stochastic and "
                "renormalization"
            )
        else:
            return self._internal_rbf_kernel._gradient(X, X_eval)

    def diag(self, X):
        """Implementing abstract function, required by the Kernel interface."""
        if not self.is_stochastic:
            return self._internal_rbf_kernel.diag(X)
        else:
            # TODO: Most likely it is required to compute the full kernel self(X, X) and
            #  then take the diag. However, there are more options required now (such
            #  as cut_off, etc.)
            raise NotImplementedError(
                "This case a bit more complicated. Also not sure if this is required."
            )


class DmapKernelVariable(StationaryKernelMixin, PCManifoldKernelMixin, Kernel):
    def __init__(self, epsilon, k, expected_dim, beta, symmetrize_kernel):
        if expected_dim <= 0 and not is_integer(expected_dim):
            raise ValueError("expected_dim has to be a non-negative integer.")

        if epsilon < 0 and not not is_float(expected_dim):
            raise ValueError("epsilon has to be positive float.")

        if k <= 0 and not is_integer(expected_dim):
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
        c2 = (
            1 / 2
            - 2 * self.alpha
            + 2 * self.expected_dim * self.alpha
            + self.expected_dim * self.beta / 2
            + self.beta
        )

        if c2 >= 0:
            raise ValueError(
                "Theory requires c2 to be negative:\n"
                "c2 = 1/2 - 2 * alpha + 2 * expected_dim * alpha + expected_dim * "
                f"beta/2 + beta \n but is {c2}"
            )

    def is_symmetric_transform(self, is_pdist):
        # If the kernel is made stochastic, it looses the symmetry, if symmetric_kernel
        # is set to True, then apply the the symmetry transformation
        return is_pdist and self.is_symmetric

    def _compute_rho0(self, distance_matrix):
        """Ad hoc bandwidth function."""
        nr_samples = distance_matrix.shape[1]

        # both modes are equivalent, MODE=1 allows to easier compare with ref3.
        MODE = 1
        if MODE == 1:  # according to Berry code
            # keep only nearest neighbors
            distance_matrix = np.sort(np.sqrt(distance_matrix), axis=1)[
                :, 1 : self.k + 1
            ]
            rho0 = np.sqrt(np.mean(distance_matrix ** 2, axis=1))
        else:  # MODE == 2:  , more performant if required
            if self.k < nr_samples:
                # TODO: have to revert setting the inf
                #   -> check that the diagonal is all zeros
                #   -> set to inf
                #   -> after computation set all infs back to zero
                #   -> this is also very similar to continous-nn in PCManifold

                # this allows to ignore the trivial distance=0 to itself
                np.fill_diagonal(distance_matrix, np.inf)

                # more efficient than sorting, everything
                distance_matrix.partition(self.k, axis=1)
                distance_matrix = np.sort(distance_matrix[:, : self.k], axis=1)
                distance_matrix = distance_matrix * distance_matrix
            else:  # self.k == self.N
                np.fill_diagonal(distance_matrix, np.nan)
                bool_mask = ~np.diag(np.ones(nr_samples)).astype(np.bool)
                distance_matrix = distance_matrix[bool_mask].reshape(
                    distance_matrix.shape[0], distance_matrix.shape[1] - 1
                )

            # experimental: --------------------------------------------------------------
            # paper: in var-bw paper (ref2) pdfp. 7
            # it is mentioned to IGNORE non-zero entries -- this is not detailed more.
            # a consequence is that the NN and kernel looses symmetry, so do (K+K^T) / 2
            # This is with a cut off rate:
            # val = 1E-2
            # distance_matrix[distance_matrix < val] = np.nan
            # experimental END -----------------------------------------------------------

            # nanmean only for the experimental part, if leaving this out, np.mean
            # suffices
            rho0 = np.sqrt(np.nanmean(distance_matrix, axis=1))

        return rho0

    def _compute_q0(self, distance_matrix, rho0):
        """The sampling density."""

        meanrho0 = np.mean(rho0)
        rho0tilde = rho0 / meanrho0

        # TODO: eps0 could also be optimized (see Berry Code + paper ref2)
        eps0 = meanrho0 ** 2

        expon_matrix = _symmetric_matrix_division(
            matrix=-distance_matrix, vec=rho0tilde, scalar=2 * eps0
        )

        nr_samples = distance_matrix.shape[0]

        # according to eq. (10) in ref1
        q0 = (
            np.power(2 * np.pi * eps0, -self.expected_dim / 2)
            / (np.power(rho0, self.expected_dim) * nr_samples)
            * np.sum(np.exp(expon_matrix), axis=1)
        )

        return q0

    def _compute_rho(self, q0):
        """The bandwidth function for K_eps_s"""
        rho = np.power(q0, self.beta)

        # Division by rho-mean is not in papers, but in berry code (ref3)
        return rho / np.mean(rho)

    def _compute_kernel_eps_s(self, distance_matrix, rho):
        expon_matrix = _symmetric_matrix_division(
            matrix=distance_matrix, vec=rho, scalar=-4 * self.epsilon
        )
        kernel_eps_s = np.exp(expon_matrix, out=expon_matrix)
        return kernel_eps_s

    def _compute_q_eps_s(self, kernel_eps_s, rho):
        rho_power_dim = np.power(rho, self.expected_dim)[:, np.newaxis]
        q_eps_s = np.sum(kernel_eps_s / rho_power_dim, axis=1)
        return q_eps_s

    def _compute_kernel_eps_alpha_s(self, kernel_eps_s, q_eps_s):
        kernel_eps_alpha_s = _symmetric_matrix_division(
            matrix=kernel_eps_s, vec=np.power(q_eps_s, self.alpha)
        )
        assert is_symmetric_matrix(kernel_eps_alpha_s)

        return kernel_eps_alpha_s

    def _compute_matrix_l(self, kernel_eps_alpha_s, rho):
        rhosq = np.square(rho)[:, np.newaxis]
        n_samples = rho.shape[0]
        matrix_l = (kernel_eps_alpha_s - np.eye(n_samples)) / (self.epsilon * rhosq)
        return matrix_l

    def _compute_matrix_s_inv(self, rho, q_eps_alpha_s):
        s_diag = np.reciprocal(rho * np.sqrt(q_eps_alpha_s))
        return scipy.sparse.diags(s_diag)

    def _compute_matrix_l_conjugate(self, kernel_eps_alpha_s, rho, q_eps_alpha_s):

        basis_change_matrix = self._compute_matrix_s_inv(rho, q_eps_alpha_s)

        p_sq_inv = scipy.sparse.diags(np.reciprocal(np.square(rho)))

        matrix_l_hat = (
            basis_change_matrix @ kernel_eps_alpha_s @ basis_change_matrix
            - (p_sq_inv - scipy.sparse.diags(np.ones(kernel_eps_alpha_s.shape[0])))
        )

        matrix_l_hat = remove_numeric_noise_symmetric_matrix(matrix_l_hat)

        return matrix_l_hat, basis_change_matrix

    def __call__(
        self,
        X,
        Y=None,
        dist_cut_off=None,
        dist_backend="guess_optimal",
        kernel_kwargs=None,
        dist_backend_kwargs=None,
    ):

        if dist_cut_off is not None and not np.isinf(dist_cut_off):
            raise NotImplementedError("Handling sparsity is currently not implemented!")

        if Y is not None:
            raise NotImplementedError("cdist case is currently not implemented!")

        if self.k > X.shape[0]:
            raise ValueError(
                f"nr of nearest neighbors (self.k={self.k}) "
                f"is larger than number of samples (={X.shape[0]})"
            )

        distance_matrix = compute_distance_matrix(
            X,
            Y,
            metric="sqeuclidean",
            cut_off=dist_cut_off,
            backend=dist_backend,
            **dist_backend_kwargs,
        )

        operator_l_matrix, basis_change_matrix, rho0, rho, q0, q_eps_s = self.eval(
            distance_matrix
        )

        # TODO: make a return_vectors option?
        return (
            operator_l_matrix,
            basis_change_matrix,
            rho0,
            rho,
            q0,
            q_eps_s,
        )

    def eval(self, distance_matrix):
        if scipy.sparse.issparse(distance_matrix):
            raise NotImplementedError(
                "Currently the variable bandwidth kernel is only implemented for the "
                "dense distance matrix case."
            )

        assert (
            is_symmetric_matrix(distance_matrix)
            and (np.diag(distance_matrix) == 0).all()
        ), "only pdist case supported at the moment"

        rho0 = self._compute_rho0(distance_matrix)
        q0 = self._compute_q0(distance_matrix, rho0)
        rho = self._compute_rho(q0)
        kernel_eps_s = self._compute_kernel_eps_s(distance_matrix, rho)
        q_eps_s = self._compute_q_eps_s(kernel_eps_s, rho)
        kernel_eps_alpha_s = self._compute_kernel_eps_alpha_s(kernel_eps_s, q_eps_s)

        if self.is_symmetric:
            q_eps_alpha_s = kernel_eps_alpha_s.sum(axis=1)
            operator_l_matrix, basis_change_matrix = self._compute_matrix_l_conjugate(
                kernel_eps_alpha_s, rho, q_eps_alpha_s
            )
        else:
            basis_change_matrix = None
            operator_l_matrix = self._compute_matrix_l(kernel_eps_alpha_s, rho)

        return (
            operator_l_matrix,
            basis_change_matrix,
            rho0,
            rho,
            q0,
            q_eps_s,
        )

    def diag(self, X):
        """Implementing abstract function, required by the Kernel interface."""
        raise NotImplementedError(
            "This case a bit more complicated. Also not sure if this is required."
        )


def _symmetric_matrix_division(matrix, vec, vec_right=None, scalar=None):
    r"""
    Solves a symmetric division of matrix and vector:

    .. math::
        \frac{M_{i, j}}{a v_i v_j}

    where :math:`M` is the matrix, and elements :math:`v` is the
    vector and :math:`a` is a scalar used for the division.

    This operation appears often in kernel based methods.

    Parameters
    ----------
    matrix
        matrix to apply symmetric division on
    vec
        vector in the denominator
        .. note::
            The reciprocal is taken inside the function.
    scalar
        scalar in the denominator
        .. note::
            The reciprocal is taken inside the function.
    Returns
    -------
    matrix

    """

    assert vec.ndim == 1

    vec_inv_left = np.reciprocal(vec)

    if vec_right is None:
        vec_inv_right = vec_inv_left.view()
    else:
        vec_inv_right = np.reciprocal(vec_right)

    if scipy.sparse.issparse(matrix):
        left_inv_diag_sparse = scipy.sparse.spdiags(
            vec_inv_left, 0, m=matrix.shape[0], n=matrix.shape[0]
        )
        right_inv_diag_sparse = scipy.sparse.spdiags(
            vec_inv_right, 0, m=matrix.shape[1], n=matrix.shape[1]
        )
        # TODO: not sure if scipy makes this efficiently?
        matrix = left_inv_diag_sparse @ matrix @ right_inv_diag_sparse
    else:
        # Solves efficiently:
        # np.diag(1/vector_elements) @ matrix @ np.diag(1/vector_elements)
        matrix = diagmat_dot_mat(vec_inv_left, matrix, out=matrix)
        matrix = mat_dot_diagmat(matrix, vec_inv_right, out=matrix)

    # sparse and dense
    if vec_right is None:
        matrix = remove_numeric_noise_symmetric_matrix(matrix)

    if scalar is not None:
        scalar = 1 / scalar
        matrix = np.multiply(matrix, scalar, out=matrix)
    return matrix


def conjugate_stochastic_kernel_matrix(
    kernel_matrix: Union[np.ndarray, scipy.sparse.spmatrix], row_sums_fit
) -> Tuple[Union[np.ndarray, scipy.sparse.spmatrix], scipy.sparse.dia_matrix]:

    """Make the kernel matrix symmetric, the spectrum is kept because of a similarity
       transformation. See e.g. [TODO: Lafon PhD] or [Rabin et. al]

       In [TODO: Rabin et. al] equation 3.1. states that (adapted to notation used here):

       .. math::
            P = D^{-1} K

        where :math:`D^{-1}` is the standard row normalization. Then equation 3.3
        states that :math:`P` has a similar matrix with

       .. math::
           A = D^{1/2} P D^{-1/2}

       Replacing :math:`P` from above we get:

       .. math::
          A = D^{1/2} D^{-1} K D^{-1/2}
          A = D^{-1/2} K D^{-1/2}

       This last operation is applied in this function. The matrix :math:`A` then has
       the same eigenvalues to :math:`P` and the eigenvectors can be recovered with
       equation 3.4. For the right eigenvecotors:

       .. math::
           \Psi = D^{-1/2} V

       where :math:`V` are the eigenvectors of :math:`A` and :math:`\Psi` from
       :math:`P`.

       .. note::
       The conjugate-stochastic matrix is not stochastic in the strict sense
       (i.e. the row-sum is not equal to 1). The eigenpairs (after a transformation
       'basis_change_matrix') of the conjugate_stochastic are the same
       to the stochastic matrix.

       """

    left_vec = kernel_matrix.sum(axis=1)

    if scipy.sparse.issparse(kernel_matrix):
        # to np.ndarray in case it is depricated format np.matrix
        left_vec = left_vec.A1

    left_vec = np.sqrt(left_vec, out=left_vec)

    kernel_matrix = _symmetric_matrix_division(
        kernel_matrix, vec=left_vec, vec_right=row_sums_fit
    )
    # TODO: maybe let _symmetric_matrix_division return the reciprocal left_vec /
    #  right_vec
    basis_change_matrix = scipy.sparse.diags(np.reciprocal(left_vec, out=left_vec))

    return kernel_matrix, basis_change_matrix


def stochastic_kernel_matrix(kernel_matrix):
    """Function to make (sparse/dense) kernel stochastic."""
    if scipy.sparse.issparse(kernel_matrix):
        # see microbenchmark_stochastic_matrix.py, for the sparse case this variant is
        # the fastest variant)
        kernel_matrix = normalize(kernel_matrix, copy=False, norm="l1")
    else:  # dense

        normalize_diagonal = np.sum(kernel_matrix, axis=1)

        with np.errstate(divide="ignore", over="ignore"):
            # especially in cdist there can be far away outliers (or very small
            # scale/epsilon), such that elements are near 0
            #  the reciprocal can then
            #     - be inf
            #     - overflow (resulting in negative values)
            #  these cases are catched with bool_invalid below
            normalize_diagonal = np.reciprocal(
                normalize_diagonal, out=normalize_diagonal
            )

        bool_invalid = np.logical_or(
            np.isinf(normalize_diagonal), normalize_diagonal < 0
        )
        normalize_diagonal[bool_invalid] = 0
        kernel_matrix = diagmat_dot_mat(normalize_diagonal, kernel_matrix)

    return kernel_matrix
