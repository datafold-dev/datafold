import abc
from typing import Tuple, Union

import numexpr as ne
import numpy as np
import scipy.sparse
import scipy.spatial
from sklearn.gaussian_process.kernels import (
    RBF,
    Kernel,
    NormalizedKernelMixin,
    StationaryKernelMixin,
    _check_length_scale,
)
from sklearn.preprocessing import normalize

from datafold.decorators import warn_experimental_class
from datafold.pcfold.distance import compute_distance_matrix
from datafold.utils.datastructure import is_integer
from datafold.utils.maths import (
    diagmat_dot_mat,
    is_symmetric_matrix,
    mat_dot_diagmat,
    remove_numeric_noise_symmetric_matrix,
)


def apply_kernel_function(distance_matrix, kernel_function):
    if scipy.sparse.issparse(distance_matrix):
        kernel = distance_matrix
        # NOTE: applies on stored data, it is VERY important, that real distance zeros are
        # included in 'distance_matrix' (E.g. normalized kernels have to have a 1.0 on
        # the diagonal) are included in the sparse matrix!
        kernel.data = kernel_function(kernel.data)
    else:
        kernel = kernel_function(distance_matrix)

    return kernel


def apply_kernel_function_numexpr(distance_matrix, expr, expr_dict):

    assert "D" not in expr_dict.keys()

    if scipy.sparse.issparse(distance_matrix):
        # copy because the distance matrix may be used further by the user
        distance_matrix = distance_matrix.copy()
        expr_dict["D"] = distance_matrix.data
        ne.evaluate(expr, expr_dict, out=distance_matrix.data)
        return distance_matrix
    else:
        expr_dict["D"] = distance_matrix
        return ne.evaluate(expr, expr_dict)


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


class PCManifoldKernelMixin(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __call__(
        self,
        X,
        Y=None,
        dist_cut_off=None,
        dist_backend="brute",
        kernel_kwargs=None,
        dist_backend_kwargs=None,
    ):
        """Evaluation calls for kernels used in PCManifold"""

    @abc.abstractmethod
    def eval(self, distance_matrix):
        """Evaluate kernel on an already computed distance matrix. Note: there are no
        checks whether the correct kernel metric was used. 'distance_matrix' may be
        sparse or dense. For the sparse case note that it acts on all stored
        data, i.e. "real zeros" by distance have to be stored."""


class RadialBasisKernel(RBF, PCManifoldKernelMixin):
    """Overwrites selected functions of sklearn.RBF in order to use sparse distance
    matrix computations."""

    def __init__(self, epsilon=1.0, length_scale_bounds=(1e-5, 1e5)):
        self._epsilon = epsilon

        # sqrt because of slightly different notation of the kernel
        super(RadialBasisKernel, self).__init__(
            length_scale=np.sqrt(epsilon), length_scale_bounds=length_scale_bounds
        )

    @property
    def epsilon(self):
        return self._epsilon

    @epsilon.setter
    def epsilon(self, value):
        self._epsilon = value
        # keep aligned with super class from scikit learn
        self.length_scale = np.sqrt(self.epsilon)

    def __call__(
        self,
        X,
        Y=None,
        eval_gradient=False,
        dist_cut_off=None,
        dist_backend="brute",
        kernel_kwargs=None,
        dist_backend_kwargs=None,
    ):
        # TODO: only this kernel so far has "eval_gradient" -- should that go into the
        #  general interface?
        # TODO: maybe remove all default parameters from the kernel interface!

        X = np.atleast_2d(X)

        if Y is not None:
            Y = np.atleast_2d(Y)

        if dist_backend_kwargs is None:
            dist_backend_kwargs = {}

        _check_length_scale(X, self.length_scale)  # sklearn function

        distance_matrix = compute_distance_matrix(
            X,
            Y,
            metric="sqeuclidean",
            cut_off=dist_cut_off,
            backend=dist_backend,
            **dist_backend_kwargs,
        )

        if eval_gradient and scipy.sparse.issparse(distance_matrix):
            raise NotImplementedError(
                "Gradient is not implemented for sparse distance matrix"
            )

        kernel_matrix = self.eval(distance_matrix)

        if not eval_gradient:
            return kernel_matrix
        else:

            # TODO: check gradient code (are both correct, what are the differences, link
            #  to comp. explanations, ...):
            if Y is None:
                # TODO: used from super class sklean.RBF kernel
                return kernel_matrix, self._gradient(distance_matrix, kernel_matrix)
            else:
                # TODO: used from kernels.py code (and adapted)
                return kernel_matrix, self._gradient_given_Y(X, Y, kernel_matrix)

    def eval(self, distance_matrix):
        # Security copy, the distance matrix is maybe required again (for gradient,
        # or other computations...)

        return apply_kernel_function_numexpr(
            distance_matrix,
            expr="exp((- 1 / (2*eps)) * D)",
            expr_dict={"eps": self._epsilon},
        )

    def _gradient(self, distance_matrix, kernel_matrix=None):

        # NOTE: Copied code from the super class RBF
        if self.hyperparameter_length_scale.fixed:
            # Hyperparameter l kept fixed
            return np.empty((kernel_matrix.shape[0], kernel_matrix.shape[0], 0))
        elif not self.anisotropic or self.length_scale.shape[0] == 1:
            kernel_gradient = kernel_matrix * distance_matrix[:, :, np.newaxis]
            return kernel_gradient
        elif self.anisotropic:

            # We need to recompute the pairwise dimension-wise distances
            kernel_gradient = (
                kernel_matrix[:, np.newaxis, :] - kernel_matrix[np.newaxis, :, :]
            ) ** 2 / (self.length_scale ** 2)
            kernel_gradient *= kernel_matrix[..., np.newaxis]
            return kernel_gradient

    def _gradient_given_Y(self, X, Y, kernel_matrix):
        """ computes the gradient of the kernel w.r.t. the x argument """

        if scipy.sparse.issparse(kernel_matrix):
            raise NotImplementedError("Not implemented sparse kernel matrix version.")

        # this is very heavy on the memory. TODO: improve it.
        xmy = np.zeros((Y.shape[1], Y.shape[0], X.shape[0]))

        if Y.shape[1] == 1:
            raise ValueError("one dim. grad not implemented yet")
        else:
            for k in range(
                Y.shape[1]
            ):  # TODO: check if this loop can be optimized, probably slow
                xm = Y[:, k].T @ np.ones((1, X.shape[0]))
                ym = X[:, k].T @ np.ones((1, Y.shape[0])).T
                xmy[k, :, :] = xm - ym

        res = np.zeros((Y.shape[1], kernel_matrix.shape[0], kernel_matrix.shape[1]))

        for k in range(res.shape[0]):
            res[k, :, :] = -2 / self._epsilon * kernel_matrix * np.array(xmy[k, :])

        return res


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
        symmetrize_kernel=True,
    ):

        self.epsilon = epsilon
        # TODO: not sure if I need this length_scale_bounds?! Check in the scikit learn
        #  doc what this is about
        self.length_scale_bounds = length_scale_bounds

        self.is_stochastic = is_stochastic

        if not (0 <= alpha <= 1):
            raise ValueError(f"alpha has to be between [0, 1]. Got alpha={alpha}")
        self.alpha = alpha
        self.symmetrize_kernel = symmetrize_kernel

        self._internal_rbf_kernel = RadialBasisKernel(
            self.epsilon, length_scale_bounds=self.length_scale_bounds
        )

        if not self.is_stochastic or self.symmetrize_kernel:
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

        self.row_sums_init = None

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
                "Currently DmapKernelVariable is only implemented to handle a dense "
                "distance matrix."
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


@warn_experimental_class  # not used so far
class MultiquadraticKernel(
    StationaryKernelMixin, NormalizedKernelMixin, PCManifoldKernelMixin, Kernel
):
    def __init__(self, epsilon):
        self.epsilon = epsilon
        super(MultiquadraticKernel, self).__init__()

    def __call__(
        self,
        X,
        Y=None,
        dist_cut_off=None,
        dist_backend="brute",
        kernel_kwargs=None,
        dist_backend_kwargs=None,
    ):

        # TODO: check metric!
        distance_matrix = compute_distance_matrix(
            X,
            Y,
            metric="XXXX",
            cut_off=dist_cut_off,
            backend=dist_backend,
            **dist_backend_kwargs,
        )

        kernel_matrix = self.eval(distance_matrix)
        return kernel_matrix

    def eval(self, distance_matrix):
        kernel_func = lambda dist: np.sqrt(
            np.square(np.multiply(dist, 1.0 / self.epsilon)) + 1
        )
        kernel_matrix = apply_kernel_function(distance_matrix, kernel_func)
        return kernel_matrix


@warn_experimental_class  # not used so far
class InverseMultiQuadraticKernel(
    StationaryKernelMixin, NormalizedKernelMixin, PCManifoldKernelMixin, Kernel
):
    def __init__(self, epsilon):
        self.epsilon = epsilon
        super(InverseMultiQuadraticKernel, self).__init__()

    def __call__(
        self, X, Y=None, dist_cut_off=None, dist_backend="brute", **backend_options
    ):

        # TODO: check metric
        distance_matrix = compute_distance_matrix(
            X,
            Y,
            metric="XXXX",
            cut_off=dist_cut_off,
            backend=dist_backend,
            **backend_options,
        )

        kernel_matrix = self.eval(distance_matrix)
        return kernel_matrix

    def eval(self, distance_matrix):
        kernel_func = lambda dist: np.reciprocal(
            np.sqrt(np.square(np.multiply(dist, 1 / self.epsilon)) + 1)
        )
        kernel_matrix = apply_kernel_function(distance_matrix, kernel_func)
        return kernel_matrix


@warn_experimental_class  # not used so far
class InverseQuadraticKernel(PCManifoldKernelMixin, Kernel):
    def __init__(self, epsilon):
        self.epsilon = epsilon
        super(InverseQuadraticKernel, self).__init__()

    def __call__(
        self,
        X,
        Y=None,
        dist_cut_off=None,
        dist_backend="brute",
        kernel_kwargs=None,
        dist_backend_kwargs=None,
    ):

        # TODO: check metric
        distance_matrix = compute_distance_matrix(
            X,
            Y,
            metric="XXXX",
            cut_off=dist_cut_off,
            backend=dist_backend,
            **dist_backend_kwargs,
        )

        kernel_matrix = self.eval(distance_matrix)
        return kernel_matrix

    def eval(self, distance_matrix):
        kernel_func = lambda dist: np.clip(
            1 - np.square(1.0 / self.epsilon * dist), 0, 1
        )
        kernel_matrix = apply_kernel_function(distance_matrix, kernel_func)
        return kernel_matrix


class OUKernel(
    StationaryKernelMixin, NormalizedKernelMixin, PCManifoldKernelMixin, Kernel
):
    def __init__(self, epsilon):
        self.epsilon = epsilon
        super(OUKernel, self).__init__()

    def __call__(
        self,
        X,
        Y=None,
        dist_cut_off=None,
        dist_backend="brute",
        kernel_kwargs=None,
        dist_backend_kwargs=None,
    ):
        distance_matrix = compute_distance_matrix(
            X,
            Y,
            metric="euclidean",
            cut_off=dist_cut_off,
            backend=dist_backend,
            **dist_backend_kwargs,
        )

        kernel_matrix = self.eval(distance_matrix)
        return kernel_matrix

    def eval(self, distance_matrix):
        kernel_matrix = apply_kernel_function(
            distance_matrix, lambda x: np.exp(-x / self.epsilon)
        )
        return kernel_matrix


@warn_experimental_class  # not used so far
class LanczosKernel(NormalizedKernelMixin, PCManifoldKernelMixin, Kernel):
    # TODO: check other kernels

    def __init__(self, epsilon=1, width=2):
        self.epsilon = epsilon
        self.width = width
        super(LanczosKernel, self).__init__()

    def __call__(
        self, X, Y=None, dist_cut_off=None, dist_backend="brute", **backend_options
    ):

        # TODO: check if the metric is correct
        distance_matrix = compute_distance_matrix(
            X,
            Y,
            metric="euclidean",
            cut_off=dist_cut_off,
            backend=dist_backend,
            **backend_options,
        )

        kernel_matrix = self.eval(distance_matrix)
        return kernel_matrix

    def eval(self, distance_matrix):
        bool_idx = distance_matrix < self.width
        kernel_matrix = bool_idx * (
            np.sinc(distance_matrix / self.epsilon)
            * np.sinc(distance_matrix / (self.epsilon * self.width))
        )
        return kernel_matrix

    def is_stationary(self):
        return False  # TODO: check if the kernel is really not stationary
