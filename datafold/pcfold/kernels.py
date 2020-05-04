import abc
from typing import Dict, Optional, Tuple, Union

import numexpr as ne
import numpy as np
import scipy.sparse
import scipy.spatial
from scipy.special import xlogy
from sklearn.gaussian_process.kernels import (
    Kernel,
    NormalizedKernelMixin,
    StationaryKernelMixin,
)
from sklearn.preprocessing import normalize

from datafold.decorators import warn_experimental_class
from datafold.pcfold.distance import compute_distance_matrix
from datafold.utils.general import (
    diagmat_dot_mat,
    is_float,
    is_integer,
    is_symmetric_matrix,
    mat_dot_diagmat,
    remove_numeric_noise_symmetric_matrix,
)


def _apply_kernel_function(distance_matrix, kernel_function):
    if scipy.sparse.issparse(distance_matrix):
        kernel = distance_matrix
        # NOTE: applies on stored data, it is VERY important, that real distance zeros are
        # included in 'distance_matrix' (E.g. normalized kernels have to have a 1.0 on
        # the diagonal) are included in the sparse matrix!
        kernel.data = kernel_function(kernel.data)
    else:
        kernel = kernel_function(distance_matrix)

    return kernel


def _apply_kernel_function_numexpr(distance_matrix, expr, expr_dict=None):

    expr_dict = {} if expr_dict is None else expr_dict
    assert "D" not in expr_dict.keys()

    if scipy.sparse.issparse(distance_matrix):
        # copy because the distance matrix may be used further by the user
        distance_matrix = distance_matrix.copy()
        expr_dict["D"] = distance_matrix.data
        ne.evaluate(expr, expr_dict, out=distance_matrix.data)
        return distance_matrix  # returns actually the kernel
    else:
        expr_dict["D"] = distance_matrix
        return ne.evaluate(expr, expr_dict)


def _symmetric_matrix_division(
    matrix: Union[np.ndarray, scipy.sparse.spmatrix],
    vec: np.ndarray,
    vec_right: Optional[np.ndarray] = None,
    scalar: float = 1.0,
) -> np.ndarray:
    r"""Symmetric division, which often appears in kernels.

    .. math::
        \frac{M_{i, j}}{a v^(l)_i v^(r)_j}

    where :math:`M` is a (kernel-) matrix and its elements are divided by the
    (left and right) vector elements :math:`v` and scalar :math:`a`.

    .. warning::
        The function is in-place and may therefore overwrite the matrix. Make a copy
        beforehand if the old values are still required.

    Parameters
    ----------
    matrix
        Matrix of shape `(n_rows, n_columns)` to apply symmetric division on.
        If matrix is square and ``vec_right=None``, then `matrix` is assumed to be
        symmetric (this enables removing numerical noise to return a perfectly symmetric
        matrix).
        
    vec
        Vector of shape `(n_rows,)` in the denominator (Note, the reciprocal
        is is internal of the function).
        
    vec_right
        Vector of shape `(n_columns,)`. If matrix is non-square or matrix input is
        not symmetric, then this input is required. If None, it is set
        to ``vec_right=vec``.

    scalar
        Scalar ``a`` in the denominator.

    Returns
    -------
    numpy.ndarray
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

        # See file mircobenchmark_sparse_outer_division.py:
        # The performance of DIA-sparse matrices is good if the matrix is actually
        # sparse. I.e. the performance drops for a dense-sparse matrix. There is another
        # implementation in the mircrobenchmark that also handles this case well,
        # however, it requires numba.
        matrix = left_inv_diag_sparse @ matrix @ right_inv_diag_sparse
    else:
        # Solves efficiently:
        # np.diag(1/vector_elements) @ matrix @ np.diag(1/vector_elements)
        matrix = diagmat_dot_mat(vec_inv_left, matrix, out=matrix)
        matrix = mat_dot_diagmat(matrix, vec_inv_right, out=matrix)

    # sparse and dense
    if vec_right is None:
        matrix = remove_numeric_noise_symmetric_matrix(matrix)

    if scalar != 1.0:
        scalar = 1 / scalar
        matrix = np.multiply(matrix, scalar, out=matrix)
    return matrix


def _conjugate_stochastic_kernel_matrix(
    kernel_matrix: Union[np.ndarray, scipy.sparse.spmatrix]
) -> Tuple[Union[np.ndarray, scipy.sparse.spmatrix], scipy.sparse.dia_matrix]:
    r"""Conjugate transformation to obtain symmetric (conjugate) kernel matrix with same
    spectrum properties.

    In Rabin et al. :cite:`rabin_heterogeneous_2012` (Eq. 3.1) states that \
    (notation adapted):

    .. math::
        P = D^{-1} K

    where :math:`D^{-1}` is the standard row normalization. Eq. 3.3 shows that matrix \
    :math:`P` has a similar matrix with

    .. math::
       A = D^{1/2} P D^{-1/2}

    Replacing :math:`P` from above we get:

    .. math::
        A = D^{1/2} D^{-1} K D^{-1/2}
        A = D^{-1/2} K D^{-1/2}

    Where the last equation is the conjugate transformation performed in this function.
    The matrix :math:`A` has the same eigenvalues to :math:`P` and the eigenvectors
    can be recovered (Eq. 3.4. in reference):

    .. math::
       \Psi = D^{-1/2} V

    where :math:`V` are the eigenvectors of :math:`A` and :math:`\Psi` from matrix \
    :math:`P`.

    .. note::
        The conjugate-stochastic matrix is not stochastic, but still has the trivial
        eigenvalue 1 (i.e. the row-sums are not equal to 1).

    Parameters
    ----------
    kernel_matrix
        non-symmetric kernel matrix

    Returns
    -------
    Tuple[Union[np.ndarray, scipy.sparse.spmatrix], scipy.sparse.dia_matrix]
        conjugate matrix (tpye as `kernel_matrix`) and (sparse) diagonal matrix to recover
        eigenvectors

    References
    ----------
    :cite:`rabin_heterogeneous_2012`

    """

    left_vec = kernel_matrix.sum(axis=1)

    if scipy.sparse.issparse(kernel_matrix):
        # to np.ndarray in case it is depricated format np.matrix
        left_vec = left_vec.A1

    left_vec = np.sqrt(left_vec, out=left_vec)

    kernel_matrix = _symmetric_matrix_division(
        kernel_matrix, vec=left_vec, vec_right=None
    )
    # TODO: maybe let _symmetric_matrix_division return the reciprocal left_vec /
    #  right_vec
    basis_change_matrix = scipy.sparse.diags(np.reciprocal(left_vec, out=left_vec))

    return kernel_matrix, basis_change_matrix


def _stochastic_kernel_matrix(kernel_matrix: Union[np.ndarray, scipy.sparse.spmatrix]):
    """Normalizes matrix rows.

    This function performs

    .. math::
        M = D^{-1} K

    where matrix :math:`M` is the row-normalized kernel from :math:`K` by the
    matrix :math:`D` with the row sums of :math:`K` on the diagonal.

    .. note::
        If the kernel matrix is evaluated component wise (points compared to reference
        points), then outliers can have a row sum close to zero. In this case the
        respective element on the diagonal is set to zero. For a pairwise kernel
        (pdist) this can not happen, as the diagonal element must be non-zero.

    Parameters
    ----------
    kernel_matrix
        kernel matrix (square or rectangular) to normalize

    Returns
    -------
    Union[np.ndarray, scipy.sparse.spmatrix]
        normalized kernel matrix with type same as `kernel_matrix`
    """
    if scipy.sparse.issparse(kernel_matrix):
        # see microbenchmark_stochastic_matrix.py, for the sparse case this variant is
        # the fastest)
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


class PCManifoldKernel(Kernel):
    """Abstract base class for kernels used in datafold.

    See Also
    --------

    :py:class:`PCManifold`
    """

    @abc.abstractmethod
    def __call__(
        self,
        X: np.ndarray,
        Y: Optional[np.ndarray] = None,
        dist_cut_off: Optional[float] = None,
        dist_backend: str = "brute",
        kernel_kwargs: Optional[Dict[str, object]] = None,
        dist_backend_kwargs: Optional[Dict[str, object]] = None,
    ) -> np.ndarray:
        """Compute the kernel matrix.
        
        If `Y` is given, the kernel matrix is computed component-wise, and if `Y=None`
        the kernel matrix is computed pairs-wise.
        
        Parameters
        ----------
        X
            data with shape `(n_samples, n_features)`

        Y
            reference data with shape `(n_samples_y, n_features_y)`

        dist_cut_off
            cut off distance

        dist_backend
            backend of distance algorithm

        kernel_kwargs
            keyword arguments for the kernel algorithm

        dist_backend_kwargs
            keyword arguments for the distance algorithm

        Returns
        -------
        np.ndarray
            kernel matrix with shape `(n_samples, n_samples)` (if `Y is None`) or
            `(n_samples_y, n_samples)` if `Y it not None`
        """

        raise NotImplementedError("base class")

    @abc.abstractmethod
    def eval(
        self, distance_matrix: Union[np.ndarray, scipy.sparse.csr_matrix]
    ) -> Union[np.ndarray, scipy.sparse.csr_matrix]:
        """Evaluate kernel on pre-computed distance matrix.

        .. note::
            There are no checks whether the correct kernel metric was used.

        Parameters
        ----------

        distance_matrix
            distance matrix with shape `(n_samples, n_samples)`. For the sparse case note
            that the kernel acts on all stored data, i.e. usually real distance zeros
            must be stored in the matrix and only very large distance values (resulting
            in small kernel values) should not be stored.
        
        Returns
        -------
        numpy.ndarray, scipy.sparse.csr_matrix
            kernel matrix with same type and shape as `distance_matrix`

        """
        raise NotImplementedError("base class")

    def diag(self, X):
        """(Not implemented, not used in datafold)

        Raises
        ------
        NotImplementedError
            this is only to overwrite abstract method in super class
        """

        raise NotImplementedError("base class")

    def is_stationary(self):
        """(Not implemented, not used in datafold)

        Raises
        ------
        NotImplementedError
            this is only to overwrite abstract method in super class
        """

        # in datafold there is no handling of this attribute, if required this has to
        # be implemented
        raise NotImplementedError("base class")


class RadialBasisKernel(PCManifoldKernel):
    """Abstract base class for radial basis kernels.

    "A radial basis function (RBF) is a real-valued function whose value depends \
    only on the distance between the input and some fixed point." from `Wikipedia <https://en.wikipedia.org/wiki/Radial_basis_function>`_


    Parameters
    ----------
    distance_metric
        distance metric required in the kernel
    """

    def __init__(self, distance_metric):

        self.distance_metric = distance_metric

    def __call__(
        self,
        X,
        Y=None,
        dist_cut_off=None,
        dist_backend="brute",
        kernel_kwargs=None,
        dist_backend_kwargs=None,
    ):
        X = np.atleast_2d(X)

        if Y is not None:
            Y = np.atleast_2d(Y)

        distance_matrix = compute_distance_matrix(
            X,
            Y,
            metric=self.distance_metric,
            cut_off=dist_cut_off,
            backend=dist_backend,
            **({} if dist_backend_kwargs is None else dist_backend_kwargs),
        )

        kernel_matrix = self.eval(distance_matrix)

        return kernel_matrix


class GaussianKernel(RadialBasisKernel):
    r"""Gaussian radial basis kernel.

    .. math::
        K = \exp(\frac{-1}{2\varepsilon} \cdot D)

    where :math:`D` is the squared euclidean distance matrix.

    Parameters
    ----------
    epsilon
        kernel scale
    """

    def __init__(self, epsilon=1.0):
        self.epsilon = epsilon
        super(GaussianKernel, self).__init__(distance_metric="sqeuclidean")

    def eval(
        self, distance_matrix: Union[np.ndarray, scipy.sparse.csr_matrix]
    ) -> Union[np.ndarray, scipy.sparse.csr_matrix]:
        # Security copy, the distance matrix is maybe required again (for gradient,
        # or other computations...)

        return _apply_kernel_function_numexpr(
            distance_matrix,
            expr="exp((- 1 / (2*eps)) * D)",
            expr_dict={"eps": self.epsilon},
        )


class MultiquadricKernel(RadialBasisKernel):
    r"""Multiquadric radial basis kernel.

    .. math::
        K = \sqrt(\frac{1}{2\varepsilon} \cdot D + 1)
        

    where :math:`D` is the squared euclidean distance matrix.

    Parameters
    ----------
    epsilon
        kernel scale
    """

    def __init__(self, epsilon=1.0):
        self.epsilon = epsilon
        super(MultiquadricKernel, self).__init__(distance_metric="sqeuclidean")

    def eval(
        self, distance_matrix: Union[np.ndarray, scipy.sparse.csr_matrix]
    ) -> Union[np.ndarray, scipy.sparse.csr_matrix]:
        return _apply_kernel_function_numexpr(
            distance_matrix,
            expr="sqrt(1.0 / (2*eps) * D + 1.0)",
            expr_dict={"eps": self.epsilon},
        )


class InverseMultiquadricKernel(RadialBasisKernel):
    r"""Inverse multiquadric radial basis kernel.

    .. math::
        K = \sqrt(\frac{1}{2\varepsilon} \cdot D + 1)^{-1}


    where :math:`D` is the squared euclidean distance matrix.

    Parameters
    ----------
    epsilon
        kernel scale
    """

    def __init__(self, epsilon=1.0):
        self.epsilon = epsilon
        super(InverseMultiquadricKernel, self).__init__(distance_metric="sqeuclidean")

    def eval(
        self, distance_matrix: Union[np.ndarray, scipy.sparse.csr_matrix]
    ) -> Union[np.ndarray, scipy.sparse.csr_matrix]:
        return _apply_kernel_function_numexpr(
            distance_matrix,
            expr="1.0 / sqrt(1.0 / (2*eps) * D + 1.0) ",
            expr_dict={"eps": self.epsilon},
        )


class CubicKernel(RadialBasisKernel):
    r"""Cubic radial basis kernel.

    .. math::
        K= D^{3}


    where :math:`D` is the euclidean distance matrix.
    """

    def __init__(self):
        super(CubicKernel, self).__init__(distance_metric="euclidean")

    def eval(
        self, distance_matrix: Union[np.ndarray, scipy.sparse.csr_matrix]
    ) -> Union[np.ndarray, scipy.sparse.csr_matrix]:
        # return r ** 3
        return _apply_kernel_function_numexpr(distance_matrix, expr="D ** 3")


class QuinticKernel(RadialBasisKernel):
    r"""Quintic radial basis kernel.

    .. math::
        K= D^{5}


    where :math:`D` is the euclidean distance matrix.
    """

    def __init__(self):
        super(QuinticKernel, self).__init__(distance_metric="euclidean")

    def eval(
        self, distance_matrix: Union[np.ndarray, scipy.sparse.csr_matrix]
    ) -> Union[np.ndarray, scipy.sparse.csr_matrix]:
        # r**5
        return _apply_kernel_function_numexpr(distance_matrix, "D ** 5")


class ThinPlateKernel(RadialBasisKernel):
    r"""Thin plate radial basis kernel.

    .. math::
        K = xlogy(D^2, D)


    where :math:`D` is the euclidean distance matrix and argument for
    :class:`scipy.special.xlogy`
    """

    def __init__(self):
        super(ThinPlateKernel, self).__init__(distance_metric="euclidean")

    def eval(self, distance_matrix: np.ndarray) -> np.ndarray:
        """Evaluate kernel on pre-computed distance matrix.

        Parameters
        ----------
        distance_matrix

        Returns
        -------
        numpy.ndarray
            kernel matrix with same type and shape as `distance_matrix`

        """
        return xlogy(np.square(distance_matrix), distance_matrix)


class DmapKernelFixed(PCManifoldKernel):
    """Diffusion maps kernel with fixed bandwidth of the internal Gaussian radial basis
    kernel.

    Parameters
    ----------

    epsilon
        Gaussian kernel scale

    is_stochastic
        If True, the kernel matrix is row-normalized.

    alpha
        Degree of re-normalization of sampling density in point cloud. `alpha` must be
        inside the interval [0, 1] (inclusive).

    symmetrize_kernel
        If True, performs a conjugate transformation which can improve numerical
        stability for operations (such as eigenpairs) of kernel matrix.


    See Also
    --------
    :py:class:`DiffusionMaps`

    References
    ----------

    :cite:`coifman_diffusion_2006`
    """

    def __init__(
        self,
        epsilon: float = 1.0,
        is_stochastic: bool = True,
        alpha: float = 1.0,
        symmetrize_kernel: bool = True,
    ):

        self.epsilon = epsilon
        self.is_stochastic = is_stochastic

        if not (0 <= alpha <= 1):
            raise ValueError(f"alpha has to be between [0, 1]. Got alpha={alpha}")
        self.alpha = alpha
        self.symmetrize_kernel = symmetrize_kernel

        self._internal_gauss_kernel = GaussianKernel(self.epsilon)

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
                rbf_kernel, basis_change_matrix = _conjugate_stochastic_kernel_matrix(
                    rbf_kernel
                )
            else:
                rbf_kernel = _stochastic_kernel_matrix(rbf_kernel)

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

    def is_symmetric_transform(self, is_pdist: bool) -> bool:
        """Indicates whether a symmetric kernel matrix transform is actually applied.

        Parameters
        ----------
        is_pdist
            True if the kernel evaluation is pairwise

        Returns
        -------

        """

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

        rbf_kernel_matrix = self._internal_gauss_kernel(
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

    def eval(
        self,
        distance_matrix: Union[np.ndarray, scipy.sparse.csr_matrix],
        is_pdist=False,
        **kernel_kwargs,
    ):
        """Evaluate kernel on pre-computed distance matrix.

        Parameters
        ----------

        distance_matrix
            distance matrix with shape `(n_samples, n_samples)`. For the sparse case note
            that the kernel acts on all stored data, i.e. usually real distance zeros
            must be stored in the matrix and only very large distance values (resulting
            in small kernel values) should not be stored.

        Returns
        -------
        :class:`numpy.ndarray`, :class:`scipy.sparse.csr_matrix`
            kernel matrix (or conjugate of it) with same type and shape as
            `distance_matrix`

        Optional[:class:`scipy.sparse.dia_matrix`]
            basis change matrix if `is_symmetrize=True` and the original kernel is
            non-symmetric

        Optional[:class:`numpy.ndarray`]
            Row sums from re-normalization, only returned for the `Y is None` case and
            are required for follow up out-of-sample kernel evaluations (`Y is not None`).
        """

        row_sums_alpha_fit = self._read_kernel_kwargs(kernel_kwargs, is_pdist)

        if is_pdist and row_sums_alpha_fit is not None:
            raise ValueError("if is_pdist=True then row_sum_alpha_fit=None")

        rbf_kernel_matrix = self._internal_gauss_kernel.eval(distance_matrix)
        kernel_matrix, basis_change_matrix, row_sums_alpha = self._normalize(
            rbf_kernel_matrix, row_sums_alpha_fit=row_sums_alpha_fit, is_pdist=is_pdist
        )

        return kernel_matrix, basis_change_matrix, row_sums_alpha


class ContinuousNNKernel(PCManifoldKernel):
    def __init__(self, k_neighbor, delta):

        if not is_integer(k_neighbor):
            raise TypeError("n_neighbors must be an integer")
        else:
            # make sure to only use Python built-in
            self.k_neighbor = int(k_neighbor)

        if not is_float(delta):
            if is_integer(delta):
                self.delta = float(delta)
            else:
                raise TypeError("delta must be of type float")
        else:
            # make sure to only use Python built-in
            self.delta = float(delta)

        super(ContinuousNNKernel, self).__init__()

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
            **{} if dist_backend_kwargs is None else dist_backend_kwargs,
        )

        is_pdist = Y is None
        return self.eval(
            distance_matrix,
            is_pdist=is_pdist,
            **{} if kernel_kwargs is None else kernel_kwargs,
        )

    def _validate(self, distance_matrix, is_pdist, reference_dist_knn):
        if distance_matrix.ndim != 2:
            raise ValueError("distance_matrix must be a two-dimensional array")

        n_samples_Y, n_samples_X = distance_matrix.shape

        if is_pdist:
            if n_samples_Y != n_samples_X:
                raise ValueError("if is_pdist=True, the distance matrix must be square")

            if not is_symmetric_matrix(distance_matrix):
                raise ValueError(
                    "if is_pdist=True, the distance matrix must be symmetric"
                )

            if isinstance(distance_matrix, np.ndarray):
                diagonal = np.diag(distance_matrix)
            else:
                diagonal = np.asarray(distance_matrix.diagonal(0))

            if (diagonal != 0).all():
                raise ValueError(
                    "if is_pdist=True, distance_matrix must have zeros on " "diagonal "
                )
        else:
            if reference_dist_knn is None:
                raise ValueError(
                    "if is_pdist=False, cdist_reference_k_nn must be provided"
                )

            if not isinstance(reference_dist_knn, np.ndarray):
                raise TypeError("cdist_reference_k_nn must be of type numpy.ndarray")

            if reference_dist_knn.ndim != 1:
                raise ValueError("cdist_reference_k_nn must be 1 dim.")

            if reference_dist_knn.shape[0] != n_samples_X:
                raise ValueError(
                    f"len(cdist_reference_k_nn)={reference_dist_knn.shape[0]} "
                    f"must be distance.shape[1]={n_samples_X}"
                )

            if self.k_neighbor < 1 or self.k_neighbor > n_samples_X - 1:
                raise ValueError(
                    "n_neighbors must be in the range 1 to number of samples"
                )

    def _kth_dist_sparse(self, distance_matrix: scipy.sparse.csr_matrix):

        # see mircorbenchmark_kth_nn.py for a comprison of implementations for the
        # sparse case

        def _get_kth_largest_elements_sparse(
            data: np.ndarray, indptr: np.ndarray, row_nnz, k_neighbor: int,
        ):
            dist_knn = np.zeros(len(row_nnz))
            for i in range(len(row_nnz)):
                start_row = indptr[i]
                dist_knn[i] = np.partition(
                    data[start_row : start_row + row_nnz[i]], k_neighbor - 1
                )[k_neighbor - 1]

            return dist_knn

        row_nnz = distance_matrix.getnnz(axis=1)

        if (row_nnz < self.k_neighbor).any():
            raise ValueError(
                f"There are {(row_nnz < self.k_neighbor).sum()} points that "
                f"do not have at least k_neighbor={self.k_neighbor}."
            )

        return _get_kth_largest_elements_sparse(
            distance_matrix.data, distance_matrix.indptr, row_nnz, self.k_neighbor,
        )

    def eval(self, distance_matrix, is_pdist=False, reference_dist_knn=None):

        self._validate(
            distance_matrix=distance_matrix,
            is_pdist=is_pdist,
            reference_dist_knn=reference_dist_knn,
        )

        if isinstance(distance_matrix, np.ndarray):
            dist_knn = np.partition(distance_matrix, self.k_neighbor, axis=-1)[
                :, self.k_neighbor
            ]
        elif isinstance(distance_matrix, scipy.sparse.csr_matrix):
            dist_knn = self._kth_dist_sparse(distance_matrix)
        else:
            raise TypeError(
                f"type(distance_matrix)={type(distance_matrix)} not supported."
            )

        distance_factors = _symmetric_matrix_division(
            distance_matrix,
            vec=np.sqrt(dist_knn),
            vec_right=np.sqrt(reference_dist_knn)
            if reference_dist_knn is not None
            else None,
        )

        if isinstance(distance_factors, np.ndarray):
            kernel_matrix = scipy.sparse.csr_matrix(
                distance_factors < self.delta, dtype=np.bool
            )
        else:
            assert isinstance(distance_factors, scipy.sparse.csr_matrix)
            distance_factors.data = (distance_factors.data < self.delta).astype(np.bool)
            distance_factors.eliminate_zeros()
            kernel_matrix = distance_factors

        if is_pdist:
            # return dist_knn, which is required for cdist_k_nearest_neighbor in
            # order to do a follow-up cdist request (then as reference_dist_knn as input).
            return kernel_matrix, dist_knn
        else:
            return kernel_matrix


class DmapKernelVariable(PCManifoldKernel):
    """Diffusion maps kernel with variable bandwidth of internal Gaussian radial basis
    kernel.

    .. warning::
        This class is not documented. Contributions are welcome
            * documentation
            * unit- or functional-testing

    References
    ----------
    :cite:`berry_nonparametric_2015`
    :cite:`berry_variable_2016`

    See Also
    --------
    :py:class:`DiffusionMapsVariable`

    """

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

    def eval(self, distance_matrix: Union[np.ndarray, scipy.sparse.csr_matrix]):

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
