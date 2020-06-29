import abc
from typing import Any, Dict, Optional, Tuple, Union

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

    if matrix.ndim != 2:
        raise ValueError("matrix must be two dimensional")

    if matrix.shape[0] != matrix.shape[1] and vec_right is None:
        raise ValueError("if matrix is non-square then vec_right must be provided")

    vec_inv_left = np.reciprocal(vec.astype(np.float64))

    if vec_right is None:
        vec_inv_right = vec_inv_left.view()
    else:
        vec_inv_right = np.reciprocal(vec_right.astype(np.float64))

    if vec_inv_left.ndim != 1 or vec_inv_left.shape[0] != matrix.shape[0]:
        raise ValueError("input 'vec' has the wrong shape or dimension")

    if vec_inv_right.ndim != 1 or vec_inv_right.shape[0] != matrix.shape[1]:
        raise ValueError("input 'vec_right' has the wrong shape or dimension")

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

        # The zeros are removed in the matrix multiplication. However, if matrix is a
        # distance matrix we need to preserve the "true zeros"!
        matrix.data[matrix.data == 0] = np.nan
        matrix = left_inv_diag_sparse @ matrix @ right_inv_diag_sparse
        matrix.data[np.isnan(matrix.data)] = 0
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


def _kth_nearest_neighbor_dist(
    distance_matrix: Union[np.ndarray, scipy.sparse.csr_matrix], k
) -> np.ndarray:
    """Compute the distance to the `k`-th nearest neighbor.

    Parameters
    ----------
    distance_matrix
        Matrix of shape `(n_samples_Y, n_samples_X)` to partition to find the distance of
        the `k`-th nearest neighbor. If the matrix is sparse each point must have a
        minimum number of `k` non-zero elements.

    k
        The distance of the `k`-th nearest neighbor is returned. The value must be a
        positive integer.

    Returns
    -------
    numpy.ndarray
        distance values
    """

    if not is_integer(k):
        raise ValueError(f"parameter 'k={k}' must be a positive integer")
    else:
        # make sure we deal with Python built-in
        k = int(k)

    if not (0 <= k <= distance_matrix.shape[1]):
        raise ValueError(
            "'k' must be an integer between 1 and "
            f"distance_matrix.shape[1]={distance_matrix.shape[1]}"
        )

    if isinstance(distance_matrix, np.ndarray):
        dist_knn = np.partition(distance_matrix, k - 1, axis=1)[:, k - 1]
    elif isinstance(distance_matrix, scipy.sparse.csr_matrix):
        # see mircobenchmark_kth_nn.py for a comparison of implementations for the
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

        if (row_nnz < k).any():
            raise ValueError(
                f"There are {(row_nnz < k).sum()} points that "
                f"do not have at least k_neighbor={k}."
            )

        dist_knn = _get_kth_largest_elements_sparse(
            distance_matrix.data, distance_matrix.indptr, row_nnz, k,
        )
    else:
        raise TypeError(f"type {type(distance_matrix)} not supported")

    return dist_knn


class PCManifoldKernel(Kernel):
    """Abstract base class for kernels used in *datafold*.

    See Also
    --------

    :py:class:`PCManifold`
    """

    @abc.abstractmethod
    def __call__(
        self,
        X: np.ndarray,
        Y: Optional[np.ndarray] = None,
        dist_kwargs: Optional[Dict[str, object]] = None,
        **kernel_kwargs,
    ):
        """Abstract method to compute the kernel matrix.
        
        If `Y=None`, then the pairwise-kernel is computed with `Y=X`. If `Y` is given,
        then the kernel matrix is computed component-wise with `X` being the reference
        and `Y` the query point cloud.

        Because the kernel can return a variable number of return values, this is
        unified with :meth:`PCManifoldKernel.read_kernel_output`.

        Parameters
        ----------
        X
            Data of shape `(n_samples_X, n_features)`.

        Y
            Reference data of shape `(n_samples_y, n_features_y)`

        dist_kwargs
            Keyword arguments for the distance algorithm.

        **kernel_kwargs
            Keyword arguments for the kernel algorithm.

        Returns
        -------
        numpy.ndarray
            Kernel matrix of shape `(n_samples_X, n_samples_X)` for the pairwise
            case or `(n_samples_y, n_samples_X)` for componentwise kernels

        Optional[Dict]
            For the pairwise computation, a kernel can return data that is required for a
            follow-up component-wise computation. The dictionary should contain keys
            that can be included as `**kernel_kwargs` to a follow-up ``__call__``. Note
            that if a kernel has no such values, this is empty (i.e. not even `None` is
            returned).
        
        Optional[Dict]
            If the kernel computes quantities of interest, then these quantities can be
            included in this dictionary. If this is returned, then this
            must be at the third return position (with a possible `None` return at the
            second position). If a kernel has no such values, this can be empty (i.e. not
            even `None` is returned).
        """

        raise NotImplementedError("base class")

    @abc.abstractmethod
    def eval(
        self, distance_matrix: Union[np.ndarray, scipy.sparse.csr_matrix]
    ) -> Union[np.ndarray, scipy.sparse.csr_matrix]:
        """Evaluate kernel on pre-computed distance matrix.

        For return values see :meth:`.__call__`.

        .. note::

            in this function there are no checks whether the correct distance
            matrix is computed with the kernel metric.

        Parameters
        ----------

        distance_matrix
            Matrix of shape `(n_samples_Y, n_samples_X)`. For the sparse case note
            that the kernel acts only on stored data, i.e. distance zeros such as
            duplicates must be stored in the matrix and only distance values exceeding a
            cut-off (i.e. large distance values) should not be stored.
        
        Returns
        -------

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

    def __repr__(self):

        param_str = ", ".join(
            [f"{name}={val}" for name, val in self.get_params().items()]
        )
        return f"{self.__class__.__name__}({param_str})"

    @staticmethod
    def read_kernel_output(
        kernel_output: Union[Union[np.ndarray, scipy.sparse.csr_matrix], Tuple]
    ) -> Tuple[Union[np.ndarray, scipy.sparse.csr_matrix], Dict, Dict]:
        """Unifies kernel output for all possible return scenarios of a kernel.
        
        This is required for models that allow generic kernels to be set where the
        number of outputs of the internal kernel are not known *apriori*.

        A kernel must return a computed kernel matrix in the first position. The two
        other places are optional:

        2. A dictionary containing keys that are required for a component-wise kernel
           computation (and set in `**kernel_kwargs`, see also below). Examples are
           computed density values.
        3. A dictionary that containes additional information computed during the
           computation. These extra information must always be at the third return
           position.

        .. code-block:: python

            # we don't know how the exact kernel and therefore not how many return
            # values are contained in kernel_output
            kernel_output = compute_kernel_matrix(X)

            # we read the output and obtain the three psossible places
            kernel_matrix, cdist_kwargs, extra_info = \
                PCManifold.read_kernel_output(kernel_output)

            # we can compute a follow up component-wise kernel matrix
            cdist_kernel_output = compute_kernel_matrix(X,Y, **cdist_kwargs)

        Parameters
        ----------
        kernel_output
            Output from an generic kernel, from which we don't know if it contains one,
            two or three return values.

        Returns
        -------
        Union[numpy.ndarray, scipy.sparse.csr_matrix]
            Kernel matrix.

        Dict
            Data required for follow-up component-wise computation. The dictionary
            should contain keys that can be included as `**kernel_kwargs` to the follow-up
            ``__call__``. Dictionary is empty if no data is contained in kernel output.

        Dict
            Quantities of interest with keys specific of the respective kernel.
            Dictionary is empty if no data is contained in kernel output.
        """

        if isinstance(kernel_output, (np.ndarray, scipy.sparse.csr_matrix)):
            # easiest case, we simply return the kernel matrix
            kernel_matrix, ret_cdist, ret_extra = [kernel_output, None, None]
        elif isinstance(kernel_output, tuple):
            if len(kernel_output) == 1:
                kernel_matrix, ret_cdist, ret_extra = [kernel_output[0], None, None]
            elif len(kernel_output) == 2:
                kernel_matrix, ret_cdist, ret_extra = (
                    kernel_output[0],
                    kernel_output[1],
                    None,
                )
            elif len(kernel_output) == 3:
                kernel_matrix, ret_cdist, ret_extra = kernel_output
            else:
                raise ValueError(
                    "kernel_output must has more than three elements. "
                    "Please report bug"
                )
        else:
            raise TypeError(
                "kernel_output must be either numpy.ndarray or tuple."
                "Please report bug."
            )

        ret_cdist = ret_cdist or {}
        ret_extra = ret_extra or {}

        return kernel_matrix, ret_cdist, ret_extra


class RadialBasisKernel(PCManifoldKernel):
    """Abstract base class for radial basis kernels.

    "A radial basis function (RBF) is a real-valued function whose value depends \
    only on the distance between the input and some fixed point." from
    `Wikipedia <https://en.wikipedia.org/wiki/Radial_basis_function>`_

    Parameters
    ----------
    distance_metric
        metric required for kernel
    """

    def __init__(self, distance_metric):
        self.distance_metric = distance_metric
        super(RadialBasisKernel, self).__init__()

    def __call__(
        self, X, Y=None, dist_kwargs=None, **kernel_kwargs
    ) -> Union[np.ndarray, scipy.sparse.csr_matrix]:
        """Compute kernel matrix.

        Parameters
        ----------
        X
            Reference point cloud of shape `(n_samples_X, n_features_X)`.

        Y
            Query point cloud of shape `(n_samples_Y, n_features_Y)`. If not given,
            then `Y=X`.

        dist_kwargs,
            Keyword arguments passed to the distance matrix computation. See
            :py:meth:`datafold.pcfold.compute_distance_matrix` for parameter arguments.

        kernel_kwargs
            ignored

        Returns
        -------
        Union[np.ndarray, scipy.sparse.csr_matrix]
            Kernel matrix of shape `(n_samples_Y, n_samples_X)`. If cut-off is
            specified in `dist_kwargs`, then the matrix is sparse.
        """

        if kernel_kwargs != {}:
            raise ValueError(f"invalid kwargs {kernel_kwargs}")

        X = np.atleast_2d(X)

        if Y is not None:
            Y = np.atleast_2d(Y)

        distance_matrix = compute_distance_matrix(
            X, Y, metric=self.distance_metric, **dist_kwargs or {},
        )

        kernel_matrix = self.eval(distance_matrix)

        return kernel_matrix


class GaussianKernel(RadialBasisKernel):
    r"""Gaussian radial basis kernel.

    .. math::
        K = \exp(\frac{-1}{2\varepsilon} \cdot D)

    where :math:`D` is the squared euclidean distance matrix.

    See also super classes :class:`RadialBasisKernel` and :class:`PCManifoldKernel`
    for more functionality and documentation.

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
        """Evaluate the kernel on pre-computed distance matrix.

        Parameters
        ----------
        distance_matrix
            Matrix of pairwise distances of shape `(n_samples_Y, n_samples_X)`.

        Returns
        -------
        Union[np.ndarray, scipy.sparse.csr_matrix]
            Kernel matrix of same shape and type as `distance_matrix`.
        """

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

    See also super classes :class:`RadialBasisKernel` and :class:`PCManifoldKernel`
    for more functionality and documentation.

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
        """Evaluate the kernel on pre-computed distance matrix.

        Parameters
        ----------
        distance_matrix
            Matrix of pairwise distances of shape `(n_samples_Y, n_samples_X)`.

        Returns
        -------
        Union[np.ndarray, scipy.sparse.csr_matrix]
            Kernel matrix of same shape and type as `distance_matrix`.
        """
        return _apply_kernel_function_numexpr(
            distance_matrix,
            expr="sqrt(1.0 / (2*eps) * D + 1.0)",
            expr_dict={"eps": self.epsilon},
        )


class InverseMultiquadricKernel(RadialBasisKernel):
    r"""Inverse multiquadric radial basis kernel.

    .. math::
        K = \sqrt(\frac{1}{2\varepsilon} \cdot D + 1)^{-1}

    where :math:`D` is the squared Euclidean distance matrix.

    See also super classes :class:`RadialBasisKernel` and :class:`PCManifoldKernel`
    for more functionality and documentation.

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
        """Evaluate the kernel on pre-computed distance matrix.

        Parameters
        ----------
        distance_matrix
            Matrix of pairwise distances of shape `(n_samples_Y, n_samples_X)`.

        Returns
        -------
        Union[np.ndarray, scipy.sparse.csr_matrix]
            Kernel matrix of same shape and type as `distance_matrix`.
        """
        return _apply_kernel_function_numexpr(
            distance_matrix,
            expr="1.0 / sqrt(1.0 / (2*eps) * D + 1.0)",
            expr_dict={"eps": self.epsilon},
        )


class CubicKernel(RadialBasisKernel):
    r"""Cubic radial basis kernel.

    .. math::
        K= D^{3}

    where :math:`D` is the Euclidean distance matrix.

    See also super classes :class:`RadialBasisKernel` and :class:`PCManifoldKernel`
    for more functionality and documentation.
    """

    def __init__(self):
        super(CubicKernel, self).__init__(distance_metric="euclidean")

    def eval(
        self, distance_matrix: Union[np.ndarray, scipy.sparse.csr_matrix]
    ) -> Union[np.ndarray, scipy.sparse.csr_matrix]:
        """Evaluate the kernel on pre-computed distance matrix.

        Parameters
        ----------
        distance_matrix
            Matrix of pairwise distances of shape `(n_samples_Y, n_samples_X)`.

        Returns
        -------
        Union[np.ndarray, scipy.sparse.csr_matrix]
            Kernel matrix of same shape and type as `distance_matrix`.
        """
        # return r ** 3
        return _apply_kernel_function_numexpr(distance_matrix, expr="D ** 3")


class QuinticKernel(RadialBasisKernel):
    r"""Quintic radial basis kernel.

    .. math::
        K= D^{5}

    where :math:`D` is the Euclidean distance matrix.

    See also super classes :class:`RadialBasisKernel` and :class:`PCManifoldKernel`
    for more functionality and documentation.
    """

    def __init__(self):
        super(QuinticKernel, self).__init__(distance_metric="euclidean")

    def eval(
        self, distance_matrix: Union[np.ndarray, scipy.sparse.csr_matrix]
    ) -> Union[np.ndarray, scipy.sparse.csr_matrix]:
        """Evaluate the kernel on pre-computed distance matrix.

        Parameters
        ----------
        distance_matrix
            Matrix of pairwise distances of shape `(n_samples_Y, n_samples_X)`.

        Returns
        -------
        Union[np.ndarray, scipy.sparse.csr_matrix]
            Kernel matrix of same shape and type as `distance_matrix`.
        """
        # r**5
        return _apply_kernel_function_numexpr(distance_matrix, "D ** 5")


class ThinPlateKernel(RadialBasisKernel):
    r"""Thin plate radial basis kernel.

    .. math::
        K = xlogy(D^2, D)


    where :math:`D` is the Euclidean distance matrix and argument for
    :class:`scipy.special.xlogy`.

    See also super classes :class:`RadialBasisKernel` and :class:`PCManifoldKernel`
    for more functionality and documentation.
    """

    def __init__(self):
        super(ThinPlateKernel, self).__init__(distance_metric="euclidean")

    def eval(self, distance_matrix: np.ndarray) -> np.ndarray:
        """Evaluate the kernel on pre-computed distance matrix.

        Parameters
        ----------
        distance_matrix
            Matrix of pairwise distances of shape `(n_samples_Y, n_samples_X)`.

        Returns
        -------
        Union[np.ndarray, scipy.sparse.csr_matrix]
            Kernel matrix of same shape and type as `distance_matrix`.
        """
        return xlogy(np.square(distance_matrix), distance_matrix)


class ContinuousNNKernel(PCManifoldKernel):
    """Compute the continuous `k` nearest-neighbor adjacency graph.

    The continuous `k` nearest neighbor (C-kNN) graph is an adjacency (i.e. unweighted)
    graph for which the (un-normalized) graph Laplacian converges spectrally to a
    Laplace-Beltrami operator on the manifold in the large data limit.

    Parameters
    ----------
    k_neighbor
        For each point the distance to the `k_neighbor` nearest neighbor is computed.
        If a sparse matrix is computed (with cut-off distance), then each point must
        have a minimum of `k` stored neighbors. (see `kmin` parameter in
        :meth:`pcfold.distance.compute_distance_matrix`).

    delta
        Unit-less scale parameter.

    References
    ----------

    :cite:`berry_consistent_2019`
    """

    def __init__(self, k_neighbor: int, delta: float):

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

        if self.k_neighbor < 1:
            raise ValueError(
                f"'parameter 'k_neighbor={self.k_neighbor}' must be a positive integer"
            )

        if self.delta <= 0.0:
            raise ValueError(
                f"'parameter 'delta={self.delta}' must be a positive float"
            )

        super(ContinuousNNKernel, self).__init__()

    def _validate_reference_dist_knn(self, is_pdist, reference_dist_knn):
        if is_pdist and reference_dist_knn is None:
            raise ValueError("For the 'cdist' case 'reference_dist_knn' must be given")

    def __call__(
        self,
        X: np.ndarray,
        Y: Optional[np.ndarray] = None,
        dist_kwargs: Optional[Dict] = None,
        reference_dist_knn=None,
        **kernel_kwargs,
    ):
        """Compute (sparse) adjacency graph to describes a point neighborhood.

        Parameters
        ----------
        X
            Reference point cloud of shape `(n_samples_X, n_features_X)`.

        Y
            Query point cloud of shape `(n_samples_Y, n_features_Y)`. If not given,
            then `Y=X`.

        dist_kwargs
            Keyword arguments passed to the internal distance matrix computation. See
            :py:meth:`datafold.pcfold.compute_distance_matrix` for parameter arguments.

        reference_dist_knn
            Distances to the `k`-th nearest neighbor for each point in `X`. The values
            are mandatory if `Y` is not `None`.

        Returns
        -------
        scipy.sparse.csr_matrix
            Sparse adjacency matrix describing the unweighted, undirected continuous
            nearest neighbor graph.

        Optional[Dict[str, numpy.ndarray]]
            For a pair-wise kernel evaluation, a Dictionary with key
            `reference_dist_knn` with the `k`-the nearest neighbors for each point are
            returned.
        """

        if kernel_kwargs != {}:
            raise ValueError(f"invalid kwargs {kernel_kwargs}")

        is_pdist = Y is None

        dist_kwargs = dist_kwargs or {}
        # minimum number of neighbors required in the sparse case!
        dist_kwargs.setdefault("kmin", self.k_neighbor)

        distance_matrix = compute_distance_matrix(
            X, Y, metric="euclidean", **dist_kwargs
        )

        return self.eval(
            distance_matrix, is_pdist=is_pdist, reference_dist_knn=reference_dist_knn
        )

    def _validate(self, distance_matrix, is_pdist, reference_dist_knn):
        if distance_matrix.ndim != 2:
            raise ValueError("distance_matrix must be a two-dimensional array")

        n_samples_Y, n_samples_X = distance_matrix.shape

        if is_pdist:
            if n_samples_Y != n_samples_X:
                raise ValueError(
                    "if is_pdist=True, the distance matrix must be square "
                    "and symmetric"
                )

            if isinstance(distance_matrix, np.ndarray):
                diagonal = np.diag(distance_matrix)
            else:
                diagonal = np.asarray(distance_matrix.diagonal(0))

            if (diagonal != 0).all():
                raise ValueError(
                    "if is_pdist=True, distance_matrix must have zeros on diagonal "
                )
        else:
            if reference_dist_knn is None:
                raise ValueError(
                    "if is_pdist=False, 'reference_dist_knn' (=None) must be provided."
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

    def eval(
        self, distance_matrix, is_pdist=False, reference_dist_knn=None
    ) -> Tuple[scipy.sparse.csr_matrix, Optional[Dict[Any, np.ndarray]]]:
        """Evaluate kernel on pre-computed distance matrix.

        For return values see :meth:`.__call__`.

        Parameters
        ----------
        distance_matrix
            Pre-computed matrix.

        is_pdist
            If True, the `distance_matrix` is assumed to be symmetric and with zeros on
            the diagonal (self distances). Note, that there are no checks to validate
            the distance matrix.

        reference_dist_knn
            An input is required for a component-wise evaluation of the kernel. This is
            the case if the distance matrix is rectangular or non-symmetric (i.e.,
            ``is_pdist=False``). The required values are returned for a pre-evaluation
            of the pair-wise evaluation.
        """

        self._validate(
            distance_matrix=distance_matrix,
            is_pdist=is_pdist,
            reference_dist_knn=reference_dist_knn,
        )

        dist_knn = _kth_nearest_neighbor_dist(distance_matrix, self.k_neighbor)

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

        # return dist_knn, which is required for cdist_k_nearest_neighbor in
        # order to do a follow-up cdist request (then as reference_dist_knn as input).
        if is_pdist:
            ret_cdist: Optional[Dict[str, np.ndarray]] = dict(
                reference_dist_knn=dist_knn
            )
        else:
            ret_cdist = None

        return kernel_matrix, ret_cdist


class DmapKernelFixed(PCManifoldKernel):
    """Diffusion map kernel with fixed kernel bandwidth.

    This kernel wraps an kernel to describe a diffusion process.

    Parameters
    ----------

    internal_kernel
        Kernel that describes the proximity between data points.

    is_stochastic
        If True, the kernel matrix is row-normalized.

    alpha
        Degree of re-normalization of sampling density in point cloud. `alpha` must be
        inside the interval [0, 1] (inclusive).

    symmetrize_kernel
        If True, performs a conjugate transformation which can improve numerical
        stability for matrix operations (such as computing eigenpairs). The matrix to
        change the basis back is provided as a quantity of interest (see
        possible return values in :meth:`PCManifoldKernel.__call__`).

    See Also
    --------
    :py:class:`DiffusionMaps`

    References
    ----------
    :cite:`coifman_diffusion_2006`
    """

    def __init__(
        self,
        internal_kernel: PCManifoldKernel = GaussianKernel(epsilon=1.0),
        is_stochastic: bool = True,
        alpha: float = 1.0,
        symmetrize_kernel: bool = True,
    ):

        self.is_stochastic = is_stochastic

        if not (0 <= alpha <= 1):
            raise ValueError(f"alpha has to be between [0, 1]. Got alpha={alpha}")
        self.alpha = alpha
        self.symmetrize_kernel = symmetrize_kernel

        self.internal_kernel = internal_kernel

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

        super(DmapKernelFixed, self).__init__()

    def _normalize_sampling_density_kernel_matrix(
        self, kernel_matrix, row_sums_alpha_fit
    ):
        """Normalize (sparse/dense) kernels with positive `alpha` value. This is also
        referred to a 'renormalization' of sampling density. """

        if row_sums_alpha_fit is None:
            assert is_symmetric_matrix(kernel_matrix)
        else:
            assert row_sums_alpha_fit.shape[0] == kernel_matrix.shape[1]

        row_sums = kernel_matrix.sum(axis=1)

        if scipy.sparse.issparse(kernel_matrix):
            row_sums = row_sums.A1  # turns matrix (deprectated) into np.ndarray

        if self.alpha < 1:
            row_sums_alpha = np.power(row_sums, self.alpha, out=row_sums)
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
                # Increases numerical stability when solving the eigenproblem
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
        """Indicates whether a symmetric conjugate matrix is computed.

        Parameters
        ----------
        is_pdist
            If True, the kernel matrix is computed with pair-wise.

        Returns
        -------
        """

        # If the kernel is made stochastic, it looses the symmetry, if symmetric_kernel
        # is set to True, then apply the the symmetry transformation
        return is_pdist and self.is_stochastic and self.is_symmetric

    def _validate_row_alpha_fit(self, is_pdist, row_sums_alpha_fit):
        if (
            self.is_stochastic
            and self.alpha > 0
            and not is_pdist
            and row_sums_alpha_fit is None
        ):
            raise ValueError(
                "cdist request can not be carried out, if 'row_sums_alpha_fit=None'"
                "Please consider to report bug."
            )

    def _eval(self, kernel_output, is_pdist, row_sums_alpha_fit):

        self._validate_row_alpha_fit(
            is_pdist=is_pdist, row_sums_alpha_fit=row_sums_alpha_fit
        )

        kernel_matrix, internal_ret_cdist, _ = PCManifoldKernel.read_kernel_output(
            kernel_output=kernel_output
        )

        kernel_matrix, basis_change_matrix, row_sums_alpha = self._normalize(
            kernel_matrix, row_sums_alpha_fit=row_sums_alpha_fit, is_pdist=is_pdist
        )

        if is_pdist:
            ret_cdist = dict(
                row_sums_alpha_fit=row_sums_alpha,
                internal_kernel_kwargs=internal_ret_cdist,
            )
            ret_extra = dict(basis_change_matrix=basis_change_matrix)
        else:
            # no need for row_sums_alpha or the basis change matrix in the cdist case
            ret_cdist = None
            ret_extra = None

        return kernel_matrix, ret_cdist, ret_extra

    def __call__(
        self,
        X: np.ndarray,
        Y: Optional[np.ndarray] = None,
        dist_kwargs: Optional[Dict] = None,
        internal_kernel_kwargs: Optional[Dict] = None,
        row_sums_alpha_fit: Optional[np.ndarray] = None,
        **kernel_kwargs,
    ) -> Tuple[
        Union[np.ndarray, scipy.sparse.csr_matrix], Optional[Dict], Optional[Dict]
    ]:
        """Compute the diffusion map kernel.

        Parameters
        ----------
        X
            Reference point cloud of shape `(n_samples_X, n_features_X)`.

        Y
            Query point cloud of shape `(n_samples_Y, n_features_Y)`. If not given,
            then `Y=X`.

        dist_kwargs
            Keyword arguments passed to the internal distance matrix computation. See
            :py:meth:`datafold.pcfold.compute_distance_matrix` for parameter arguments.

        internal_kernel_kwargs
            Keyword arguments passed to the set internal kernel.

        row_sums_alpha_fit
            Row sum values during re-normalization computed during pair-wise kernel
            computation. The parameter is mandatory for the compontent-wise kernel
            computation and if `alpha>0`.

        Returns
        -------
        numpy.ndarray`, `scipy.sparse.csr_matrix`
            kernel matrix (or conjugate of it) with same type and shape as
            `distance_matrix`
        
        Optional[Dict[str, numpy.ndarray]]
            Row sums from re-normalization in key 'row_sums_alpha_fit', only returned for
            pairwise computations. The values are required for follow up out-of-sample
            kernel evaluations (`Y is not None`).

        Optional[Dict[str, scipy.sparse.dia_matrix]]
            Basis change matrix (sparse diagonal) if `is_symmetrize=True` and only
            returned if the kernel matrix is a symmetric conjugate of the true
            diffusion kernel matrix. Required to recover the diffusion map eigenvectors
            from the symmetric conjugate matrix.
        """

        is_pdist = Y is None

        kernel_output = self.internal_kernel(
            X, Y=Y, dist_kwargs=dist_kwargs or {}, **internal_kernel_kwargs or {}
        )

        return self._eval(
            kernel_output=kernel_output,
            is_pdist=is_pdist,
            row_sums_alpha_fit=row_sums_alpha_fit,
        )

    def eval(
        self,
        distance_matrix: Union[np.ndarray, scipy.sparse.csr_matrix],
        is_pdist=False,
        row_sums_alpha_fit=None,
    ):
        """Evaluate kernel on pre-computed distance matrix.

        For return values see :meth:`.__call__`.

        Parameters
        ----------

        distance_matrix
            Matrix of shape `(n_samples_Y, n_samples_X)`.

        is_pdist:
            If True, the distance matrix must be square

        Returns
        -------
        """

        kernel_output = self.internal_kernel.eval(distance_matrix)
        return self._eval(
            kernel_output=kernel_output,
            is_pdist=is_pdist,
            row_sums_alpha_fit=row_sums_alpha_fit,
        )


class DmapKernelVariable(PCManifoldKernel):
    """Diffusion maps kernel with variable kernel bandwidth.

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
            # This is with a cut-off rate:
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
        self, X, Y=None, kernel_kwargs=None, dist_kwargs=None,
    ):

        dist_kwargs = dist_kwargs or {}
        cut_off = dist_kwargs.pop("cut_off", None)

        if cut_off is not None and not np.isinf(cut_off):
            raise NotImplementedError("Handling sparsity is currently not implemented!")

        if Y is not None:
            raise NotImplementedError("cdist case is currently not implemented!")

        if self.k > X.shape[0]:
            raise ValueError(
                f"nr of nearest neighbors (self.k={self.k}) "
                f"is larger than number of samples (={X.shape[0]})"
            )

        distance_matrix = compute_distance_matrix(
            X, Y, metric="sqeuclidean", **dist_kwargs,
        )

        operator_l_matrix, basis_change_matrix, rho0, rho, q0, q_eps_s = self.eval(
            distance_matrix
        )

        # TODO: this is not yet aligned to the kernel_matrix, cdist_dict, extra_dict
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
