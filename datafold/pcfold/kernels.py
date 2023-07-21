import abc
import warnings
from typing import Any, Callable, Optional, Union

import numexpr as ne
import numpy as np
import pandas as pd
import scipy.sparse
import scipy.spatial
from sklearn.gaussian_process.kernels import Kernel
from sklearn.preprocessing import normalize
from sklearn.utils import check_scalar

from datafold.pcfold.distance import DistanceAlgorithm, init_distance_algorithm
from datafold.pcfold.timeseries.accessor import TSCAccessor
from datafold.utils.general import (
    df_type_and_indices_from,
    diagmat_dot_mat,
    is_df_same_index,
    is_float,
    is_integer,
    is_symmetric_matrix,
    mat_dot_diagmat,
    remove_numeric_noise_symmetric_matrix,
)

KernelType = Union[pd.DataFrame, np.ndarray, scipy.sparse.csr_matrix]


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
    expr_dict = expr_dict or {}
    assert "D" not in expr_dict.keys()

    if scipy.sparse.issparse(distance_matrix):
        # copy because the distance matrix may be used further by the user
        distance_matrix = distance_matrix.copy()
        expr_dict["D"] = distance_matrix.data
        ne.evaluate(expr, expr_dict, out=distance_matrix.data)
        return distance_matrix  # returns actually the kernel matrix
    else:
        expr_dict["D"] = distance_matrix
        return ne.evaluate(expr, expr_dict)


def _symmetric_matrix_division(
    matrix: Union[np.ndarray, scipy.sparse.spmatrix],
    vec: np.ndarray,
    is_symmetric: bool,
    vec_right: Optional[np.ndarray] = None,
    scalar: float = 1.0,
    value_zero_division: Union[str, float] = "raise",
) -> Union[np.ndarray, scipy.sparse.csr_matrix]:
    r"""Symmetric division, often appearing in kernels.

    .. math::
        \frac{M_{i, j}}{a v^(l)_i v^(r)_j}

    where :math:`M` is a (kernel-) matrix and its elements are divided by the
    (left and right) vector elements :math:`v` and scalar :math:`a`.

    .. warning::
        The implementation is in-place and can overwrites the input matrix. Make a copy
        beforehand if the matrix values are still required.

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

    """
    if matrix.ndim != 2:
        raise ValueError("Parameter 'matrix' must be a two dimensional array.")

    if matrix.shape[0] != matrix.shape[1] and vec_right is None:
        raise ValueError(
            "If 'matrix' is non-square, then 'vec_right' must be provided."
        )

    vec = vec.astype(float)
    zero_idx_vec = vec == 0.0

    if zero_idx_vec.any():
        if value_zero_division == "raise":
            raise ZeroDivisionError(
                f"Encountered zero values in division in {zero_idx_vec.sum()} points."
            )
        else:
            # division results into 'nan' without raising a ZeroDivisionWarning. The
            # nan values will be replaced later
            vec[zero_idx_vec] = np.nan

    vec_inv_left = np.reciprocal(vec)

    if vec_right is None:
        vec_inv_right = vec_inv_left.view()
    else:
        vec_right = vec_right.astype(float)
        zero_idx_vec = vec_right == 0.0

        if zero_idx_vec.any():
            if value_zero_division == "raise":
                raise ZeroDivisionError(
                    f"Encountered zero values in division in {zero_idx_vec.sum()}"
                )
            else:
                vec_right[zero_idx_vec] = np.nan

        vec_inv_right = np.reciprocal(vec_right.astype(float))

    if vec_inv_left.ndim != 1 or vec_inv_left.shape[0] != matrix.shape[0]:
        raise ValueError(
            f"Invalid input: 'vec.shape={vec.shape}' is not compatible with "
            f"'matrix.shape={matrix.shape}'."
        )

    if vec_inv_right.ndim != 1 or vec_inv_right.shape[0] != matrix.shape[1]:
        raise ValueError(
            f"Invalid input: 'vec_right.shape={vec_inv_right.shape}' is not compatible "
            f"with 'matrix.shape={matrix.shape}'."
        )

    if scipy.sparse.issparse(matrix):
        # The zeros are removed in the matrix multiplication, but because 'matrix' is
        # usually a distance matrix we need to preserve the "true zeros"!
        matrix.data[matrix.data == 0] = np.inf

        matrix = mat_dot_diagmat(matrix, vec_inv_right)
        matrix = diagmat_dot_mat(vec_inv_left, matrix)

        matrix.data[np.isinf(matrix.data)] = 0

        # this imposes precedence order
        #    --> np.inf/np.nan -> np.nan
        # i.e. for cases with 0/0, set 'value_zero_division'
        if isinstance(value_zero_division, (int, float)):
            matrix.data[np.isnan(matrix.data)] = value_zero_division

    else:
        # This computes efficiently:
        # np.diag(1/vector_elements) @ matrix @ np.diag(1/vector_elements)
        matrix = diagmat_dot_mat(vec_inv_left, matrix, out=matrix)
        matrix = mat_dot_diagmat(matrix, vec_inv_right, out=matrix)

        if isinstance(value_zero_division, (int, float)):
            matrix[np.isnan(matrix)] = value_zero_division

    # sparse and dense
    if is_symmetric and vec_right is None:
        matrix = remove_numeric_noise_symmetric_matrix(matrix)

    if scalar != 1.0:
        scalar = 1.0 / scalar
        matrix = np.multiply(matrix, scalar, out=matrix)

    return matrix


def _conjugate_stochastic_kernel_matrix(
    kernel_matrix: Union[np.ndarray, scipy.sparse.spmatrix]
) -> tuple[Union[np.ndarray, scipy.sparse.spmatrix], scipy.sparse.dia_matrix]:
    r"""Conjugate transformation to obtain symmetric (conjugate) kernel matrix with same
    spectrum properties.

    Rabin et al. :cite:`rabin_heterogeneous_2012` states in equation Eq. 3.1 \
    (notation adapted):

    .. math::
        P = D^{-1} K

    the standard row normalization. Eq. 3.3 shows that matrix :math:`P` has a similar
    matrix with

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
    Tuple[Union[np.ndarray, scipy.sparse.csr_matrix], scipy.sparse.dia_matrix]
        conjugate matrix (type as `kernel_matrix`) and (sparse) diagonal matrix to recover
        eigenvectors

    References
    ----------
    :cite:`rabin-2012`

    """
    left_vec = kernel_matrix.sum(axis=1)

    if scipy.sparse.issparse(kernel_matrix):
        # to np.ndarray in case it is deprecated format np.matrix
        left_vec = left_vec.A1

    if left_vec.dtype.kind != "f":
        left_vec = left_vec.astype(float)

    left_vec = np.sqrt(left_vec, out=left_vec)

    kernel_matrix = _symmetric_matrix_division(
        kernel_matrix, vec=left_vec, is_symmetric=True, vec_right=None
    )

    # This is D^{-1/2} in sparse matrix form.
    basis_change_matrix = scipy.sparse.diags(np.reciprocal(left_vec, out=left_vec))
    return kernel_matrix, basis_change_matrix


def _stochastic_kernel_matrix(kernel_matrix: Union[np.ndarray, scipy.sparse.spmatrix]):
    """Normalizes a matrix to a row-stochastic matrix.

    Mathematically,

    .. math::

        M = D^{-1} K

    where matrix :math:`M` is the row-normalized (kernel) matrix from :math:`K`. The matrix
    :math:`D` has the row sums of :math:`K` on the diagonal.

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
        # in a microbenchmark this turned out to be the fastest solution for sparse
        # matrices
        kernel_matrix = normalize(kernel_matrix, copy=False, norm="l1")
    else:  # dense
        normalize_diagonal = np.sum(kernel_matrix, axis=1)

        with np.errstate(divide="ignore", over="ignore"):
            # especially in cdist computations there can be far away outliers
            # (or very small scale/epsilon). This results in elements near 0 and
            #  the reciprocal can then
            #     - be inf
            #     - overflow (resulting in negative values)
            #  these cases are catched with 'bool_invalid' below
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
        Distance matrix of shape `(n_samples_Y, n_samples_X)` from which to find the
        `k`-th nearest neighbor and its corresponding distance to return. If the matrix is
        sparse each point must have a minimum number of `k` neighbours (i.e. non-zero
        elements per row).

    k
        The distance to the `k`-th nearest neighbor.

    Returns
    -------
    numpy.ndarray
        distance values
    """
    if not is_integer(k):
        raise ValueError(f"parameter {k=} must be a positive integer")
    else:
        # make sure we deal with Python built-in
        k = int(k)

    if not (0 < k <= distance_matrix.shape[1]):
        raise ValueError(
            f"{k=} must be an integer between 1 and {distance_matrix.shape[1]=}"
        )

    if isinstance(distance_matrix, np.ndarray):
        dist_knn = np.partition(distance_matrix, k - 1, axis=1)[:, k - 1]
    elif isinstance(distance_matrix, scipy.sparse.csr_matrix):
        # see mircobenchmark_kth_nn.py for a comparison of implementations for the
        # sparse case

        def _get_kth_largest_elements_sparse(
            data: np.ndarray,
            indptr: np.ndarray,
            row_nnz,
            k_neighbor: int,
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
            distance_matrix.data,
            distance_matrix.indptr,
            row_nnz,
            k,
        )
    else:
        raise TypeError(f"type {type(distance_matrix)} not supported")

    return dist_knn


class BaseManifoldKernel(Kernel):
    def __init__(
        self,
        is_symmetric: bool = True,
        is_stochastic: bool = False,
        distance: Optional[Union[dict, DistanceAlgorithm]] = None,
    ):
        """Initialize new kernel.

        Parameters
        ----------
        is_symmetric
            Indicate whether the resulting kernel matrix is symmetric. A kernel may be
            symmetric but computed with a `k` nearest-neighbor algorithm, ending in a
            non-symmetric sparse matrix.

        is_stochastic
            Indicate whether the pairwise kernel matrix is stochastic. The typical case is
            that the rows sum up to one (but columns-wise or double-stochastic matrix could
            also be possible).
        """
        self._is_symmetric_kernel = is_symmetric
        self.is_stochastic = is_stochastic

        if distance is None or isinstance(distance, dict):
            self.distance: DistanceAlgorithm = init_distance_algorithm(
                **(distance or {})
            )
        else:
            self.distance = distance

    @property
    def metric(self):
        """Distance metric in the kernel."""
        return self.distance.metric

    @property
    def is_symmetric(self):
        """Indicates whether a pairwise kernel matrix is symmetric.

        Note that a kernel matrix to be symmetric, also the distance matrix must be symmetric
        (especially for k-nearest-neighbor this is often not the case).
        """
        return self.distance.is_symmetric and self._is_symmetric_kernel

    @abc.abstractmethod
    def __call__(
        self, X, Y=None, **kernel_kwargs
    ) -> Union[np.ndarray, scipy.sparse.spmatrix]:
        """Compute kernel matrix.

        If `Y=None`, then the pairwise-kernel is computed with `Y=X`. If `Y` is given,
        then the kernel matrix is computed component-wise with `X` being the reference
        and `Y` the query points.

        Parameters
        ----------
        Args:
        ----
        kwargs
            See parameter documentation in subclasses.

        Returns
        -------
        Union[np.ndarray, scipy.sparse.spmatrix]
            Kernel matrix of shape `(n_samples_Y, n_samples_X)`. If cut-off is
            specified in `dist_kwargs`, then the matrix is sparse.

        """

    def diag(self, X):
        """(Not implemented, not used in datafold).

        Raises
        ------
        NotImplementedError
            this is only to overwrite abstract method in super class
        """
        raise NotImplementedError("base class")

    def is_stationary(self):
        """(Not implemented, not used in datafold).

        Raises
        ------
        NotImplementedError
            this is only to overwrite abstract method in super class
        """
        # in datafold there is no handling of this attribute, if required this has to
        # be implemented
        raise NotImplementedError("base class")

    def __repr__(self, print_distance=True):
        from copy import deepcopy

        _params = deepcopy(self.get_params())
        _params.pop("distance", None)

        param_str = ", ".join([f"{name}={val}" for name, val in _params.items()])

        if print_distance:
            _distance = f"\n\t{self.distance=}".replace("self.", "")
        else:
            _distance = ""

        return f"{self.__class__.__name__}({_distance}\n\t{param_str}\n)"

    def _read_kernel_kwargs(self, attrs: Optional[list[str]], kernel_kwargs: dict):
        return_values: list[Any] = []

        if attrs is not None:
            for attr in attrs:
                return_values.append(kernel_kwargs.pop(attr, None))

        if kernel_kwargs != {}:
            raise KeyError(
                f"kernel_kwargs.keys = {kernel_kwargs.keys()} are not supported"
            )

        if len(return_values) == 0:
            return None
        elif len(return_values) == 1:
            return return_values[0]
        else:
            return return_values

    def _required_attrs(self, attrs: list, is_fit):
        for a in attrs:
            if is_fit:
                if hasattr(self, a):
                    raise AttributeError(
                        f"Attribute {a} is already set from a previous "
                        f"pairwise kernel evaluation. The attributes cannot "
                        f"be reset, use a new kernel object."
                    )
            else:
                if not hasattr(self, a):
                    raise AttributeError(
                        f"Attribute {a} is missing in kernel. It is required to "
                        f"first compute the pairwise kernel matrix "
                        f"(i.e. :code:`kernel(X)`) to set the attributes."
                    )


class PCManifoldKernel(BaseManifoldKernel):
    """Abstract base class for kernels evaluated on static point clouds or time series.

    See Also
    --------
    :py:class:`PCManifold`
    :py:class:`TSCDataFrame`
    """

    @abc.abstractmethod
    def __call__(
        self,
        X: np.ndarray,
        Y: Optional[np.ndarray] = None,
        **kernel_kwargs,
    ):
        """Abstract method to compute the kernel matrix from a point cloud.

        Parameters
        ----------
        X
            Data of shape `(n_samples_X, n_features)`.

        Y
            Reference data of shape `(n_samples_y, n_features_y)`

        dist_kwargs
            Keyword arguments for the distance computation.

        **kernel_kwargs
            Keyword arguments for the kernel algorithm.

        Returns
        -------
        numpy.ndarray
            Kernel matrix of shape `(n_samples_X, n_samples_X)` for the pairwise
            case or `(n_samples_y, n_samples_X)` for component-wise kernels

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
    def evaluate(
        self, distance_matrix: Union[np.ndarray, scipy.sparse.csr_matrix]
    ) -> Union[np.ndarray, scipy.sparse.csr_matrix]:
        """Evaluate kernel on pre-computed distance matrix.

        For return values see :meth:`.__call__`.

        .. note::

            In this function there are no checks of whether the distance matrix was
            computed with the correct metric required by the kernel.

        Parameters
        ----------
        distance_matrix
            Matrix of shape `(n_samples_Y, n_samples_X)`. For the sparse matrix case note
            that the kernel acts only on stored data, i.e. distance values with
            exactly zero (duplicates and self distance) must be stored explicitly in the
            matrix. Only large distance values exceeding a cut-off should not be stored.

        Returns
        -------

        """
        raise NotImplementedError("base class")


class TSCManifoldKernel(BaseManifoldKernel):
    """Abstract base class for kernels evaluating exclusively on time series.

    See Also
    --------
    :py:class:`.TSCDataFrame`
    """

    @abc.abstractmethod
    def __call__(
        self,
        X: pd.DataFrame,
        Y: Optional[pd.DataFrame] = None,
        *,
        dist_kwargs: Optional[dict[str, object]] = None,
        **kernel_kwargs,
    ):
        """Abstract method to compute the kernel matrix from a time series collection.

        Parameters
        ----------
        X
            Data of shape `(n_samples_X, n_features)`.

        Y
            Data of shape `(n_samples_Y, n_features)`.

        dist_kwargs
            Keyword arguments for the distance computation.

        **kernel_kwargs
            Keyword arguments for the kernel algorithm.

        Returns
        -------
        Union[TSCDataFrame, pd.DataFrame]
            The computed kernel matrix as ``TSCDataFrame`` (or fallback
            ``pandas.DataFrame``, if not regular time series). The basis shape of the
            kernel matrix `(n_samples_Y, n_samples_X)`. However, the kernel may not be
            evaluated at all given input time values and is then reduced accordingly.

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


class RadialBasisKernel(PCManifoldKernel, metaclass=abc.ABCMeta):
    """Abstract base class for radial basis kernels.

    "A radial basis function (RBF) is a real-valued function whose value depends \
    only on the distance between the input and some fixed point." (taken from
    `Wikipedia <https://en.wikipedia.org/wiki/Radial_basis_function>`__)

    Parameters
    ----------
    required_metric
        metric required for kernel
    """

    def __init__(
        self,
        required_metric: str,
        distance: Optional[Union[dict, DistanceAlgorithm]] = None,
    ):
        _metric_mismatch = ValueError(
            "The metric is fixed for radial basis kernel and should not set in "
            "dist_kwargs"
        )

        if distance is None:
            distance = {"metric": required_metric}
        elif isinstance(distance, DistanceAlgorithm):
            if distance.metric != required_metric:
                raise _metric_mismatch
        elif isinstance(distance, dict):
            _exist_metric = distance.pop("metric", None)

            if _exist_metric is None or _exist_metric == required_metric:
                distance["metric"] = required_metric
            else:
                raise _metric_mismatch

        super().__init__(distance=distance)

    @classmethod
    def _check_bandwidth_parameter(cls, parameter, name) -> float:
        check_scalar(
            parameter,
            name=name,
            target_type=(float, np.floating, int, np.integer),
            min_val=0,
            include_boundaries="neither",
        )
        return float(parameter)

    def __call__(
        self, X, Y=None, **kernel_kwargs
    ) -> Union[np.ndarray, scipy.sparse.csr_matrix]:
        """Compute kernel matrix.

        Parameters
        ----------
        X
            Reference point cloud of shape `(n_samples_X, n_features_X)`.

        Y
            Query point cloud of shape `(n_samples_Y, n_features_Y)`. If not given,
            then `Y=X`.

        **kernel_kwargs
            None

        Returns
        -------
        Union[np.ndarray, scipy.sparse.csr_matrix]
            Kernel matrix of shape `(n_samples_Y, n_samples_X)`. If cut-off is
            specified in `dist_kwargs`, then the matrix is sparse.
        """
        self._read_kernel_kwargs(attrs=None, kernel_kwargs=kernel_kwargs)

        X = np.atleast_2d(X)

        is_pdist = Y is None

        if not is_pdist:
            Y = np.atleast_2d(Y)

        distance_matrix = self.distance(X, Y)

        kernel_matrix = self.evaluate(distance_matrix)
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
        The kernel scale as a positive float value. Alternatively, a
        callable can be passed to which the distance matrix is
        (i.e. ``function(distance_matrix)``). The return value of this function must be a
        positive float that is used as the epsilon.
    """

    def __init__(self, epsilon: Union[float, Callable] = 1.0, distance=None):
        self.epsilon = epsilon
        super().__init__(required_metric="sqeuclidean", distance=distance)

    def evaluate(
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

        if callable(self.epsilon):
            if isinstance(distance_matrix, scipy.sparse.csr_matrix):
                self.epsilon = self.epsilon(distance_matrix.data)
            elif isinstance(distance_matrix, np.ndarray):
                self.epsilon = self.epsilon(distance_matrix)
                print(self.epsilon)
            else:
                raise TypeError(
                    f"Invalid type: type(distance_matrix)={type(distance_matrix)}."
                    f"Please report bug."
                )

        self.epsilon = self._check_bandwidth_parameter(
            parameter=self.epsilon, name="epsilon"
        )

        kernel_matrix = _apply_kernel_function_numexpr(
            distance_matrix,
            expr="exp((- 1 / (2*eps)) * D)",
            expr_dict={"eps": self.epsilon},
        )

        return kernel_matrix


class MultiquadricKernel(RadialBasisKernel):
    r"""Multiquadric radial basis kernel.

    .. math::
        K = \sqrt(\frac{1}{2 \varepsilon} \cdot D + 1)

    where :math:`D` is the squared euclidean distance matrix.

    See also super classes :class:`RadialBasisKernel` and :class:`PCManifoldKernel`
    for more functionality and documentation.

    Parameters
    ----------
    epsilon
        Positive float to scale the kernel weights.
    """

    def __init__(self, epsilon: float = 1.0, distance: Optional[dict] = None):
        self.epsilon = epsilon
        super().__init__(required_metric="sqeuclidean", distance=distance)

    def evaluate(
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
        self.epsilon = self._check_bandwidth_parameter(
            parameter=self.epsilon, name="epsilon"
        )

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
        Positive float to scale the kernel weights.
    """

    def __init__(self, epsilon: float = 1.0, distance=None):
        self.epsilon = epsilon
        super().__init__(required_metric="sqeuclidean", distance=distance)

    def evaluate(
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
        self.epsilon = self._check_bandwidth_parameter(
            parameter=self.epsilon, name="epsilon"
        )

        return _apply_kernel_function_numexpr(
            distance_matrix,
            expr="1.0 / sqrt(1.0 / (2*eps) * D + 1.0)",
            expr_dict={"eps": self.epsilon},
        )


class InverseQuadraticKernel(RadialBasisKernel):
    r"""Inverse quadratic radial basis kernel.

    .. math::
        K = (\frac{1}{2\varepsilon} \cdot D + 1)^{-1}

    where :math:`D` is the squared Euclidean distance matrix.

    See also super classes :class:`RadialBasisKernel` and :class:`PCManifoldKernel`
    for more functionality and documentation.

    Parameters
    ----------
    epsilon
        kernel scale
    """

    def __init__(self, epsilon: float = 1.0):
        self.epsilon = epsilon
        super().__init__(required_metric="sqeuclidean")

    def evaluate(
        self,
        distance_matrix: Union[np.ndarray, scipy.sparse.csr_matrix],
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
        self.epsilon = self._check_bandwidth_parameter(
            parameter=self.epsilon, name="epsilon"
        )

        return _apply_kernel_function_numexpr(
            distance_matrix,
            expr="1.0 / (1.0 / (2*eps) * D + 1.0)",
            expr_dict={"eps": self.epsilon},
        )


class ThinplateKernel(RadialBasisKernel):
    r"""Thinplate radial basis kernel."""

    def __init__(self, distance=None):
        super().__init__(required_metric="euclidean", distance=distance)

    def evaluate(
        self, distance_matrix: Union[np.ndarray, scipy.sparse.csr_matrix]
    ) -> Union[np.ndarray, scipy.sparse.csr_matrix]:
        kernel_matrix = _apply_kernel_function_numexpr(
            distance_matrix, expr="D**2 * log(D)"
        )
        kernel_matrix[np.isnan(kernel_matrix)] = 0
        return kernel_matrix


class CubicKernel(RadialBasisKernel):
    r"""Cubic radial basis kernel.

    .. math::
        K= D^{3}

    where :math:`D` is the Euclidean distance matrix.

    See also super classes :class:`RadialBasisKernel` and :class:`PCManifoldKernel`
    for more functionality and documentation.
    """

    def __init__(self, distance=None):
        super().__init__(required_metric="euclidean", distance=distance)

    def evaluate(
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

    def __init__(self, distance=None):
        super().__init__(required_metric="euclidean", distance=distance)

    def evaluate(
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


class ContinuousNNKernel(PCManifoldKernel):
    """Compute the continuous `k` nearest-neighbor adjacency graph.

    The continuous `k` nearest neighbor (C-kNN) graph is an adjacency graph (i.e. weights are
    only one or zero). This (un-normalized) graph Laplacian converges spectrally to the
    Laplace-Beltrami operator on the manifold in the large data limit (see reference).

    Parameters
    ----------
    k_neighbor
        For each point the distance to the ``k_neighbor`` nearest neighbor is computed.
        If a sparse matrix is computed (with cut-off distance), then each point must
        have a minimum of `k` neighbors. (see `kmin` parameter in
        :meth:`pcfold.distance.compute_distance_matrix`).

    delta
        Unit-less scale parameter.

    Attributes
    ----------
    reference_dist_knn_
        The `k`-th nearest neighbors stored for the reference dataset in a pairwise
        computation. The data is required for component-wise evaluation.

    References
    ----------
    :cite:t:`berry-2019`
    """

    def __init__(self, k_neighbor: int, delta: float, distance=None):
        if not is_float(delta):
            if is_integer(delta):
                self.delta = float(delta)
            else:
                raise TypeError("delta must be of type float")
        else:
            # make sure to only use Python built-in
            self.delta = float(delta)

        if self.delta <= 0.0:
            raise ValueError(f"parameter '{self.delta=}' must be a positive float")

        if k_neighbor < 1:
            raise ValueError(f"parameter '{k_neighbor=}' must be a positive integer")

        if not is_integer(k_neighbor):
            raise TypeError(f"k_neighbor must be an integer (got {type(k_neighbor)=})")
        else:
            # make sure to only use Python built-in
            self.k_neighbor = int(k_neighbor)

        if distance is None or isinstance(distance, dict):
            distance = distance or {}
            distance.setdefault("kmin", self.k_neighbor)
            distance.setdefault("metric", "euclidean")
            distance = init_distance_algorithm(backend="guess_optimal", **distance)
        elif isinstance(distance, DistanceAlgorithm) and distance.dist_type == "knn":
            if distance.k < self.k_neighbor:
                raise ValueError(
                    f"{self.distance=} must have at least {self.k_neighbor=} neighbors"
                )
        elif (
            isinstance(distance, DistanceAlgorithm) and distance.dist_type == "range-nn"
        ):
            if distance.kmin < self.k_neighbor:
                raise ValueError(
                    f"{self.distance=} must assure that each point has at least "
                    f"{self.k_neighbor=} neighbors (set kmin)"
                )
        else:
            raise TypeError(f"{type(self.distance)=} not understood")
        self.distance = distance
        self.reference_dist_knn_: np.ndarray
        super().__init__(distance=distance)

    def _validate_reference_dist_knn(self, is_pdist, reference_dist_knn):
        if is_pdist and reference_dist_knn is None:
            raise ValueError("For the 'cdist' case 'reference_dist_knn' must be given")

    def _validate(self, distance_matrix, is_pdist):
        if distance_matrix.ndim != 2:
            raise ValueError("distance_matrix must be a two-dimensional array")

        n_samples_Y, n_samples_X = distance_matrix.shape

        if is_pdist:
            if n_samples_Y != n_samples_X:
                raise ValueError(
                    f"If {is_pdist=}, the distance matrix must be square and symmetric."
                )

            if isinstance(distance_matrix, np.ndarray):
                diagonal = np.diag(distance_matrix)
            else:
                diagonal = np.asarray(distance_matrix.diagonal(0))

            if (diagonal != 0).all():
                raise ValueError(
                    f"If {is_pdist=}, distance_matrix must have zeros on diagonal."
                )
        else:
            if self.k_neighbor < 1 or self.k_neighbor > n_samples_X - 1:
                raise ValueError(
                    f"{self.k_neighbor=}' must be in a range between 1 to the number of "
                    f"samples ({distance_matrix.shape[1]=})"
                )

            if self.reference_dist_knn_.shape[0] != n_samples_X:
                raise ValueError(
                    f"{self.reference_dist_knn_.shape[0]} "
                    f"must have {distance_matrix.shape[1]=} samples"
                )

    def __call__(
        self,
        X: np.ndarray,
        Y: Optional[np.ndarray] = None,
        **kernel_kwargs,
    ):
        """Compute (sparse) adjacency graph to describe point neighborhood.

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

        **kernel_kwargs: Dict[str, object]
            ignored

        Returns
        -------
        scipy.sparse.csr_matrix
            Sparse adjacency matrix describing the unweighted, undirected continuous
            nearest neighbor graph.

        Optional[Dict[str, numpy.ndarray]]
            For a pair-wise kernel evaluation, a dictionary with key
            `reference_dist_knn` with the `k`-the nearest neighbors for each point is
            returned.
        """
        self._read_kernel_kwargs(attrs=None, kernel_kwargs=kernel_kwargs)
        is_pdist = Y is None

        distance_matrix = self.distance(X, Y)
        return self.evaluate(distance_matrix, is_pdist=is_pdist)

    def evaluate(self, distance_matrix, is_pdist=False) -> scipy.sparse.csr_matrix:
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
        """
        self._validate(distance_matrix=distance_matrix, is_pdist=is_pdist)
        self._required_attrs(["reference_dist_knn_"], is_fit=is_pdist)

        dist_knn = _kth_nearest_neighbor_dist(distance_matrix, self.k_neighbor)

        distance_factors = _symmetric_matrix_division(
            distance_matrix,
            vec=np.sqrt(dist_knn),
            is_symmetric=self.distance.is_symmetric,
            vec_right=np.sqrt(self.reference_dist_knn_) if not is_pdist else None,
        )

        if isinstance(distance_factors, np.ndarray):
            kernel_matrix = scipy.sparse.csr_matrix(
                distance_factors < self.delta, dtype=bool
            )
        else:
            assert isinstance(distance_factors, scipy.sparse.csr_matrix)
            distance_factors.data = (distance_factors.data < self.delta).astype(bool)
            distance_factors.eliminate_zeros()
            kernel_matrix = distance_factors

        if is_pdist:
            self.reference_dist_knn_ = dist_knn

        return kernel_matrix


class MahalanobisKernel(PCManifoldKernel):  # pragma: no cover
    """# TODO - description
    # TODO - citations.

    Parameters
    ----------
    epsilon
        The kernel bandwidth. If "None" (default), it will be estimated from the
        mahalanobis distance matrix, using the median of 10-th nearest neighbor distances.
    distance_metric
        distance metric to use in the pre-computation of the neighborhoods.
        Default: "euclidean"
    cov_matrices
        N*m*m array of N covariance matrices of shape m*m each.
    """

    def __init__(self, epsilon=None, distance=None):
        warnings.warn(
            f"Class '{MahalanobisKernel}' is marked as experimental. This means "
            f"the intended functionality may not be complete and there is no sufficient "
            f"testing. Use class with caution!",
            stacklevel=2,
        )

        self.epsilon = epsilon
        super().__init__(is_symmetric=True, distance=distance)

    def __call__(
        self, X, Y=None, **kernel_kwargs
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
            not supplied

        Returns
        -------
        Union[np.ndarray, scipy.sparse.csr_matrix]
            Kernel matrix of shape `(n_samples_Y, n_samples_X)`. If cut-off is
            specified in `dist_kwargs`, then the matrix is sparse.
        """
        # if not("covariance_matrices" in kernel_kwargs):
        #    raise ValueError(f"invalid kwargs {kernel_kwargs}, must have covariance_matrices")

        X = np.atleast_2d(X)

        if Y is not None:
            Y = np.atleast_2d(Y)

        # TODO: Can they also be computed within the kernel?
        cov_matrices = self._read_kernel_kwargs(
            attrs=["cov_matrices"], kernel_kwargs=kernel_kwargs
        )

        distance_matrix = self.distance(X, Y)

        # TODO: the kernel can not handle the out-of-sample case if Y is not None
        kernel_matrix = self.evaluate(distance_matrix, X, cov_matrices)
        return kernel_matrix

    def evaluate(
        self, distance_matrix, X=None, cov_matrices=None
    ) -> Union[np.ndarray, scipy.sparse.csr_matrix]:
        """Replace the given distance_matrix with the mahalanobis kernel matrix."""
        assert X is not None and cov_matrices is not None

        # TODO: efficiency
        #    -- only compute upper triangle from symmetric distance matrix?
        #    -- possible to vectorize the inner loop in evaluate()?
        #    -- parallelize / use numba
        #    -- efficiently compute cov-matrices?
        # TODO: out-of-sample property is currently missing
        # TODO: compute and store cov-matrices for training data within kernel?
        # TODO: avoid to *only* consider sparse distance matrices -> make it a case
        #       distinction between dense and sparse
        # TODO: require epsilon (no additional routine for estimating epsilon)

        # Need only connectivity matrix, NOT distance matrix!!
        distance_matrix = scipy.sparse.csr_matrix(distance_matrix)

        for row_idx in range(distance_matrix.shape[0]):
            row = distance_matrix.getrow(row_idx)

            row_point = X[row_idx, :]
            row_cov_matrix = cov_matrices[row_idx, :, :]

            # iterate through non-zero entries in data
            for data_index in range(len(row.data)):
                col_idx = row.indices[data_index]
                if col_idx != row_idx:
                    col_point = X[col_idx, :]
                    col_cov_matrix = cov_matrices[col_idx, :, :]

                    point_diff = row_point - col_point

                    # dist = np.sqrt(0.5 * ||x-y|| @ (V_x + V_y) @ ||x-y||)
                    mahalanobis_distance = np.sqrt(
                        0.5
                        * point_diff
                        @ (row_cov_matrix + col_cov_matrix)
                        @ point_diff.T
                    )

                    # TODO: the indexing could be improved!
                    distance_matrix.data[
                        distance_matrix.indptr[row_idx] : distance_matrix.indptr[
                            row_idx + 1
                        ]
                    ][data_index] = mahalanobis_distance

        _epsilon = self.epsilon
        if _epsilon is None:
            from datafold.pcfold.estimators import estimate_cutoff, estimate_scale

            cut_off = estimate_cutoff(X, k=25, distance_matrix=distance_matrix)
            _epsilon = estimate_scale(X, cut_off=cut_off)

        # convert distance to kernel
        # distance_matrix.data = np.exp(-np.square(distance_matrix.data) / (2 * _epsilon))
        # TODO: check: above we have sqrt around the mahalanobis_distance, here we have
        #  a square -- shouldn't this cancel out?
        distance_matrix.data = np.square(distance_matrix.data, out=distance_matrix.data)
        kernel_matrix = _apply_kernel_function_numexpr(
            distance_matrix=distance_matrix,
            expr="exp((- 1. / (2*eps)) * D)",
            expr_dict={"eps": _epsilon},
        )

        # TODO: this operation fails for a k-NN sparse matrix (one that is not symmetric)
        # make it symmetric
        kernel_matrix = scipy.sparse.lil_matrix(kernel_matrix)
        kernel_matrix = scipy.sparse.csr_matrix(0.5 * (kernel_matrix + kernel_matrix.T))

        return kernel_matrix

    def eval2(
        self, X, distance_matrix, cov_matrices, dist_kwargs
    ) -> Union[np.ndarray, scipy.sparse.csr_matrix]:
        """Replace the given distance_matrix with the mahalanobis kernel matrix."""
        distance_matrix = scipy.sparse.csr_matrix(distance_matrix)

        for row in range(distance_matrix.shape[0]):
            r = distance_matrix.getrow(row)

            p1 = X[row, :]
            c1 = cov_matrices[row, :, :]
            for d_index in range(len(r.data)):
                col = r.indices[d_index]
                if col != row:
                    p2 = X[col, :]
                    c2 = cov_matrices[col, :, :]
                    diff = p1 - p2
                    M_distance = np.sqrt(1 / 2 * diff @ (c1 + c2) @ diff.T)

                    distance_matrix.data[
                        distance_matrix.indptr[row] : distance_matrix.indptr[row + 1]
                    ][d_index] = M_distance

        if dist_kwargs is None:
            dist_kwargs = {}

        _epsilon = self.epsilon
        if _epsilon is None:
            from datafold.pcfold.estimators import estimate_cutoff, estimate_scale

            cut_off = estimate_cutoff(
                X, k=dist_kwargs.get("kmin", 25), distance_matrix=distance_matrix
            )
            _epsilon = estimate_scale(X, cut_off=cut_off)

        # convert distance to kernel
        distance_matrix.data = np.exp(
            -np.power(distance_matrix.data, 2) / (2 * _epsilon)
        )

        # make it symmetric
        distance_matrix = scipy.sparse.lil_matrix(distance_matrix)
        distance_matrix = scipy.sparse.csr_matrix(
            0.5 * (distance_matrix + distance_matrix.T)
        )

        return distance_matrix


class DmapKernelFixed(BaseManifoldKernel):
    """Diffusion map kernel with fixed kernel bandwidth.

    This (meta) kernel wraps another kernel to describe a diffusion process.

    Parameters
    ----------
    internal_kernel
        Kernel that describes the proximity between data points.

    is_stochastic
        If True, the kernel matrix is normalized such that the rows sum up to one.

    alpha
        Degree of re-normalization of sampling density in point cloud. The parameter ``alpha``
        must be a float in [0, 1].

    symmetrize_kernel
        If True, perform a conjugate transformation which can improve numerical
        stability for follow-up operations (such as computing the kernel eigenpairs). The
        matrix to change the basis back is stored in the object attribute
        ``basis_change_matrix_``.

    Attributes
    ----------
    row_sums_alpha_: Optional[np.ndarray]
        Row sum values computed for the re-normalization during an initial pair-wise kernel
        computation. The parameter is mandatory for the component-wise kernel
        evaluation and if `alpha>0`.

    basis_change_matrix_: Optional[scipy.sparse.dia_matrix]
        Basis change matrix (sparse diagonal) if `is_symmetrize=True` and only
        set if the kernel matrix is a symmetric conjugate of the true
        diffusion kernel matrix. Required to recover the diffusion map eigenvectors
        from the eigenvectors of the symmetric conjugate matrix (see :cite:t:`rabin-2012`).

    See Also
    --------
    :py:class:`DiffusionMaps`

    References
    ----------
    :cite:t:`coifman-2006,rabin-2012`
    """

    def __init__(
        self,
        internal_kernel: PCManifoldKernel,
        *,
        is_stochastic: bool = True,
        alpha: float = 1.0,
        symmetrize_kernel: bool = True,
    ):
        check_scalar(
            alpha,
            name="alpha",
            target_type=(np.integer, int, np.floating, float),
            min_val=0,
            max_val=1,
        )

        self.alpha = alpha
        self.symmetrize_kernel = symmetrize_kernel
        self.internal_kernel = internal_kernel

        is_symmetric = symmetrize_kernel or not is_stochastic
        super().__init__(is_symmetric=is_symmetric, is_stochastic=is_stochastic)

    def __repr__(self):
        return super().__repr__(print_distance=False)

    @property  # type: ignore
    def distance(self):
        return self.internal_kernel.distance

    @distance.setter
    def distance(self, d):
        # do not use distance here (but forward the distance from the internal kernel
        return None

    @property
    def is_conjugate(self) -> bool:
        """Indicates whether a symmetric matrix is computed that is conjugate to a
        (non-symmetric) stochastic kernel matrix.
        """
        # If the kernel is made stochastic, it looses the symmetry, if symmetric_kernel
        # is set to True, then apply a symmetry transformation
        return (
            self.is_stochastic
            and self._is_symmetric_kernel
            and self.distance.is_symmetric
        )

    def _normalize_sampling_density(
        self, kernel_matrix: Union[np.ndarray, scipy.sparse.csr_matrix], is_pdist: bool
    ) -> tuple[Union[np.ndarray, scipy.sparse.csr_matrix], Optional[np.ndarray]]:
        """Normalize (sparse/dense) kernels with positive `alpha` value. This is also
        referred to a 'renormalization' of sampling density.
        """
        if not is_pdist:
            row_sums_alpha_fit = self.row_sums_alpha_

            if self.row_sums_alpha_.shape[0] != kernel_matrix.shape[1]:
                raise ValueError("'kernel_matrix' does not have the correct shape")
        else:
            row_sums_alpha_fit = None

        row_sums = kernel_matrix.sum(axis=1)

        if scipy.sparse.issparse(kernel_matrix):
            # np.matrix to np.ndarray
            # (np.matrix is deprecated but still used in scipy.sparse)
            row_sums = row_sums.A1

        if self.alpha < 1:
            if row_sums.dtype.kind != "f":
                # This is required for case when 'row_sums' contains boolean or integer
                # values; for inplace operations the type has to be the same
                row_sums = row_sums.astype(float)

            row_sums_alpha = np.power(row_sums, self.alpha, out=row_sums)
        else:  # no need to power with 1
            row_sums_alpha = row_sums

        normalized_kernel = _symmetric_matrix_division(
            matrix=kernel_matrix,
            vec=row_sums_alpha,
            is_symmetric=self.distance.is_symmetric,
            vec_right=row_sums_alpha_fit,
        )

        if not is_pdist:
            # Set row_sums_alpha to None for security, because in a cdist-case (if
            # row_sums_alpha_fit) there is no need to further process row_sums_alpha, yet.
            row_sums_alpha = None

        return normalized_kernel, row_sums_alpha

    def _normalize(
        self,
        kernel_matrix: KernelType,
        is_pdist: bool,
    ):
        # defaults to None -- only required for a symmetric conjugate kernel matrix
        basis_change_matrix = None

        # defaults ot None -- only required if alpha>0 and normalize is called later for a
        # cdist case
        row_sums_alpha = None

        if self.is_stochastic:
            if self.alpha > 0:
                # if pdist: kernel is still symmetric after this function call
                (
                    kernel_matrix,
                    row_sums_alpha,
                ) = self._normalize_sampling_density(kernel_matrix, is_pdist)

            if is_pdist and self.is_conjugate:
                # Increases numerical stability when solving the eigenproblem
                # Note1: when using the (symmetric) conjugate matrix, the eigenvectors
                #        have to be transformed back to match the original
                # Note2: the similarity transform only works for the is_pdist case
                #        (for cdist, there is no symmetric kernel in the first place,
                #        because it is generally rectangular and does not include self
                #        points)
                (
                    kernel_matrix,
                    basis_change_matrix,
                ) = _conjugate_stochastic_kernel_matrix(kernel_matrix)
            else:
                kernel_matrix = _stochastic_kernel_matrix(kernel_matrix)

            # check that if     "is symmetric pdist" -> require basis change
            #            else   no basis change
            assert not (
                (is_pdist and self.is_conjugate) ^ (basis_change_matrix is not None)
            )

        return kernel_matrix, basis_change_matrix, row_sums_alpha

    def _eval_kernel_matrix(self, kernel_matrix, is_pdist):
        self._required_attrs(
            attrs=["row_sums_alpha_", "basis_change_matrix_"], is_fit=is_pdist
        )

        if isinstance(kernel_matrix, pd.DataFrame):
            # store indices and cast to same type later
            _type = type(kernel_matrix)
            rows_idx, columns_idx = kernel_matrix.index, kernel_matrix.columns
            kernel_matrix = kernel_matrix.to_numpy()
        else:
            _type, rows_idx, columns_idx = None, None, None

        kernel_matrix, basis_change_matrix, row_sums_alpha = self._normalize(
            kernel_matrix,
            is_pdist=is_pdist,
        )

        if rows_idx is not None and columns_idx is not None:
            kernel_matrix = _type(kernel_matrix, index=rows_idx, columns=columns_idx)

        if is_pdist:
            self.row_sums_alpha_ = row_sums_alpha
            self.basis_change_matrix_ = basis_change_matrix

        return kernel_matrix

    def __call__(
        self,
        X: np.ndarray,
        Y: Optional[np.ndarray] = None,
        **kernel_kwargs,
    ) -> Union[np.ndarray, scipy.sparse.csr_matrix]:
        """Compute kernel matrix.

        .. note::
            Before evaluating the component-wise kernel matrix (with :code:`kernel(X,Y)`) it
            is first required to evaluate the pairwise kernel matrix (with :code:`kernel(X)`)
            to set necessary attributes from the reference points.

        Parameters
        ----------
        X
            Reference point cloud of shape `(n_samples_X, n_features_X)`.

        Y
            Query point cloud of shape `(n_samples_Y, n_features_Y)`. If not given,
            then `Y=X`.

        **kernel_kwargs
            passed to `kernel_kwargs` of internal kernel

        Returns
        -------
        `numpy.ndarray`, `scipy.sparse.csr_matrix`
            kernel matrix (or symmetric conjugate of it) with same type and shape as
            `distance_matrix`
        """
        is_pdist = Y is None

        if is_pdist:
            kernel_matrix = self.internal_kernel(X, **kernel_kwargs)
        else:
            kernel_matrix = self.internal_kernel(X, Y, **kernel_kwargs)

        return self._eval_kernel_matrix(kernel_matrix=kernel_matrix, is_pdist=is_pdist)

    def evaluate(
        self,
        distance_matrix: Union[np.ndarray, scipy.sparse.csr_matrix],
        is_pdist=False,
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
        `numpy.ndarray`, `scipy.sparse.csr_matrix`
            Kernel matrix with same shape and type of distance matrix.
        """
        kernel_matrix = self.internal_kernel.evaluate(distance_matrix)

        return self._eval_kernel_matrix(
            kernel_matrix=kernel_matrix,
            is_pdist=is_pdist,
        )


class RoselandKernel(PCManifoldKernel):
    """Roseland kernel to describe a diffusion process with respect to landmarks on a point
    cloud.

    This kernel wraps a kernel that describes the point neighborhood. Note that the kernel
    matrix is usually not square because there are less landmarks than points.

    Parameters
    ----------
    internal_kernel
        Kernel that describes the proximity between data points.

    alpha
        Degree of re-normalization of sampling density in point cloud. The parameter ``alpha``
        must be a float inside the interval [0, 1].

    Attributes
    ----------
    landmark_density_: Optional[np.ndarray]
        Point density values at the reference points in a pairwise kernel evaluation. This
        attribute is only set if ``alpha>0``. The values are required for follow-up
        component-wise kernel evaluations.

    stochastic_normalization_: np.ndarray
        Diagonal elements of the normalization matrix for the refrence points in a pairwise
        kernel evaluation. The values are required for follow-up component-wise kernel
        evaluations.

    References
    ----------
    :cite:t:`shen-2020`
    """

    def __init__(self, internal_kernel, alpha=0):
        self.internal_kernel = internal_kernel

        check_scalar(
            alpha,
            "alpha",
            target_type=(np.integer, int, np.floating, float),
            min_val=0,
            max_val=1,
        )
        self.alpha = alpha
        super().__init__(is_symmetric=False, is_stochastic=False)

    def _cast_array(self, obj, is_sparse):
        # Scipy's sparse matrices use the deprecated matrix module from Numpy
        # The attribute A1 turns a matrix object to an array
        return obj.A1 if is_sparse else obj

    def _normalize_density(self, kernel_matrix, is_fit):
        is_sparse = scipy.sparse.spmatrix(kernel_matrix)

        if is_fit:
            landmark_density = self._cast_array(kernel_matrix.sum(axis=0), is_sparse)
        else:
            landmark_density = self.landmark_density_

        data_density = self._cast_array(kernel_matrix.sum(axis=1), is_sparse)

        normalized_kernel_matrix = _symmetric_matrix_division(
            kernel_matrix,
            vec=data_density,
            is_symmetric=False,
            vec_right=landmark_density,
        )

        return normalized_kernel_matrix, landmark_density

    def _compute_normalize_diagonal(self, kernel_matrix, is_fit):
        """This function computes.

        .. code::

            normalize_diagonal = np.sqrt((kernel_matrix @ kernel_matrix.T).sum(axis=1))

        without evaluating :code:`kernel_matrix @ kernel_matrix.T`.

        In the paper this is the diagonal of :math:`D^{-1/2}`.
        """
        is_sparse = scipy.sparse.issparse(kernel_matrix)

        if is_fit:
            stochastic_normalization = self._cast_array(
                kernel_matrix.sum(axis=0), is_sparse
            )
        else:
            stochastic_normalization = self.stochastic_normalization_

        # Alternative computation,
        #   normalize_diagonal = np.sqrt((kernel_matrix @ kernel_matrix.T).sum(axis=1))
        # However, this does not separate the "stochastic_normalization" part,
        # which is required later for an out-of-sample embedding.

        kernel_matrix_adapted = mat_dot_diagmat(kernel_matrix, stochastic_normalization)

        normalize_diagonal = self._cast_array(
            np.sqrt(np.sum(kernel_matrix_adapted, axis=1)), is_sparse
        )

        with np.errstate(divide="ignore"):
            # The reciprocal can be inf when a landmark has no neighbors.
            # These cases are treated separately in 'bool_invalid' below
            normalize_diagonal = np.reciprocal(
                normalize_diagonal, out=normalize_diagonal
            )

            bool_invalid = np.logical_or(
                np.isinf(normalize_diagonal), normalize_diagonal < 0
            )
            normalize_diagonal[bool_invalid] = 0

        return stochastic_normalization, normalize_diagonal

    def _eval_kernel_matrix(self, kernel_matrix):
        """Normalizes the kernel matrix.

        This function performs

        .. math::

            M = D^{-1/2} K

        where matrix :math:`M` is the normalized kernel matrix from :math:`K` by the
        diagonal matrix :math:`D`.

        Parameters
        ----------
        kernel_matrix
            kernel matrix (square or rectangular) to normalize

        Returns
        -------
        Union[np.ndarray, scipy.sparse.spmatrix]
            normalized kernel matrix with type same as `kernel_matrix`
        """
        # indicates whether the kernel has been executed for pairwise before
        is_fit = not hasattr(self, "stochastic_normalization_") or not hasattr(
            self, "landmark_density_"
        )

        self._required_attrs(
            attrs=["stochastic_normalization_", "landmark_density_"], is_fit=is_fit
        )

        if self.alpha > 0:
            kernel_matrix, landmark_density = self._normalize_density(
                kernel_matrix, is_fit=is_fit
            )
        else:
            landmark_density = None

        (
            stochastic_normalization,
            normalize_diagonal,
        ) = self._compute_normalize_diagonal(kernel_matrix=kernel_matrix, is_fit=is_fit)

        kernel_matrix = diagmat_dot_mat(
            normalize_diagonal, kernel_matrix, out=kernel_matrix
        )

        if is_fit:
            self.stochastic_normalization_ = stochastic_normalization
            self.landmark_density_ = landmark_density

        self.normalize_diagonal_ = normalize_diagonal

        return kernel_matrix

    def __call__(
        self,
        X: np.ndarray,
        Y: Optional[np.ndarray] = None,
        **kernel_kwargs,
    ):
        """Compute kernel matrix.

        Parameters
        ----------
        X
            Landmark points of shape `(n_samples_X, n_features_X)`.

        Y
            Query point cloud of shape `(n_samples_Y, n_features_Y)`. Note that `Y` is not
            optional here.

        **kernel_kwargs
            None

        Returns
        -------
        `numpy.ndarray`, `scipy.sparse.csr_matrix`
            kernel matrix (or symmetric conjugate of it) with same type and shape of
            `(n_samples_X, n_samples_Y)`.
        """
        if Y is None:
            raise ValueError(
                "For the landmarked kernel parameter Y must always be provided"
            )

        kernel_matrix = self.internal_kernel(X, Y=Y)
        return self._eval_kernel_matrix(kernel_matrix=kernel_matrix)

    def evaluate(
        self, distance_matrix: Union[np.ndarray, scipy.sparse.csr_matrix], is_fit=False
    ):
        """Evaluate kernel on pre-computed distance matrix.

        Parameters
        ----------
        distance_matrix
            Matrix of shape `(n_samples_Y, n_samples_X)`.

        is_fit:
            If True,

        Returns
        -------
        `np.ndarray`, `scipy.sparse.csr_matrix`
            kernel matrix with same type and shape as distance matrix
        """
        kernel_matrix = self.internal_kernel.evaluate(distance_matrix)
        return self._eval_kernel_matrix(kernel_matrix=kernel_matrix)


class ConeKernel(TSCManifoldKernel):
    r"""Compute a dynamically adapted cone kernel on time series collection data.

    The equations below describe the kernel evaluation and are taken from the referenced
    paper below.

    A single kernel evaluation between time series samples :math:`x` and :math:`y` is computed
    with

    .. math::
        K(x, y) = \exp
        \left(
        -\frac{\vert\vert \omega_{ij}\vert\vert^2}
        {\varepsilon \Delta t^2 \vert\vert \xi_i \vert\vert \vert\vert \xi_j \vert\vert }
        \left[ (1-\zeta \cos^2 \theta_i)(1-\zeta \cos^2 \theta_j) \right]^{0.5}
        \right)

    where,

    .. math::
        \cos \theta_i =
        \frac{(\xi_i, \omega_{ij})}
        {\vert\vert \xi_i \vert\vert \vert\vert \omega_{ij} \vert\vert}

    is the angle between samples,

    .. math::
        \omega_{ij} = y - x

    is a difference vector between the point pairs,

    .. math::
        \Delta t

    is the (constant) time sampling in the time series,

    .. math::
        \varepsilon

    is an additional scaling parameter of the kernel bandwidth,

    .. math::
        \zeta

    is the parameter to control the angular influence, and

    .. math::
        \xi_i = \Delta_p x_i = \sum_{j=-p/2}^{p/2} w_j x_{i+j}

    is the approximation of the dynamical vector field. The approximation is carried
    out with :math:`\Delta_p`, a :math:`p`-th order accurate central finite difference
    (in a sense that :math:`\frac{\xi}{\Delta t} + \mathcal{O}(\Delta t^p)`) with
    associated weights :math:`w`.

    .. note::
        In the centered finite difference the time values are shifted such that no
        samples are taken from the future. For exmaple, for the scheme
        :math:`x_{t+1} - x_{t-1}`, at time :math:`t`, then the new assigned time value
        is `t+1`. See also :py:meth:`.TSCAccessor.time_derivative`.

    Parameters
    ----------
    zeta
        A scalar between :math:`[0, 1)` to control the angular influence. The
        weight from one point to a neighboring point is increased if the relative
        displacement vector is aligned with the dynamical flow. The special case of
        `zeta=0`, corresponds to the so-called "Non-Linear Laplacian Spectral
        Analysis" kernel (NLSA).

    epsilon
        An additional scaling parameter with which the kernel scale can be adapted to
        the actual time sampling frequency.

    fd_accuracy
        The accuracy of the centered finite difference scheme (:math:`p`
        in the description). Note, that the higher the order the more smaples are
        required in a warm-up phase, where the centered scheme cannot be evaluated with
        the given accuracy. All samples from this warm-up phase are dropped in the
        kernel evaluation.

    Attributes
    ----------
    timederiv_X_: np.ndarray
        The time derivative from the finite difference scheme for the reference data `X`.
        Required for the component-wise evaluation and only available after a pairse
        evaluation of the kernel.

    norm_timederiv_X_: np.ndarray
        Norm of the time derivative for the reference data `X`. Required for the
        component-wise evaluation and only available after a pairse evaluation of the kernel.

    References
    ----------
    :cite:t:`giannakis-2015` (the equations are taken from the
    `arXiv version <https://arxiv.org/abs/1403.0361>`__)
    """

    def __init__(
        self,
        zeta: float = 0.0,
        epsilon: float = 1.0,
        fd_accuracy: int = 4,
        distance=None,
    ):
        self.zeta = zeta
        self.epsilon = epsilon
        self.fd_accuracy = fd_accuracy

        distance = distance or {}

        distance.setdefault("metric", "sqeuclidean")

        cut_off = distance.get("cut_off", None)
        if cut_off is not None and not np.isinf(cut_off):
            raise NotImplementedError(
                "Sparse kernel matrix (by setting cut_off) is currently not implemented!"
            )

        self.timederiv_X_: np.ndarray
        self.norm_timederiv_X_: np.ndarray

        super().__init__(distance=distance)

    def _validate_parameter(self, X, Y):
        # cannot import in top of file, because this creates circular imports
        from datafold.pcfold.timeseries.collection import TSCDataFrame

        check_scalar(
            self.zeta,
            name="zeta",
            target_type=(float, np.floating, int, np.integer),
            min_val=0.0,
            max_val=1.0,
            include_boundaries="left",
        )

        check_scalar(
            self.epsilon,
            name="epsilon",
            target_type=(float, np.floating, int, np.integer),
            min_val=0,
        )

        check_scalar(
            self.fd_accuracy,
            "fd_accuracy",
            target_type=(int, np.integer),
            min_val=1,
        )

        # make sure to only deal with Python built-in types
        self.zeta = float(self.zeta)
        self.epsilon = float(self.epsilon)
        self.fd_accuracy = int(self.fd_accuracy)

        if not isinstance(X, TSCDataFrame):
            raise TypeError(f"X must be a TSCDataFrame (got: {type(X)})")

        if Y is not None and not isinstance(Y, TSCDataFrame):
            raise TypeError(f"Y must be a TSCDataFrame (got: {type(X)}")

        if Y is not None:
            is_df_same_index(X, Y, check_index=False, check_column=True, handle="raise")

        # checks that if scalar, if yes returns delta_time
        if Y is None:
            X_dt = X.tsc.check_const_time_delta()
        else:
            X_dt, _ = TSCAccessor.check_equal_delta_time(
                X,
                Y,
                atol=1e-15,
                require_const=True,
            )

        # return here to not compute delta_time again
        return X_dt

    def _compute_distance_and_cosinus_matrix(
        self,
        Y_numpy: np.ndarray,
        timederiv_Y: np.ndarray,
        norm_timederiv_Y: np.ndarray,
        X_numpy: Optional[np.ndarray] = None,
        distance_matrix=None,
    ):
        if X_numpy is None:
            X_numpy = Y_numpy

        is_compute_distance = distance_matrix is None

        # pre-allocate cosine- and distance-matrix
        cos_matrix = np.zeros((Y_numpy.shape[0], X_numpy.shape[0]))

        if is_compute_distance:
            distance_matrix = np.zeros_like(cos_matrix)

        # define names and init as None to already to use in "out"
        diff_matrix, denominator, zero_mask = [None] * 3

        for row_idx in range(cos_matrix.shape[0]):
            diff_matrix = np.subtract(X_numpy, Y_numpy[row_idx, :], out=diff_matrix)

            # distance matrix is not computed via "compute_distance_matrix" function
            if is_compute_distance:
                distance_matrix[row_idx, :] = scipy.linalg.norm(
                    diff_matrix, axis=1, check_finite=False
                )

            # norm of time_derivative * norm_difference
            denominator = np.multiply(
                norm_timederiv_Y[row_idx], distance_matrix[row_idx, :], out=denominator
            )

            # nominator: scalar product (time_derivative, differences)
            # in paper: (\xi, \omega)
            cos_matrix[row_idx, :] = np.dot(
                timederiv_Y[row_idx, :],
                diff_matrix.T,
                out=cos_matrix[row_idx, :],
            )

            # special handling of (almost) duplicates -> denominator by zero leads to nan
            zero_mask = np.less_equal(denominator, 1e-14, out=zero_mask)
            cos_matrix[row_idx, zero_mask] = 0.0  # -> np.cos(0) = 1 later
            cos_matrix[row_idx, ~zero_mask] /= denominator[~zero_mask]

        # memory and cache efficient solving with no intermediate memory allocations:
        # cos_matrix = 1 - self.zeta * np.square(np.cos(cos_matrix))
        cos_matrix = np.cos(cos_matrix, out=cos_matrix)
        cos_matrix = np.square(cos_matrix, out=cos_matrix)
        cos_matrix = np.multiply(self.zeta, cos_matrix, out=cos_matrix)
        cos_matrix = np.subtract(1.0, cos_matrix, out=cos_matrix)

        if not np.isfinite(cos_matrix).all():
            raise ValueError("not all finite")

        return cos_matrix, distance_matrix

    def _approx_dynflow(self, X):
        timederiv = X.tsc.time_derivative(
            scheme="center", diff_order=1, accuracy=self.fd_accuracy, shift_index=True
        )
        norm_timederiv = df_type_and_indices_from(
            timederiv,
            values=np.linalg.norm(timederiv, axis=1),
            except_columns=["fd_norm"],
        )

        return timederiv, norm_timederiv

    def __call__(
        self,
        X: pd.DataFrame,
        Y: Optional[pd.DataFrame] = None,
        **kernel_kwargs,
    ):
        """Compute kernel matrix.

        Parameters
        ----------
        X
            The reference time series collection of shape `(n_samples_X, n_features_X)`.
        Y
            The query time series collection of shape `(n_samples_Y, n_features_Y)`. If
            `Y` is not provided, then ``Y=X``.

        **kernel_kwargs
            None

        Returns
        -------
        TSCDataFrame
            The kernel matrix with time information.
        """
        delta_time = self._validate_parameter(X, Y)
        is_pdist = Y is None

        compute_attributes = is_pdist or (
            not hasattr(self, "timederiv_X_") or not hasattr(self, "norm_timederiv_X_")
        )

        if compute_attributes:
            timederiv_X, norm_timederiv_X = self._approx_dynflow(X=X)
        else:
            timederiv_X, norm_timederiv_X = self.timederiv_X_, self.norm_timederiv_X_

        # NOTE: samples are dropped here which are at the time series boundaries. How
        # many, depends on the accuracy level of the time derivative.
        X_numpy = X.loc[timederiv_X.index].to_numpy()

        if is_pdist:
            timederiv_Y = timederiv_X

            if self.zeta != 0.0:
                (
                    cos_matrix_Y_X,
                    distance_matrix,
                ) = self._compute_distance_and_cosinus_matrix(
                    Y_numpy=X_numpy,  # query (Y) = reference (X)
                    timederiv_Y=timederiv_X.to_numpy(),
                    norm_timederiv_Y=norm_timederiv_X.to_numpy(),
                )

                cos_matrix = np.multiply(
                    cos_matrix_Y_X, cos_matrix_Y_X.T, out=cos_matrix_Y_X
                )
                cos_matrix = np.sqrt(cos_matrix, out=cos_matrix)
                # squared Euclidean metric
                distance_matrix = np.square(distance_matrix, out=distance_matrix)
            else:
                distance_matrix = self.distance(X=X_numpy)
                cos_matrix = np.ones((X_numpy.shape[0], X_numpy.shape[0]))

            factor_matrix = _symmetric_matrix_division(
                cos_matrix,
                vec=norm_timederiv_X.to_numpy().ravel(),
                is_symmetric=True,
                vec_right=None,
                scalar=(delta_time**2) * self.epsilon,
                value_zero_division=0,
            )

        else:
            assert isinstance(Y, pd.DataFrame)  # mypy

            timederiv_Y, norm_timederiv_Y = self._approx_dynflow(X=Y)
            Y_numpy = Y.loc[timederiv_Y.index].to_numpy()

            if self.zeta != 0.0:
                (
                    cos_matrix_Y_X,
                    distance_matrix,
                ) = self._compute_distance_and_cosinus_matrix(
                    Y_numpy=Y_numpy,
                    timederiv_Y=timederiv_Y.to_numpy(),
                    norm_timederiv_Y=norm_timederiv_Y.to_numpy(),
                    X_numpy=X_numpy,
                )

                # because of the time derivative cos_matrix is not symmetric between
                #   reference / query set
                # cos_matrix(i,j) != cos_matrix.T(j,i)
                # --> compute from the other way
                cos_matrix_X_Y, _ = self._compute_distance_and_cosinus_matrix(
                    X_numpy,
                    timederiv_Y=timederiv_X.to_numpy(),
                    norm_timederiv_Y=norm_timederiv_X.to_numpy(),
                    X_numpy=Y_numpy,
                    distance_matrix=distance_matrix.T,
                )

                cos_matrix = np.multiply(
                    cos_matrix_Y_X, cos_matrix_X_Y.T, out=cos_matrix_Y_X
                )
                cos_matrix = np.sqrt(cos_matrix, out=cos_matrix)
                # squared Euclidean metric
                distance_matrix = np.square(distance_matrix, out=distance_matrix)
            else:
                distance_matrix = self.distance(X_numpy, Y_numpy)
                cos_matrix = np.ones((Y_numpy.shape[0], X_numpy.shape[0]))

            factor_matrix = _symmetric_matrix_division(
                cos_matrix,
                vec=norm_timederiv_Y.to_numpy().ravel(),
                is_symmetric=False,
                vec_right=norm_timederiv_X.to_numpy().ravel(),
                scalar=(delta_time**2) * self.epsilon,
                value_zero_division=0,
            )

        assert np.isfinite(factor_matrix).all()

        kernel_matrix = _apply_kernel_function_numexpr(
            distance_matrix=distance_matrix,
            expr="exp(-1.0 * D * factor_matrix)",
            expr_dict={"factor_matrix": factor_matrix},
        )

        kernel_matrix = df_type_and_indices_from(
            indices_from=timederiv_Y,
            values=kernel_matrix,
            except_columns=[f"X{i}" for i in np.arange(X_numpy.shape[0])],
        )

        if compute_attributes:
            self.timederiv_X_ = timederiv_X
            self.norm_timederiv_X_ = norm_timederiv_X

        return kernel_matrix


class DmapKernelVariable(BaseManifoldKernel):  # pragma: no cover
    """Diffusion maps kernel with variable kernel bandwidth.

    .. warning::
        This class is not documented. Contributions are welcome
            * documentation
            * unit- or functional-testing

    References
    ----------
    :cite:`berry-2015,berry-2016`

    See Also
    --------
    :py:class:`DiffusionMapsVariable`

    """

    def __init__(
        self, epsilon, k, expected_dim, beta, symmetrize_kernel, distance=None
    ):
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

        # TODO: this overwrites the user input -- also allow type DistanceAlgorithm
        #  (e.g. sparse matrix)
        distance = {}
        distance["backend"] = "brute"
        distance["metric"] = "sqeuclidean"
        distance["cut_off"] = None

        # TODO: currently the kernel computes the LB generator (i.e. is_stochastic must be
        #  False, if the operator is computed, then max(eigval)==1
        super().__init__(
            is_symmetric=symmetrize_kernel, is_stochastic=False, distance=distance
        )

    def is_symmetric_transform(self, is_pdist):
        # If the kernel is made stochastic, it looses the symmetry, if symmetric_kernel
        # is set to True, then apply the symmetry transformation
        return is_pdist and self._is_symmetric_kernel

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
            rho0 = np.sqrt(np.mean(distance_matrix**2, axis=1))
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
                bool_mask = ~np.diag(np.ones(nr_samples)).astype(bool)
                distance_matrix = distance_matrix[bool_mask].reshape(
                    distance_matrix.shape[0], distance_matrix.shape[1] - 1
                )

            # experimental: --------------------------------------------------------------
            # paper: in var-bw paper (ref2) pdfp. 7
            # it is mentioned to IGNORE non-zero entries -- this is not detailed more.
            # a consequence is that the NN and kernel looses symmetry, so does (K+K^T) / 2
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
        eps0 = meanrho0**2

        expon_matrix = _symmetric_matrix_division(
            matrix=-distance_matrix, vec=rho0tilde, is_symmetric=True, scalar=2 * eps0
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
        """The bandwidth function for K_eps_s."""
        rho = np.power(q0, self.beta)

        # Division by rho-mean is not in papers, but in berry code (ref3)
        return rho / np.mean(rho)

    def _compute_kernel_eps_s(self, distance_matrix, rho):
        expon_matrix = _symmetric_matrix_division(
            matrix=distance_matrix, vec=rho, is_symmetric=True, scalar=-4 * self.epsilon
        )
        kernel_eps_s = np.exp(expon_matrix, out=expon_matrix)
        return kernel_eps_s

    def _compute_q_eps_s(self, kernel_eps_s, rho):
        rho_power_dim = np.power(rho, self.expected_dim)[:, np.newaxis]
        q_eps_s = np.sum(kernel_eps_s / rho_power_dim, axis=1)
        return q_eps_s

    def _compute_kernel_eps_alpha_s(self, kernel_eps_s, q_eps_s):
        kernel_eps_alpha_s = _symmetric_matrix_division(
            matrix=kernel_eps_s, vec=np.power(q_eps_s, self.alpha), is_symmetric=True
        )

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

    def __call__(self, X, Y=None, **kernel_kwargs):
        self._read_kernel_kwargs(attrs=None, kernel_kwargs=kernel_kwargs)

        if Y is not None:
            raise NotImplementedError("cdist case is currently not implemented!")

        if self.k > X.shape[0]:
            raise ValueError(
                f"nr of nearest neighbors (self.k={self.k}) "
                f"is larger than number of samples (={X.shape[0]})"
            )

        distance_matrix = self.distance(X, Y)

        (
            operator_l_matrix,
            self.basis_change_matrix_,
            self.rho0_,
            self.rho_,
            self.q0_,
            self.q_eps_s_,
        ) = self.evaluate(distance_matrix)

        return operator_l_matrix

    def evaluate(self, distance_matrix: Union[np.ndarray, scipy.sparse.csr_matrix]):
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

        # can be expensive
        assert is_symmetric_matrix(kernel_eps_alpha_s)

        if self._is_symmetric_kernel:
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
