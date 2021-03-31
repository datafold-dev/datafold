import abc
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numexpr as ne
import numpy as np
import pandas as pd
import scipy.sparse
import scipy.spatial
from sklearn.gaussian_process.kernels import Kernel
from sklearn.preprocessing import normalize
from sklearn.utils import check_scalar

from datafold.pcfold.distance import compute_distance_matrix
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
        return distance_matrix  # returns actually the kernel
    else:
        expr_dict["D"] = distance_matrix
        return ne.evaluate(expr, expr_dict)


def _symmetric_matrix_division(
    matrix: Union[np.ndarray, scipy.sparse.spmatrix],
    vec: np.ndarray,
    vec_right: Optional[np.ndarray] = None,
    scalar: float = 1.0,
    value_zero_division: Union[str, float] = "raise",
) -> Union[np.ndarray, scipy.sparse.csr_matrix,]:
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

    if (vec == 0.0).any():
        if value_zero_division == "raise":
            raise ZeroDivisionError(
                f"Encountered zero values in division in {(vec == 0).sum()} points."
            )
        else:
            # division results into 'nan' without raising a ZeroDivisionWarning. The
            # nan values will be replaced later
            vec[vec == 0.0] = np.nan

    vec_inv_left = np.reciprocal(vec)

    if vec_right is None:
        vec_inv_right = vec_inv_left.view()
    else:
        vec_right = vec_right.astype(float)
        if (vec_right == 0.0).any():
            if value_zero_division == "raise":
                raise ZeroDivisionError(
                    f"Encountered zero values in division in {(vec == 0).sum()}"
                )
            else:
                vec_right[vec_right == 0.0] = np.inf

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
        left_inv_diag_sparse = scipy.sparse.spdiags(
            vec_inv_left, 0, m=matrix.shape[0], n=matrix.shape[0]
        )
        right_inv_diag_sparse = scipy.sparse.spdiags(
            vec_inv_right, 0, m=matrix.shape[1], n=matrix.shape[1]
        )

        # The performance of DIA-sparse matrices is good if the matrix is actually
        # sparse. I.e. the performance drops for a sparse-dense-sparse multiplication.

        # The zeros are removed in the matrix multiplication, but because 'matrix' is
        # usually a distance matrix we need to preserve the "true zeros"!
        matrix.data[matrix.data == 0] = np.inf
        matrix = left_inv_diag_sparse @ matrix @ right_inv_diag_sparse

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
    if vec_right is None:
        matrix = remove_numeric_noise_symmetric_matrix(matrix)

    if scalar != 1.0:
        scalar = 1.0 / scalar
        matrix = np.multiply(matrix, scalar, out=matrix)

    return matrix


def _conjugate_stochastic_kernel_matrix(
    kernel_matrix: Union[np.ndarray, scipy.sparse.spmatrix]
) -> Tuple[Union[np.ndarray, scipy.sparse.spmatrix], scipy.sparse.dia_matrix]:
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

    if left_vec.dtype.kind != "f":
        left_vec = left_vec.astype(float)

    left_vec = np.sqrt(left_vec, out=left_vec)

    kernel_matrix = _symmetric_matrix_division(
        kernel_matrix, vec=left_vec, vec_right=None
    )

    # This is D^{-1/2} in sparse matrix form.
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
        The distance of the `k`-th nearest neighbor.

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
    @abc.abstractmethod
    def __call__(self, X, Y=None, *, dist_kwargs=None, **kernel_kwargs):
        """Compute kernel matrix.

        If `Y=None`, then the pairwise-kernel is computed with `Y=X`. If `Y` is given,
        then the kernel matrix is computed component-wise with `X` being the reference
        and `Y` the query point cloud.

        Because the kernel can return a variable number of return values, this is
        unified with :meth:`PCManifoldKernel.read_kernel_output`.

        Parameters
        ----------
        args
        kwargs
            See parameter documentation in subclasses.

        Returns
        -------
        """

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

    def _read_kernel_kwargs(self, attrs: Optional[List[str]], kernel_kwargs: dict):

        return_values: List[Any] = []

        if attrs is not None:
            for attr in attrs:
                return_values.append(kernel_kwargs.pop(attr, None))

        if kernel_kwargs != {}:
            raise KeyError(
                f"kernel_kwargs.keys = {kernel_kwargs.keys()} are not " f"supported"
            )

        if len(return_values) == 0:
            return None
        elif len(return_values) == 1:
            return return_values[0]
        else:
            return return_values

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

        if isinstance(
            kernel_output, (pd.DataFrame, np.ndarray, scipy.sparse.csr_matrix)
        ):
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
                "'kernel_output' must be either pandas.DataFrame (incl. TSCDataFrame), "
                "numpy.ndarray or tuple. Please report bug."
            )

        ret_cdist = ret_cdist or {}
        ret_extra = ret_extra or {}

        if not isinstance(
            kernel_matrix, (pd.DataFrame, np.ndarray, scipy.sparse.csr_matrix)
        ):
            raise TypeError(
                f"Illegal type of kernel_matrix (type={type(kernel_matrix)}. "
                f"Please report bug."
            )

        return kernel_matrix, ret_cdist, ret_extra


class PCManifoldKernel(BaseManifoldKernel):
    """Abstract base class for kernels acting on point clouds.

    See Also
    --------
    :py:class:`PCManifold`
    """

    @abc.abstractmethod
    def __call__(
        self,
        X: np.ndarray,
        Y: Optional[np.ndarray] = None,
        *,
        dist_kwargs: Optional[Dict[str, object]] = None,
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
    def eval(
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
    """Abstract base class for kernels acting on time series collections.

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
        dist_kwargs: Optional[Dict[str, object]] = None,
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

    @classmethod
    def _check_bandwidth_parameter(cls, parameter, name) -> float:
        check_scalar(
            parameter,
            name=name,
            target_type=(float, np.floating, int, np.integer),
            min_val=np.finfo(float).eps,
        )
        return float(parameter)

    def __call__(
        self, X, Y=None, *, dist_kwargs=None, **kernel_kwargs
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

        if Y is not None:
            Y = np.atleast_2d(Y)

        distance_matrix = compute_distance_matrix(
            X,
            Y,
            metric=self.distance_metric,
            **dist_kwargs or {},
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
        The kernel scale as a positive float value. Alternatively, a a
        callable can be passed. After computing the distance matrix, the distance matrix
        will be passed to this function i.e. ``function(distance_matrix)``. The result
        of this function must be a again a positive float to describe the kernel scale.
    """

    def __init__(self, epsilon: Union[float, Callable] = 1.0):
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

        if callable(self.epsilon):
            if isinstance(distance_matrix, scipy.sparse.csr_matrix):
                self.epsilon = self.epsilon(distance_matrix.data)
            elif isinstance(distance_matrix, np.ndarray):
                self.epsilon = self.epsilon(distance_matrix)
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
        kernel scale
    """

    def __init__(self, epsilon: float = 1.0):
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
        kernel scale
    """

    def __init__(self, epsilon: float = 1.0):
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

        self.epsilon = self._check_bandwidth_parameter(
            parameter=self.epsilon, name="epsilon"
        )

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
                f"parameter 'k_neighbor={self.k_neighbor}' must be a positive integer"
            )

        if self.delta <= 0.0:
            raise ValueError(
                f"parrameter 'delta={self.delta}' must be a positive float"
            )

        super(ContinuousNNKernel, self).__init__()

    def _validate_reference_dist_knn(self, is_pdist, reference_dist_knn):
        if is_pdist and reference_dist_knn is None:
            raise ValueError("For the 'cdist' case 'reference_dist_knn' must be given")

    def __call__(
        self,
        X: np.ndarray,
        Y: Optional[np.ndarray] = None,
        *,
        dist_kwargs: Optional[Dict] = None,
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

        **kernel_kwargs: Dict[str, object]
            - reference_dist_knn: Optional[np.ndarray]
                Distances to the `k`-th nearest neighbor for each point in `X`. The
                parameter is mandatory if `Y` is not `None`.

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

        is_pdist = Y is None

        reference_dist_knn = self._read_kernel_kwargs(
            attrs=["reference_dist_knn"], kernel_kwargs=kernel_kwargs
        )

        dist_kwargs = dist_kwargs or {}
        # minimum number of neighbors required in sparse case!
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
                    "If is_pdist=True, the distance matrix must be square and symmetric."
                )

            if isinstance(distance_matrix, np.ndarray):
                diagonal = np.diag(distance_matrix)
            else:
                diagonal = np.asarray(distance_matrix.diagonal(0))

            if (diagonal != 0).all():
                raise ValueError(
                    "If is_pdist=True, distance_matrix must have zeros on diagonal."
                )
        else:
            if reference_dist_knn is None:
                raise ValueError(
                    "If is_pdist=False, 'reference_dist_knn' (=None) must be provided."
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
                    "'n_neighbors' must be in a range between 1 to the number of samples."
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
                distance_factors < self.delta, dtype=bool
            )
        else:
            assert isinstance(distance_factors, scipy.sparse.csr_matrix)
            distance_factors.data = (distance_factors.data < self.delta).astype(bool)
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


class DmapKernelFixed(BaseManifoldKernel):
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

        # i) not stochastic -> if the kernel is symmetric the kernel is always
        # symmetric
        # `symmetrize_kernel` indicates if the user wants the kernel to use
        # similarity transformations to solve the
        # eigenproblem on a symmetric kernel (if required).

        # NOTE: a necessary condition to symmetrize the kernel is that the kernel
        # is evaluated pairwise
        #  (i.e. is_pdist = True)
        #     self._is_symmetric = True
        # else:
        #     self._is_symmetric = False

        self.row_sums_init = None

        super(DmapKernelFixed, self).__init__()

    @property
    def is_symmetric(self):
        return self.symmetrize_kernel or not self.is_stochastic

    def is_symmetric_transform(self) -> bool:
        """Indicates whether a symmetric conjugate kernel matrix was computed.

        Returns
        -------

        """

        # If the kernel is made stochastic, it looses the symmetry, if symmetric_kernel
        # is set to True, then apply the the symmetry transformation
        return self.is_stochastic and self.is_symmetric

    def _normalize_sampling_density(
        self,
        kernel_matrix: Union[np.ndarray, scipy.sparse.csr_matrix],
        row_sums_alpha_fit: np.ndarray,
    ) -> Tuple[Union[np.ndarray, scipy.sparse.csr_matrix], Optional[np.ndarray]]:
        """Normalize (sparse/dense) kernels with positive `alpha` value. This is also
        referred to a 'renormalization' of sampling density."""

        if row_sums_alpha_fit is None:
            assert is_symmetric_matrix(kernel_matrix)
        else:
            assert row_sums_alpha_fit.shape[0] == kernel_matrix.shape[1]

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
            vec_right=row_sums_alpha_fit,
        )

        if row_sums_alpha_fit is not None:
            # Set row_sums_alpha to None for security, because in a cdist-case (if
            # row_sums_alpha_fit) there is no need to further process row_sums_alpha, yet.
            row_sums_alpha = None

        return normalized_kernel, row_sums_alpha

    def _normalize(
        self,
        internal_kernel: KernelType,
        row_sums_alpha_fit: np.ndarray,
        is_pdist: bool,
    ):

        # only required for symmetric kernel, return None if not used
        basis_change_matrix = None

        # required if alpha>0 and _normalize is called later for a cdist case
        # set in the pdist, alpha > 0 case
        row_sums_alpha = None

        if self.is_stochastic:

            if self.alpha > 0:
                # if pdist: kernel is still symmetric after this function call
                (internal_kernel, row_sums_alpha,) = self._normalize_sampling_density(
                    internal_kernel, row_sums_alpha_fit
                )

            if is_pdist and self.is_symmetric_transform():
                # Increases numerical stability when solving the eigenproblem
                # Note1: when using the (symmetric) conjugate matrix, the eigenvectors
                #        have to be transformed back to match the original
                # Note2: the similarity transform only works for the is_pdist case
                #        (for cdist, there is no symmetric kernel in the first place,
                #        because it is generally rectangular and does not include self
                #        points)
                (
                    internal_kernel,
                    basis_change_matrix,
                ) = _conjugate_stochastic_kernel_matrix(internal_kernel)
            else:
                internal_kernel = _stochastic_kernel_matrix(internal_kernel)

            # check that if     "is symmetric pdist" -> require basis change
            #            else   no basis change
            assert not (
                (is_pdist and self.is_symmetric_transform())
                ^ (basis_change_matrix is not None)
            )

        if is_pdist and self.is_symmetric:
            assert is_symmetric_matrix(internal_kernel)

        return internal_kernel, basis_change_matrix, row_sums_alpha

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

        if isinstance(kernel_matrix, pd.DataFrame):
            # store indices and cast to same type later
            _type = type(kernel_matrix)
            rows_idx, columns_idx = kernel_matrix.index, kernel_matrix.columns
            kernel_matrix = kernel_matrix.to_numpy()
        else:
            _type, rows_idx, columns_idx = None, None, None

        kernel_matrix, basis_change_matrix, row_sums_alpha = self._normalize(
            kernel_matrix,
            row_sums_alpha_fit=row_sums_alpha_fit,
            is_pdist=is_pdist,
        )

        if rows_idx is not None and columns_idx is not None:
            kernel_matrix = _type(kernel_matrix, index=rows_idx, columns=columns_idx)

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
        *,
        dist_kwargs: Optional[Dict] = None,
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

        **kernel_kwargs: Dict[str, object]
            - internal_kernel_kwargs: Optional[Dict]
                Keyword arguments passed to the set internal kernel.
            - row_sums_alpha_fit: Optional[np.ndarray]
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

        internal_kernel_kwargs, row_sums_alpha_fit = self._read_kernel_kwargs(
            attrs=["internal_kernel_kwargs", "row_sums_alpha_fit"],
            kernel_kwargs=kernel_kwargs,
        )

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


class ConeKernel(TSCManifoldKernel):
    r"""Compute a dynamics adapted cone kernel for time series collection data.

    The equations below describe the kernel evaluation and are taken from the referenced
    paper below.

    A single kernel evaluation between samples :math:`x` and :math:`y` is computed with

    .. math::
        K(x, y) = \exp
        \left(
        -\frac{\vert\vert \omega_{ij}\vert\vert^2}
        {\varepsilon \delta t^2 \vert\vert \xi_i \vert\vert \vert\vert \xi_j \vert\vert }
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
        \delta t

    is the (constant) time sampling in the time series,

    .. math::
        \varepsilon

    is an additional scaling parameter of the kernel bandwidth,

    .. math::
        \zeta

    is the parameter to control the angular influence, and

    .. math::
        \xi_i = \delta_p x_i = \sum_{j=-p/2}^{p/2} w_j x_{i+j}

    is the approximation of the dynamical vector field. The approximation is carried
    out with :math:`\delta_p`, a :math:`p`-th order accurate central finite difference
    (in a sense that :math:`\frac{\xi}{\delta t} + \mathcal{O}(\delta t^p)`) with
    associated weights :math:`w`.

    .. note::
        In the centered finite difference the time values are shifted such that no
        samples are taken from the future. For exmaple, for the scheme
        :math:`x_{t+1} - x_{t-1}`, at time :math:`t`, then the new assigned time value
        is `t+1`. See also :py:meth:`.TSCAccessor.time_derivative`.

    Parameters
    ----------

    zeta
        A scalar between :math:`[0, 1)` that controls the angular influence . The
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

    References
    ----------

    :cite:`giannakis_dynamics-adapted_2015` (the equations are taken from the
    `arXiv version <https://arxiv.org/abs/1403.0361>`__)
    """

    def __init__(self, zeta: float = 0.0, epsilon: float = 1.0, fd_accuracy: int = 4):
        self.zeta = zeta
        self.epsilon = epsilon
        self.fd_accuracy = fd_accuracy

    def _validate_setting(self, X, Y):

        # cannot import in top of file, because this creates circular imports
        from datafold.pcfold.timeseries.collection import TSCDataFrame, TSCException

        check_scalar(
            self.zeta,
            name="zeta",
            target_type=(float, np.floating, int, np.integer),
            min_val=0.0,
            max_val=1.0 - np.finfo(float).eps,
        )

        check_scalar(
            self.epsilon,
            name="epsilon",
            target_type=(float, np.floating, int, np.integer),
            min_val=np.finfo(float).eps,
            max_val=None,
        )

        check_scalar(
            self.fd_accuracy,
            "fd_accuracy",
            target_type=(int, np.integer),
            min_val=1,
            max_val=None,
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
        *,
        dist_kwargs: Optional[Dict[str, object]] = None,
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

        dist_kwargs
            ignored `(The distance matrix is computed as part of the kernel evaluation.
            For now this can only be a dense matrix).`

        **kernel_kwargs: Dict[str, object]
            - timederiv_X
                The time derivative from a finite difference scheme. Required for a
                component-wise evaluation.
            - norm_timederiv_X
                Norm of the time derivative. Required for a component-wise evaluation.

        Returns
        -------
        TSCDataFrame
            The kernel matrix with time information.
        """

        delta_time = self._validate_setting(X, Y)

        timederiv_X, norm_timederiv_X = self._read_kernel_kwargs(
            attrs=["timederiv_X", "norm_timederiv_X"], kernel_kwargs=kernel_kwargs
        )

        is_pdist = Y is None

        if is_pdist:
            timederiv_X, norm_timederiv_X = self._approx_dynflow(X=X)
        else:
            if timederiv_X is None or norm_timederiv_X is None:
                raise ValueError(
                    "For component wise computation the parameters 'timederiv_X' "
                    "and 'norm_timederiv_X' must be provided. "
                )

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
                distance_matrix = compute_distance_matrix(X_numpy, metric="sqeuclidean")
                cos_matrix = np.ones((X_numpy.shape[0], X_numpy.shape[0]))

            factor_matrix = _symmetric_matrix_division(
                cos_matrix,
                vec=norm_timederiv_X.to_numpy().ravel(),
                vec_right=None,
                scalar=(delta_time ** 2) * self.epsilon,
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
                distance_matrix = compute_distance_matrix(
                    X_numpy, Y_numpy, metric="sqeuclidean"
                )
                cos_matrix = np.ones((Y_numpy.shape[0], X_numpy.shape[0]))

            factor_matrix = _symmetric_matrix_division(
                cos_matrix,
                vec=norm_timederiv_Y.to_numpy().ravel(),
                vec_right=norm_timederiv_X.to_numpy().ravel(),
                scalar=(delta_time ** 2) * self.epsilon,
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

        if is_pdist:
            ret_cdist: Optional[Dict[str, Any]] = dict(
                timederiv_X=timederiv_X, norm_timederiv_X=norm_timederiv_X
            )
        else:
            ret_cdist = None

        return kernel_matrix, ret_cdist


class DmapKernelVariable(BaseManifoldKernel):  # pragma: no cover
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

    def __call__(self, X, Y=None, *, dist_kwargs=None, **kernel_kwargs):

        dist_kwargs = dist_kwargs or {}
        cut_off = dist_kwargs.pop("cut_off", None)

        self._read_kernel_kwargs(attrs=None, kernel_kwargs=kernel_kwargs)

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
            X,
            Y,
            metric="sqeuclidean",
            **dist_kwargs,
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

        # can be expensive
        assert is_symmetric_matrix(kernel_eps_alpha_s)

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
