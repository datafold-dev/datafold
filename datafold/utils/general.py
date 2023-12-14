#!/usr/bin/env python3

from typing import Optional, Union

import numpy as np
import numpy.testing as nptest
import pandas as pd
import pandas.testing as pdtest
import scipy.sparse
from sklearn.utils.validation import check_scalar


def series_if_applicable(ds: Union[pd.Series, pd.DataFrame]):
    """Turns a DataFrame with only one column into a :class:`pandas.Series`."""
    import datafold.pcfold

    if isinstance(ds, pd.Series):
        pass  # do nothing and return same object

    elif isinstance(ds, pd.DataFrame) and not isinstance(
        ds, datafold.pcfold.TSCDataFrame
    ):
        # Make to pd.Series of pd.DataFrame -- but not from a TSCDataFrame), as this
        # has no "TSCSeries" (yet).
        if ds.shape[1] == 1:
            # column slice is a Series in Pandas
            ds = ds.iloc[:, 0]
    else:
        raise TypeError(f"type={type(ds)} not supported")

    return ds


def assert_equal_eigenvectors(eigvec1, eigvec2, tol=1e-14):
    # Allows to also check orthogonality, but is not yet implemented
    norms1 = np.linalg.norm(eigvec1, axis=0)
    norms2 = np.linalg.norm(eigvec2, axis=0)
    eigvec_test = (eigvec1.conj().T @ eigvec2) * np.reciprocal(np.outer(norms1, norms2))

    actual = np.abs(np.diag(eigvec_test))  # -1 is also allowed for same direction
    expected = np.ones(actual.shape[0])

    nptest.assert_allclose(expected, actual, atol=tol, rtol=0)


def is_df_same_index(
    df_left: pd.DataFrame,
    df_right: pd.DataFrame,
    check_index=True,
    check_column=True,
    check_names=True,
    handle: Optional[str] = "raise",
):
    """Check that two data frames have the same properties (index, columns, names)."""
    assert check_index + check_column >= 1

    is_index_same = True
    is_columns_same = True

    if check_index:
        try:
            pdtest.assert_index_equal(
                df_left.index, df_right.index, check_names=check_names
            )
        except AssertionError as e:
            if handle == "raise":
                raise e
            is_index_same = False

    if check_column:
        try:
            pdtest.assert_index_equal(
                df_left.columns, df_right.columns, check_names=check_names
            )
        except AssertionError:
            if handle == "raise":
                raise
            is_columns_same = False

    return is_index_same and is_columns_same


def is_integer(n: object) -> bool:
    """Checks if `n` is an integer scalar.

    * `n` is a float (built-in) -> check if conversion to int is without precision loss
    + `n` is an integer built-in or numpy.integer

    Parameters
    ----------
    n
        object to check

    Returns
    -------
    bool
        True if `n` is an integer or a float without decimal places.

    """
    return isinstance(n, (int, np.integer)) or (
        isinstance(n, (float, np.floating)) and n / int(n) == 1.0
    )


def is_float(n: object) -> bool:
    """Checks if `n` is a floating scalar.

    Parameters
    ----------
    n
        object to check

    Returns
    -------
    bool
        True if `n` is a float.

    """
    return isinstance(n, (float, np.floating))


def is_scalar(n: object):
    """Checks if `n` is a scalar.

    Parameters
    ----------
    n
        object to check

    Returns
    -------
    bool
        True if `n` is a scalar.

    """
    return is_float(n) or is_integer(n)


def is_matrix(
    matrix,
    name="matrix",
    square=False,
    allow_sparse=False,
    handle: Optional[str] = "raise",
):
    # TODO: also include allowed dtype kind?
    if isinstance(matrix, np.ndarray):
        if matrix.ndim != 2:
            if handle == "raise":
                raise ValueError(
                    f"The matrix '{name}' must have two dimensions. Got {matrix.ndim=}."
                )
            else:
                return False
    elif allow_sparse and scipy.sparse.issparse(matrix):
        pass
    else:
        if handle == "raise":
            raise TypeError(
                f"The parameter '{name}' is not a valid matrix format. Got {type(matrix)=}."
            )
        else:
            return False

    if square and matrix.shape[0] != matrix.shape[1]:
        if handle == "raise":
            raise ValueError(
                f"The parameter '{name}' must be a square matrix. Got {matrix.shape=}"
            )
        else:
            return False

    return True


def is_vector(
    vector,
    name="vector",
    handle: Optional[str] = "raise",
):
    if isinstance(vector, np.ndarray):
        if vector.ndim != 1 or vector.shape[0] < 1:
            if handle == "raise":
                raise ValueError(
                    f"The vector '{name}' must be one dimensional and have at least one "
                    f"element. Got {vector.ndim=} and {vector.shape=}."
                )
            else:
                return False
    else:
        if handle == "raise":
            raise TypeError(
                f"The parameter '{name}' is not a valid matrix format. Got {type(vector)=}."
            )
        else:
            return False

    return True


def if1dim_colvec(vec: np.ndarray) -> np.ndarray:
    if vec.ndim == 1:
        return vec[:, np.newaxis]
    else:
        return vec


def if1dim_rowvec(vec: np.ndarray) -> np.ndarray:
    if vec.ndim == 1:
        return vec[np.newaxis, :]
    else:
        return vec


def projection_matrix_from_feature_names(
    features_all: pd.Index, features_select: pd.Index
) -> scipy.sparse.csr_matrix:
    r"""Compute a sparse projection matrix which maps to selected columns from a matrix.

    .. math::
        A \cdot P = A^*

    If matrix :math:`A` has a set of features (column-oriented), then the projection
    matrix :math:`P` selects the requested sub-selection of features in matrix
    :math:`A^*` (by performing the matrix multiplication).

    Parameters
    ----------
    features_all
        All original feature names.

    features_select
        The (partial or re-ordered) feature names after the projection

    Returns
    -------
    scipy.sparse.csr_matrix
        projection matrix
    """
    project_indices = np.where(np.isin(features_all, features_select))[0]

    if len(project_indices) != len(features_select):
        raise ValueError(
            "Not all features from 'feature_select' are contained in 'features_all'."
        )

    project_matrix = scipy.sparse.lil_matrix((len(features_all), len(features_select)))
    project_matrix[project_indices, np.arange(len(features_select))] = 1
    return project_matrix.tocsr()


def sort_eigenpairs(
    eigenvalues: np.ndarray,
    right_eigenvectors: np.ndarray,
    *,
    left_eigenvectors=None,
    ascending: bool = False,
):
    r"""Sort eigenpairs according to magnitude (absolute value) of corresponding
    eigenvalue.

    The right eigenvectors :math:`\Psi_r` are given by the standard eigenproblem

    .. math::

        A \Psi_r = \Psi_r \Lambda

    The left eigenvectors :math:`\Psi_l` are from the transposed eigenproblem

    .. math::

        \Psi_l A  = \Lambda \Psi_l

    By convention the right eigenvectors are column wise in :math:`\Psi_r` and the left
    eigenvectors are column-wise in :math:`\Psi_l`.

    Parameters
    ----------
    eigenvalues
        complex or real-valued

    right_eigenvectors
        vectors, column-wise

    left_eigenvectors
        vectors, row-wise

    ascending
        If True, sort from low magnitude to high magnitude.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        sorted eigenvalues and -vectors
    """
    is_left_eigvec = left_eigenvectors is not None

    if eigenvalues.ndim != 1:
        raise ValueError("Parameter 'eigenvalues' must be a one dimensional array")

    is_matrix(right_eigenvectors, "right_eigenvectors")

    if is_left_eigvec:
        is_matrix(left_eigenvectors, "left_eigenvectors")

    if eigenvalues.shape[0] != right_eigenvectors.shape[1]:
        raise ValueError(
            f"The number of eigenvalues (={eigenvalues.shape[0]}) does not match the "
            f"number of eigenvectors (={right_eigenvectors.shape[1]})"
        )

    if is_left_eigvec and eigenvalues.shape[0] != left_eigenvectors.shape[0]:
        raise ValueError(
            f"The number of eigenvalues (={eigenvalues.shape[0]}) does not match the "
            f"number of left eigenvectors (={left_eigenvectors.shape[0]})"
        )

    # Sort eigenvectors according to (complex) value of eigenvalue
    #  -- NOTE: sorting according to complex values is preferred over sorting
    #           absolute complex value here. This is because complex conjugate eigenvalues
    #           have the same absolute value which makes sorting unstable (i.e.
    #           there can be two equivalent absolute values but the associate complex
    #           values are at different places after separate sorting)
    idx = np.argsort(eigenvalues)

    if not ascending:
        # creates a view on array and is the most efficient way for reversing order
        # see: https://stackoverflow.com/q/6771428
        idx = idx[::-1]

    if is_left_eigvec:
        return eigenvalues[idx], right_eigenvectors[:, idx], left_eigenvectors[idx, :]
    else:
        return eigenvalues[idx], right_eigenvectors[:, idx]


def mat_dot_diagmat(
    matrix: Union[np.ndarray, scipy.sparse.spmatrix],
    diag_elements: np.ndarray,
    out: Optional[np.ndarray] = None,
) -> Union[np.ndarray, scipy.sparse.spmatrix]:
    """Efficient computation of "(sparse/dense) matrix times diagonal matrix".

    Compute the often common but inefficient way

    .. code::
        matrix @ np.diag(diag_elements)

    by using element-wise computations.

    Parameters
    ----------
    matrix
        Dense matrix of shape `(I,J)`.

    diag_elements
        Diagonal elements in 1 dim. array of `J` elements.

    out
        Select NumPy ndarray (with shape ``(I,J)`` to write the result into. A usual choice is
        setting it to the same matrix as in the argument input. This can improve memory
        efficient as no new memory needs to be allocated. Ignored if matrix is sparse.

    Returns
    -------
    """
    assert diag_elements.ndim == 1 and is_matrix(matrix, "matrix", allow_sparse=True)

    if scipy.sparse.issparse(matrix):
        # out is ignored here, because it is not supported by scipy sparse
        return matrix @ scipy.sparse.diags(diagonals=diag_elements)
    else:
        return np.multiply(diag_elements, matrix, out=out)


def diagmat_dot_mat(
    diag_elements: Union[np.ndarray, scipy.sparse.spmatrix],
    matrix: np.ndarray,
    out=None,
) -> Union[np.ndarray, scipy.sparse.spmatrix]:
    """Efficient computation of "diagonal matrix times (sparse/dense) matrix".

    Compute the often common but inefficient way

    .. code::
        np.diag(diag_elements) @ matrix

    by using element-wise computations.

    Parameters
    ----------
    diag_elements
         Diagonal elements in 1 dim. array of ``I`` elements.

    matrix
        Dense matrix of shape ``(I,J)``.

    out
        Select NumPy ndarray (with shape ``(I,J)`` to write the result into. A usual choice is
        setting it to the same matrix as in the argument input. This can improve memory
        efficient as no new memory needs to be allocated. Ignored if matrix is sparse.

    Returns
    -------
    """
    assert diag_elements.ndim == 1 and is_matrix(matrix, "matrix", allow_sparse=True)

    if scipy.sparse.issparse(matrix):
        # out is not supported by scipy sparse dot
        # sparse.diags does not fill memory with zeros
        return scipy.sparse.diags(diag_elements) @ matrix
    else:
        return np.multiply(matrix, diag_elements[:, np.newaxis], out=out)


def generate_2d_regular_mesh(
    low=(-1, -1), high=(1, 1), n_xvalues=10, n_yvalues=10, feature_names=None
):
    # TODO: possibly attach this to InitialCondition class (could also be generalized
    #   to multiple D)
    x_values = np.linspace(low[0], high[0], num=n_xvalues)
    y_values = np.linspace(low[1], high[1], num=n_yvalues)

    x_mesh, y_mesh = np.meshgrid(x_values, y_values)
    X = np.column_stack((x_mesh.ravel(), y_mesh.ravel()))

    if feature_names is not None:
        # import here to avoid circular imports
        from datafold import InitialCondition

        X = InitialCondition.from_array(X, time_value=0, feature_names=feature_names)

    return X


def df_type_and_indices_from(
    indices_from: pd.DataFrame,
    values: Union[np.ndarray, pd.DataFrame],
    except_index: Optional[Union[pd.Index, list[str]]] = None,
    except_columns: Optional[Union[pd.Index, list[str]]] = None,
):
    # import here to prevent circular imports
    from datafold.pcfold import TSCDataFrame

    if except_index is not None and except_columns is not None:
        raise ValueError(
            "Both parameters 'except_index' and 'except_columns' are provided. For this "
            "function provide none or one of the optional parameters."
        )

    # view input as array (allows for different input, which is
    # compatible with numpy.ndarray
    values = np.asarray(values)

    if except_index is None:
        index = indices_from.index
    else:
        index = except_index

    if except_columns is None:
        columns = indices_from.columns
    else:
        columns = except_columns

    if isinstance(indices_from, TSCDataFrame):
        return TSCDataFrame(data=values, index=index, columns=columns)
    elif isinstance(indices_from, pd.DataFrame):
        return pd.DataFrame(data=values, index=index, columns=columns)
    else:
        raise TypeError(f"The argument type {type(indices_from)=} is invalid.")


def is_symmetric_matrix(
    matrix: Union[np.ndarray, scipy.sparse.csr_matrix], tol: float = 0
) -> bool:
    """Check whether a matrix is symmetric.

    Parameters
    ----------
    matrix
       A square matrix to be checked for symmetry.
    tol
       The tolerance of absolute deviation between corresponding elements
       (k[i,j] and k[j,i]).

    Returns
    -------
    bool
        True if symmetric else False.
    """
    is_matrix(matrix, "matrix", square=True, allow_sparse=True)
    max_abs_deviation = np.max(np.abs(matrix - matrix.T))
    return max_abs_deviation <= tol


def is_stochastic_matrix(
    matrix: Union[np.ndarray, scipy.sparse.csr_matrix], axis=1, tol=1e-15
) -> bool:
    """Check whether a matrix is stochastic.

    A matrix is stochastic if either the columns (axis=1) or the rows (axis=0) sum up to 1.

    Parameters
    ----------
    matrix
         The matrix to be checked for stochasticity.

    axis
       The axis along which to check the stochasticity (0 or row and 1 for column).

    tol
       The tolerance of absolute deviation from the row or column sum from 1.

    Returns
    -------
    bool
        True if matrix is stochastic else False
    """
    is_matrix(matrix, "matrix", allow_sparse=True)

    sum_array = matrix.sum(axis=axis)

    if scipy.sparse.issparse(matrix):
        sum_array = sum_array.A1

    return (np.abs(sum_array - 1) <= tol).all()


def remove_numeric_noise_symmetric_matrix(
    matrix: Union[np.ndarray, scipy.sparse.spmatrix]
) -> Union[np.ndarray, scipy.sparse.spmatrix]:
    r"""Remove numerical noise from (almost) symmetric matrix.

    Even symmetric operations can sometimes introduce noise. The
    operations are often executed in different order, for example, evaluations such as in

    .. math::

        D^{-1} M D^{-1},

    where :math:`D` is a diagonal matrix can break exact floating point symmetry in a
    symmetric matrix :math:`M`.

    This function recovers an exact symmetry of an "almost symmetric" matrix, such as in the
    following situation:

    .. code::
        np.max(np.abs(matrix - matrix.T)) # 1.1102230246251565e-16

    The symmetry is recovered with:

    .. math::
        M_{sym} = \frac{M + M^T}{2}

    Parameters
    ----------
    matrix
        square matrix (dense/sparse) to remove numerical noise from

    Returns
    -------
    Union[numpy.ndarray, scipy.sparse.csr_matrix]
        symmetric matrix without noise
    """
    is_matrix(matrix, "matrix", square=True, allow_sparse=True)

    if scipy.sparse.issparse(matrix):
        # need to preserve stored zeros (e.g. in case of distance matrices)
        # NOTE: there is no check that the sparse structure is symmetric, this leads to
        # wrong results if the symmetric value of a (non-negative) value is not there.
        matrix.data[matrix.data == 0] = np.nan
        matrix = (matrix + matrix.T) / 2.0
        matrix.data[np.isnan(matrix.data)] = 0
    else:
        matrix = np.add(matrix, matrix.T, out=matrix)
        matrix = np.divide(matrix, 2.0, out=matrix)

    return matrix


def random_subsample(
    data: np.ndarray, n_samples: int, random_state: Optional[int] = None
) -> tuple[np.ndarray, np.ndarray]:
    """Random uniform subsample without replacement of data.

    Parameters
    ----------
    data
        point cloud

    n_samples
        Number of points to sample.

    random_state
        Seed for random permutation of sample indices.
        :code:`np.random.default_rng(random_state)`

    Returns
    -------
    numpy.ndarray
        subsampled array

    numpy.ndarray
        indices in the subsample from the original array
    """
    is_matrix(data, "data", allow_sparse=False)

    n_samples_data = data.shape[0]

    check_scalar(
        n_samples,
        name="n_samples",
        target_type=int,
        min_val=1,
        max_val=n_samples_data - 1,
    )

    indices = np.random.default_rng(seed=random_state).permutation(n_samples_data)
    indices = indices[:n_samples]

    return data[indices, :], indices
