#!/usr/bin/env python3

import warnings
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd
import pandas.testing as pdtest
import scipy.sparse
from sklearn.utils.validation import check_array, check_scalar


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


def is_df_same_index(
    df_left: pd.DataFrame,
    df_right: pd.DataFrame,
    check_index=True,
    check_column=True,
    handle="raise",
):

    assert check_index + check_column >= 1

    is_index_same = True
    is_columns_same = True

    if check_index:
        try:
            pdtest.assert_index_equal(df_left.index, df_right.index, check_names=True)
        except AssertionError:
            if handle == "raise":
                raise
            is_index_same = False

    if check_column:
        try:
            pdtest.assert_index_equal(
                df_left.columns, df_right.columns, check_names=True
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
        isinstance(n, (float, np.floating)) and n.is_integer()
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


def if1dim_colvec(vec: np.ndarray) -> np.ndarray:
    if vec.ndim == 1:
        return vec[:, np.newaxis]
    else:
        return vec


def if1dim_rowvec(vec: np.ndarray):
    if vec.ndim == 1:
        return vec[np.newaxis, :]
    else:
        return vec


def sort_eigenpairs(
    eigenvalues: np.ndarray, eigenvectors: np.ndarray, ascending: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """Sort eigenpairs according to magnitude (absolute value) of corresponding
    eigenvalue.

    Parameters
    ----------

    eigenvalues
        complex or real-valued

    eigenvectors
        vectors, column wise

    ascending
        If True, sort from low magnitude to high magnitude.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        sorted eigenvalues and -vectors
    """
    if eigenvalues.ndim != 1 or eigenvectors.ndim != 2:
        raise ValueError(
            "eigenvalues have to be 1-dim and eigenvectors "
            "2-dim np.ndarray respectively"
        )

    if eigenvalues.shape[0] != eigenvectors.shape[1]:
        raise ValueError(
            f"the number of eigenvalues (={eigenvalues.shape[0]}) does not match the "
            f"number of eigenvectors (={eigenvectors.shape[1]})"
        )

    # Sort eigenvectors according to absolute value of eigenvalue:
    idx = np.abs(eigenvalues).argsort()

    if not ascending:
        # creates a view on array and is most efficient way for reversing order
        # see: https://stackoverflow.com/q/6771428
        idx = idx[::-1]

    return eigenvalues[idx], eigenvectors[:, idx]


def mat_dot_diagmat(
    matrix: np.ndarray, diag_elements: np.ndarray, out: Optional[np.ndarray] = None
) -> np.ndarray:
    """Efficient computation of matrix time diagonal matrix.

    This computes

    .. code::
        matrix @ np.diag(diag_elements)

    by using element-wise computations.

    Parameters
    ----------
    matrix, diag_elements
        elements of computation

    out
        select where to write the result, Usual choice is setting it to matrix,
        which simply overwrites matrix and is therefore more memory efficient

    Returns
    -------
    """
    assert diag_elements.ndim == 1 and matrix.ndim == 2
    return np.multiply(diag_elements, matrix, out=out)


def diagmat_dot_mat(diag_elements: np.ndarray, matrix: np.ndarray, out=None):
    """Efficient computation of diagonal matrix times matrix.

    This computes

    .. code::
        np.diag(diag_elements) @ matrix

    by using element-wise computations.

    Parameters
    ----------
    diag_elements, matrix,
        elements of computation

    out
        select where to write the result, Usual choice is setting it to matrix,
        which simply overwrites matrix and is therefore more memory efficient

    Returns
    -------
    """
    assert diag_elements.ndim == 1 and matrix.ndim == 2
    return np.multiply(matrix, diag_elements[:, np.newaxis], out=out)


def is_symmetric_matrix(
    matrix: Union[np.ndarray, scipy.sparse.csr_matrix], tol: float = 0
) -> bool:
    """Checks whether a matrix is symmetric.

    Parameters
    ----------
    matrix
        square matrix
    tol
        maximum allowed absolute deviation between corresponding elements

    Returns
    -------
    bool
        True if symmetric.
    """

    if matrix.ndim != 2:
        raise ValueError("matrix has to be 2-dim.")
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("matrix has to be square")

    max_abs_deviation = np.max(np.abs(matrix - matrix.T))
    return max_abs_deviation <= tol


def remove_numeric_noise_symmetric_matrix(
    matrix: Union[np.ndarray, scipy.sparse.spmatrix]
) -> Union[np.ndarray, scipy.sparse.spmatrix]:
    r"""Remove numerical noise from (almost) symmetric matrix.

    Noise can get sometimes be introduced in symmetric operations. The operations are
    then exectued in a different order (e.g. due to optimizations) and can then break
    exact symmetry.

    This function is intended for "almost" symmetric matrices to recover symmetry.
    Like in the following situation:

    .. code::
        np.max(np.abs(matrix - matrix.T)) # 1.1102230246251565e-16

    The symmetry is removed with:

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

    # A faster way would be to truncate the floating values of the matrix to a certain
    # precision, but NumPy does not seem to provide anything for this?

    if scipy.sparse.issparse(matrix):
        # need to preserve of explicit stored zeros (-> distance matrix)
        matrix.data[matrix.data == 0] = np.nan
        matrix = (matrix + matrix.T) / 2.0
        matrix.data[np.isnan(matrix.data)] = 0
    else:
        matrix = np.add(matrix, matrix.T, out=matrix)
        matrix = np.divide(matrix, 2.0, out=matrix)

    return matrix


def random_subsample(
    data: np.ndarray, n_samples: int, random_state: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
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

    data = check_array(data, force_all_finite=False, ensure_min_samples=2,)
    assert isinstance(data, np.ndarray)  # for mypy

    n_samples_data = data.shape[0]

    check_scalar(
        n_samples,
        name="n_samples",
        target_type=(int, np.integer),
        min_val=1,
        max_val=n_samples_data - 1,
    )

    indices = np.random.default_rng(seed=random_state).permutation(n_samples_data)[
        :n_samples
    ]
    return data[indices, :], indices
