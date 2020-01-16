#!/usr/bin/env python3

from typing import Union

import numpy as np
import pandas as pd
import pandas.testing as pdtest

import datafold.pcfold.timeseries


def series_if_applicable(ds: Union[pd.Series, pd.DataFrame]):
    """Turns a DataFrame with only one column into a Series."""

    if isinstance(ds, pd.Series):
        pass  # do nothing and return same object

    elif isinstance(ds, pd.DataFrame) and not isinstance(
        ds, datafold.pcfold.timeseries.TSCDataFrame
    ):
        # Make to pd.Series of pd.DataFrame -- but not from a TSCDataFrame), as this
        # has no "TSCSeries" (yet).
        if ds.shape[1] == 1:
            # column slice is a Series in Pandas
            ds = ds.iloc[:, 0]
    else:
        raise TypeError(f"ds.type={type(ds)} not supported")

    return ds


def is_df_same_index_columns(
    df_left: pd.DataFrame, df_right: pd.DataFrame, check_index=True, check_column=True
):

    assert check_index + check_column >= 1

    is_index_same = True
    is_columns_same = True

    if check_index:
        try:
            pdtest.assert_index_equal(df_left.index, df_right.index, check_names=True)
        except AssertionError:
            is_index_same = False

    if check_column:
        try:
            pdtest.assert_index_equal(
                df_left.columns, df_right.columns, check_names=True
            )
        except AssertionError:
            is_columns_same = False

    return is_index_same and is_columns_same


def is_integer(n) -> bool:
    """Checks if `n` is an integer scalar, with the following considerations:
    * `n` is a float (built in) -> check if conversion to int is without losses
    * `n` is

    Parameters
    ----------
    n
        Data structure to check.

    Returns
    -------
    bool
        Whether it is an `n` is an integer.

    """
    return isinstance(n, (int, np.integer)) or (
        isinstance(n, (float, np.floating)) and n.is_integer()
    )


def is_float(n) -> bool:
    """Checks if `n` is a floating scalar.

    Parameters
    ----------
    n
        Data structure to check.

    Returns
    -------
    bool
        Whether it is an `n` is a float.

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
