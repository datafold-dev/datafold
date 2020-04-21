#!/usr/bin/env python3

import sys
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import pandas.testing as pdtest
import scipy
import scipy.sparse
import scipy.sparse.linalg
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_array, check_is_fitted

from datafold.pcfold import TSCDataFrame, TSCMetric, TSCScoring
from datafold.pcfold.eigsolver import NumericalMathError, compute_kernel_eigenpairs
from datafold.pcfold.kernels import DmapKernelFixed
from datafold.utils.general import if1dim_rowvec

DataFrameType = Union[TSCDataFrame, pd.DataFrame]

# types allowed for transformation
TransformType = Union[DataFrameType, np.ndarray]

# types allowed for time predictions
TimePredictType = TSCDataFrame
InitialConditionType = Union[TSCDataFrame, pd.DataFrame, np.ndarray]


class TSCBaseMixIn:
    def _strictly_pandas_df(self, df):
        # This returns False for subclasses (TSCDataFrame)
        return type(df) == pd.DataFrame

    def _has_feature_names(self, _obj):
        return isinstance(_obj, pd.DataFrame)

    def _X_to_numpy(self, X):
        if self._has_feature_names(X):
            X = X.to_numpy()
            # a row in a df is always a single sample (which requires to be
            # represented in a 2D matrix)
            return if1dim_rowvec(X)
        else:
            return X

    def _check_attributes_set_up(self, check_attributes):
        try:
            check_is_fitted(
                self, attributes=check_attributes,
            )
        except NotFittedError:
            raise RuntimeError(
                f"{check_attributes} are not available for estimator {self}. "
                f"Please report bug."
            )

    def _validate_data(
        self,
        X,
        ensure_feature_name_type=False,
        validate_array_kwargs=None,
        validate_tsc_kwargs=None,
    ):
        """Provides a general function to check data -- can be overwritten if an
        implementation requires different checks."""

        if validate_array_kwargs is None:
            validate_array_kwargs = {}

        if validate_tsc_kwargs is None:
            validate_tsc_kwargs = {}

        if ensure_feature_name_type and not self._has_feature_names(X):
            raise TypeError(
                f"X is of type {type(X)} but frame types ("
                f"pd.DataFrame of TSCDataFrame) are required."
            )

        if not isinstance(X, TSCDataFrame):
            # Currently, a pd.DataFrame is treated like numpy data
            #  -- there is no time required such as in TSCDataFrame

            validate_tsc_kwargs = {}  # no need to check

            if self._strictly_pandas_df(X):
                assert isinstance(X, pd.DataFrame)  # for mypy checking
                revert_to_data_frame = True
                idx, col = X.index, X.columns
            else:
                revert_to_data_frame = False
                idx, col = [None] * 2

            X = check_array(
                X,
                accept_sparse=validate_array_kwargs.pop("accept_sparse", False),
                accept_large_sparse=validate_array_kwargs.pop(
                    "accept_large_sparse", False
                ),
                dtype=validate_array_kwargs.pop("dtype", "numeric"),
                order=validate_array_kwargs.pop("order", None),
                copy=validate_array_kwargs.pop("copy", False),
                force_all_finite=validate_array_kwargs.pop("force_all_finite", True),
                ensure_2d=validate_array_kwargs.pop("ensure_2d", True),
                allow_nd=validate_array_kwargs.pop("allow_nd", False),
                ensure_min_samples=validate_array_kwargs.pop("ensure_min_samples", 1),
                ensure_min_features=validate_array_kwargs.pop("ensure_min_features", 1),
                estimator=self,
            )

            if revert_to_data_frame:
                X = pd.DataFrame(X, index=idx, columns=col)

        else:

            validate_array_kwargs = {}  # no need to check

            X = X.tsc.check_tsc(
                ensure_all_finite=validate_tsc_kwargs.pop("ensure_all_finite", True),
                ensure_same_length=validate_tsc_kwargs.pop("ensure_same_length", False),
                ensure_const_delta_time=validate_tsc_kwargs.pop(
                    "ensure_const_delta_time", False
                ),
                ensure_delta_time=validate_tsc_kwargs.pop("ensure_delta_time", None),
                ensure_same_time_values=validate_tsc_kwargs.pop(
                    "ensure_same_time_values", False
                ),
                ensure_normalized_time=validate_tsc_kwargs.pop(
                    "ensure_normalized_time", False
                ),
                ensure_n_timeseries=validate_tsc_kwargs.pop(
                    "ensure_n_timeseries", None
                ),
                ensure_min_timesteps=validate_tsc_kwargs.pop(
                    "ensure_min_timesteps", None
                ),
            )

        if validate_array_kwargs != {} or validate_tsc_kwargs != {}:
            # validate_kwargs have to be empty and must only contain key-values that can
            # be handled to check_array / check_tsc

            left_over_keys = list(validate_array_kwargs.keys()) + list(
                validate_tsc_kwargs.keys()
            )
            raise ValueError(
                f"{left_over_keys} are no valid validation keys. Please report bug."
            )

        return X


class TSCTransformerMixIn(TSCBaseMixIn, TransformerMixin):
    """Provides infrastructure for (optional) time series transformations.

    Allows to transform data with numpy.ndarray or pandas.DataFrame (including
    TSCDataFrame). Usually, the input type is the output type.

    Parameters
    ----------

    features_in_: Tuple[int, pandas.Index]
        Number of features during fit and corresponding feature names (intended to
        be stored during `fit`). Suitable to validate intput data.

    features_out_: Tuple[int, pandas.Index]
        Number of features after trasnform and corresponding feature names (intended to
        be stored during `fit`). Suitable to set feature names for transform and
        validate input for `inverse_transform`.

    """

    _feature_attrs = ["features_in_", "features_out_"]

    def _setup_frame_input_fit(self, features_in: pd.Index, features_out: pd.Index):

        if features_in.has_duplicates or features_out.has_duplicates:
            raise ValueError(
                "duplicated indices detected. \n"
                f"features_in={features_in.duplicated()} \n"
                f"features_out={features_out.duplicated()}"
            )

        if features_in.ndim != 1 or features_out.ndim != 1:
            raise ValueError("feature names must be 1-dim.")

        self.features_in_ = (len(features_in), features_in)
        self.features_out_ = (len(features_out), features_out)

    def _setup_array_input_fit(self, features_in: int, features_out: int):
        self.features_in_ = (features_in, None)
        self.features_out_ = (features_out, None)

    def _setup_features_fit(self, X, features_out):

        if isinstance(features_out, str):
            assert features_out == "like_features_in"

        if self._has_feature_names(X):

            if features_out == "like_features_in":
                features_out = X.columns

            if isinstance(features_out, list):
                # For convenience features_out can be given as a list
                # (better code readability than pd.Index)
                features_out = pd.Index(
                    features_out, dtype=np.str, name=TSCDataFrame.tsc_feature_col_name,
                )

            self._setup_frame_input_fit(
                features_in=X.columns, features_out=features_out
            )
        else:
            if features_out == "like_features_in":
                features_out = X.shape[1]
            else:
                # if list or pd.Index use the number of features out
                features_out = len(features_out)

            self._setup_array_input_fit(
                features_in=X.shape[1], features_out=features_out
            )

    def _validate_feature_input(self, X: TransformType, direction):

        self._check_attributes_set_up(self._feature_attrs)

        if not self._has_feature_names(X) or self.features_out_[1] is None:
            # Either
            # * now X has no feature names, or
            # * during fit X had no feature names given.
            # --> Only check if shape is correct and trust user with the rest
            if self.features_in_[0] != X.shape[1]:
                raise ValueError(
                    f"shape mismatch expected {self.features_out_[0]} features (cols in "
                    f"X) but got {X.shape[1]}"
                )
        else:  # self._has_feature_names(X)
            # Now X has features and during fit features were given. So now we can
            # check if both feature names match:

            _check_features: Tuple[int, pd.Index]

            if direction == "transform":
                _check_features = self.features_in_
            elif direction == "inverse_transform":
                _check_features = self.features_out_
            else:
                raise RuntimeError(
                    f"'direction'={direction} not known. Please report bug."
                )

            if isinstance(X, pd.Series):
                # if X is a Series, then the columns of the original data are in a
                # Series this usually happens if X.iloc[0, :] --> returns a Series
                pdtest.assert_index_equal(right=_check_features[1], left=X.index)
            else:
                pdtest.assert_index_equal(right=_check_features[1], left=X.columns)

    def _same_type_X(
        self, X: TransformType, values: np.ndarray, set_columns: pd.Index
    ) -> TransformType:

        _type = type(X)

        if isinstance(X, TSCDataFrame):
            # NOTE: order is important here TSCDataFrame is also a DataFrame, so first
            # check for the special case, then for the more general case.

            return TSCDataFrame.from_same_indices_as(
                X, values=np.asarray(values), except_columns=set_columns
            )
        elif isinstance(X, pd.DataFrame):
            return pd.DataFrame(values, index=X.index, columns=set_columns)
        else:
            try:
                # try to view as numpy.array
                values = np.asarray(values)
            except Exception:
                raise TypeError(f"input type {type(X)} is not supported.")
            else:
                return values


class TSCPredictMixIn(TSCBaseMixIn):
    """Methods and functionality to provide time series prediction.

    Parameters
    ----------

    features_in_: Tuple[int, pandas.Index]
        Number of features during fit and corresponding feature names (intended to
        be stored during `fit`). Suitable to test or create initial conditions.

    time_values_in_: Tuple[int, numpy.ndarray]
        Number of time values and array with all actual time values in at least on time
        series (intended to be stored during `fit`). This allows to check for
        time extrapolation.

    dt_
        time delta during fit

    """

    _cls_feature_attrs = ["features_in_", "time_values_in_", "dt_"]

    @property
    def time_interval_(self):
        self._check_attributes_set_up(check_attributes="time_values_in_")
        return (self.time_values_in_[1][0], self.time_values_in_[1][-1])

    def _setup_default_tsc_metric_and_score(self):
        self.metric_eval = TSCMetric(metric="rmse", mode="feature", scaling="min-max")
        self._score_eval = TSCScoring(self.metric_eval)

    def _setup_features_and_time_fit(self, X: TSCDataFrame):

        if not isinstance(X, TSCDataFrame):
            raise TypeError("Only TSCDataFrame can be used for 'X'. ")

        time_values = X.time_values()
        features_in = X.columns

        time_values = self._validate_time_values(time_values=time_values)
        self.time_values_in_ = (len(time_values), time_values)

        self.dt_ = X.delta_time
        if isinstance(self.dt_, pd.Series) or np.isnan(
            self.dt_
        ):  # Series if dt_ is not the same across multiple time series.
            raise NotImplementedError(
                "Currently, all algorithms assume a constant time "
                f"delta. Got X.time_delta={X.time_delta}"
            )

        # TODO: check this closer why are there 5 decimals required?
        assert (
            np.around(
                (self.time_interval_[1] - self.time_interval_[0]) / self.dt_, decimals=5
            )
            % 1
            == 0
        )

        self.features_in_ = (len(features_in), features_in)

    def _validate_time_values(self, time_values: np.ndarray):

        try:
            time_values = np.asarray(time_values)
        except Exception:
            raise TypeError("Cannot convert 'time_values' to array.")

        if not isinstance(time_values, np.ndarray):
            raise TypeError("time_values has to be a NumPy array")

        if time_values.dtype.kind not in "iufM":
            # see for dtype.kind values:
            # https://docs.scipy.org/doc/numpy/reference/generated/numpy.dtype.kind.html
            raise TypeError(f"time_values.dtype {time_values.dtype} not supported")

        _is_datetime = time_values.dtype.kind == "M"

        if not _is_datetime and (time_values < 0).any():
            # "datetime" cannot be negative and cannot be checked with "< 0"
            raise ValueError("in time_values no negative values are allowed")

        if time_values.ndim != 1:
            raise ValueError("time_values must be be one dimensional")

        if not (np.diff(time_values).astype(np.float64) >= 0).all():
            # as "float64" is required in case of datetime where the differences are in
            # terms of "np.timedelta"
            raise ValueError("time_values must be sorted")

        return time_values

    def _validate_delta_time(self, delta_time):
        self._check_attributes_set_up(check_attributes=["dt_"])

        if isinstance(delta_time, pd.Series):
            raise NotImplementedError(
                "Currently, all methods assume that dt_ is const."
            )

        if delta_time != self.dt_:
            raise ValueError(
                f"delta_time during fit was {self.dt_}, now it is {delta_time}"
            )

    def _validate_feature_names(self, X: TransformType):
        self._check_attributes_set_up(check_attributes=["features_in_"])

        try:
            pdtest.assert_index_equal(right=self.features_in_[1], left=X.columns)
        except AssertionError as e:
            raise ValueError(e.args[0])

    def _validate_features_and_time_values(
        self, X: DataFrameType, time_values: Optional[np.ndarray] = None
    ):

        self._check_attributes_set_up(check_attributes=["time_values_in_"])

        if time_values is None:
            time_values = self.time_values_in_[1]

        if not self._has_feature_names(X):
            raise TypeError("only types that support feature names are supported")

        self._validate_time_values(time_values=time_values)

        if isinstance(X, TSCDataFrame):
            # sometimes also for initial conditions a TSCDataFrame is required (e.g.
            # for transformation with Takens) -- for this case check also that the
            # delta_time matches.
            self._validate_delta_time(delta_time=X.delta_time)

        self._validate_feature_names(X)

        return X, time_values

    def fit(self, X: TimePredictType, **fit_params):
        raise NotImplementedError("method not implemented")

    def reconstruct(self, X: TSCDataFrame):
        raise NotImplementedError("method not implemented")

    def predict(self, X: InitialConditionType, time_values=None, **predict_params):
        raise NotImplementedError("method not implemented")


class DmapKernelMethod(BaseEstimator):
    def __init__(
        self,
        epsilon: float,
        n_eigenpairs: int,
        cut_off,
        is_stochastic: bool,
        alpha: float,
        symmetrize_kernel,
        dist_backend,
        dist_backend_kwargs,
    ):
        self.epsilon = epsilon
        self.n_eigenpairs = n_eigenpairs
        self.cut_off = cut_off
        self.is_stochastic = is_stochastic
        self.alpha = alpha
        self.symmetrize_kernel = symmetrize_kernel
        self.dist_backend = dist_backend
        self.dist_backend_kwargs = dist_backend_kwargs

    @property
    def kernel_(self) -> DmapKernelFixed:
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

    def _solve_eigenproblem(
        self, kernel_matrix, basis_change_matrix
    ) -> Tuple[np.ndarray, np.ndarray]:

        try:
            eigvals, eigvect = compute_kernel_eigenpairs(
                kernel_matrix=kernel_matrix,
                n_eigenpairs=self.n_eigenpairs,
                is_symmetric=self.kernel_.is_symmetric,
                is_stochastic=self.is_stochastic,
                backend="scipy",
            )
        except NumericalMathError:
            # re-raise with more details for the DMAP
            raise NumericalMathError(
                "Eigenvalues have non-negligible imaginary part (larger than "
                f"{1e2 * sys.float_info.epsilon}. First try to use "
                f"parameter 'symmetrize_kernel=True' (improves numerical stability) and "
                f"only if this is not working adjust epsilon."
            )

        if basis_change_matrix is not None:
            eigvect = basis_change_matrix @ eigvect

        eigvect /= np.linalg.norm(eigvect, axis=0)[np.newaxis, :]

        return np.real(eigvals), np.real(eigvect)
