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
    """Base class to provide functionality required in the MixIn's provided in *datafold*.

    See Also
    --------

    :class:`.TSCTransformerMixIn`

    :class:`.TSCPredictMixIn`
    """

    def _strictly_pandas_df(self, df):
        """Check if the type is strcitly a pandas.Dataframe (i.e., is False for
        TSCDataFrame).
        """
        return type(df) == pd.DataFrame

    def _has_feature_names(self, _obj):
        # True, for pandas.DataFrame or TSCDataFrame
        return isinstance(_obj, pd.DataFrame)

    def _X_to_numpy(self, X):
        """ Returns a numpy array of the data.
        """
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
        """Provides a general function to validate data -- can be overwritten if a
        concrete implementation requires different checks."""

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
            #  -- there is no assumption of the content in on the index (for
            #  TSCDataFrame it is time series ID and time)

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
    """Mixin to provide functionality for point cloud and time series transformations.

    Generally, the following input/output types are supported:

    * :class:`numpy.ndarray`
    * :class:`pandas.DataFrame` no restriction on the frame's index and column format
    * :class:`.TSCDataFrame` as a special data frame for time series collections

    Parameters
    ----------

    features_in_: Tuple[int, pandas.Index]
        Number of features during fit and corresponding feature names. The attribute
        should be set in during `fit`. Set feature names in `inverse_transform` and
        validate input in `transform`.

    features_out_: Tuple[int, pandas.Index]
        Number of features and corresponding feature names after transformation.
        The attribute should be set in during `fit`. Set feature names in
        `transform` and validate input in `inverse_transform`.
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

            if isinstance(features_out, str) and features_out == "like_features_in":
                features_out = X.columns

            if isinstance(features_out, (list, np.ndarray)):
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
            # * X has no feature names, or
            # * during fit X had no feature names given.
            # --> Only check if shape is correct and trust user with the rest
            if self.features_in_[0] != X.shape[1]:
                raise ValueError(
                    f"shape mismatch expected {self.features_out_[0]} features (cols in "
                    f"X) but got {X.shape[1]}"
                )
        else:  # self._has_feature_names(X)
            # Now X has features and during fit features were given. So now we can
            # check if feature names of X match with data during fit:

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
        self, X: TransformType, values: np.ndarray, feature_names: pd.Index
    ) -> TransformType:
        """Chooses the same type for input as type of `X`.

        Parameters
        ----------
        X
            Object from which the type will be inferred.

        values
            Data to transform in the same format as `X`.

        feature_names
            Feature names in case `X` is a :class:`pandas.DataFrame`.

        Returns
        -------

        """

        _type = type(X)

        if isinstance(X, TSCDataFrame):
            # NOTE: order is important here TSCDataFrame is also a DataFrame, so first
            # check for the special case, then for the more general case.

            return TSCDataFrame.from_same_indices_as(
                X, values=np.asarray(values), except_columns=feature_names
            )
        elif isinstance(X, pd.DataFrame):
            return pd.DataFrame(values, index=X.index, columns=feature_names)
        else:
            try:
                # last resort: try to view as numpy.array
                values = np.asarray(values)
            except Exception:
                raise TypeError(f"input type {type(X)} is not supported.")
            else:
                return values

    def fit_transform(self, X: TransformType, y=None, **fit_params) -> TransformType:
        """Fit to data, then transform it.

        Fits transformer to `X` and `y` (only if applicable) with optional parameters
        `fit_params` and returns a transformed version of `X`.

        Parameters
        ----------
        X
            Training data to transform of shape `(n_samples, n_features)`.

        y : None
            ignored

        **fit_params : dict
            Additional fit parameters.

        Returns
        -------
        numpy.ndarray, pandas.DataFrame, TSCDataFrame
            Transformed array of shape `(n_samples, n_transformed_features)` and of same
            type as input `X`.
        """
        # This is only to overwrite the datafold documentation from scikit-learns docs
        return super(TSCTransformerMixIn, self).fit_transform(X=X, y=y, **fit_params)


class TSCPredictMixIn(TSCBaseMixIn):
    """Mixin to provide functionality for models that train on time series data.

    Parameters
    ----------

    features_in_: Tuple[int, pandas.Index]
        Number of features during fit and corresponding feature names. The attribute
        should be set during `fit` and used to validate during `predict`.

    time_values_in_: Tuple[int, numpy.ndarray]
        Number of time values and array with all time values during `fit`. Note
        because in a time seres collection not all time series share the same time
        samples, a time value has to appear in at least one time series to be listed in
        `time_values_in_`. The attribute should be set during `fit`, can be used for
        default time values and allows reasonable prediction intervals to be validated.

    dt_
        Time sampling in the data during fit.
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
            pdtest.assert_index_equal(
                right=self.features_in_[1], left=X.columns, check_names=False
            )
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
