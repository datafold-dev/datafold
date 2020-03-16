#!/usr/bin/env python3

from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import pandas.testing as pdtest
import scipy.sparse
from sklearn.base import TransformerMixin
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_array, check_is_fitted

from datafold.pcfold.timeseries import TSCDataFrame
from datafold.pcfold.timeseries.metric import TSCMetric, make_tsc_scorer
from datafold.utils.datastructure import if1dim_rowvec

FEATURE_NAME_TYPES = Union[TSCDataFrame, pd.DataFrame]

# types allowed for transformation
TRANF_TYPES = Union[FEATURE_NAME_TYPES, np.ndarray]

# types allowed for predict
PRE_FIT_TYPES = TSCDataFrame
PRE_IC_TYPES = Union[TSCDataFrame, pd.DataFrame, np.ndarray]


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
            # Currently, a pd.DataFrame is treated like data (there is not even a time
            # involvement required).

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
                force_all_finite=validate_tsc_kwargs.pop("force_all_finite", True),
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

    _FEAT_ATTR = ["features_in_", "features_out_"]

    def _setup_features_input_fit(
        self, features_in: pd.Index, features_out: Union[List[str], pd.Index]
    ):
        # For convenience features_out can be set as a list (better readability in the
        # implementations)
        if isinstance(features_out, list):
            features_out = pd.Index(
                features_out, dtype=np.str, name=TSCDataFrame.IDX_QOI_NAME,
            )

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

    def _validate_features_transform(self, X: TRANF_TYPES):
        self._check_attributes_set_up(self._FEAT_ATTR)

        if self._has_feature_names(X):
            if self.features_in_[1] is None:
                raise ValueError(
                    "fit method was called with np.ndarray, there is no "
                    "support for pd.DataFrame for transform or "
                    "inverse_transform"
                )
            self._validate_feature_names(X=X, direction="transform")
        else:
            if self.features_in_[0] != X.shape[1]:
                raise ValueError(
                    f"shape mismatch expected {self.features_in_[0]} "
                    f"got {X.shape[1]}"
                )

    def _validate_features_inverse_transform(self, X: TRANF_TYPES):
        self._check_attributes_set_up(self._FEAT_ATTR)

        if self._has_feature_names(X):
            if self.features_in_[1] is None:
                raise ValueError(
                    "fit method was called with np.ndarray, there is no "
                    "support for pd.DataFrame for transform or "
                    "inverse_transform"
                )
            self._validate_feature_names(X, direction="inverse")

    def _validate_feature_names(self, X: TRANF_TYPES, direction):

        _check_features: Tuple[int, pd.Index]

        if direction == "transform":
            _check_features = self.features_in_
        elif direction == "inverse":
            _check_features = self.features_out_
        else:
            raise RuntimeError("Please report bug.")

        if self._has_feature_names(X):
            if isinstance(X, pd.Series):
                # if X is a Series, then the columns of the original data are in a Series
                # this usually happens if X.iloc[0, :] --> returns a Series
                pdtest.assert_index_equal(right=_check_features[1], left=X.index)
            else:
                pdtest.assert_index_equal(right=_check_features[1], left=X.columns)

    def _same_type_X(
        self, X: TRANF_TYPES, values: np.ndarray, set_columns
    ) -> TRANF_TYPES:

        _type = type(X)

        if isinstance(X, TSCDataFrame):
            # NOTE: order is important here TSCDataFrame is also a DataFrame, so first
            # check for the special case, then for the more general case.

            return TSCDataFrame.from_same_indices_as(
                X, values=values, except_columns=set_columns
            )
        elif isinstance(X, pd.DataFrame):
            return pd.DataFrame(values, index=X.index, columns=set_columns)
        elif isinstance(X, np.ndarray):
            return values
        else:
            raise TypeError(f"input type {type(X)} is not supported")


class TSCPredictMixIn(TSCBaseMixIn):

    _FEAT_ATTR = ["features_in_", "time_values_in_", "dt_"]

    @property
    def time_interval_(self):
        self._check_attributes_set_up(check_attributes="time_values_in_")
        return (self.time_values_in_[1][0], self.time_values_in_[1][-1])

    def _setup_default_tsc_scorer_and_metric(self):
        self._metric_eval = TSCMetric.make_tsc_metric(
            metric="rmse", mode="qoi", scaling="min-max"
        )
        self._score_eval = make_tsc_scorer(self._metric_eval)

    def _setup_features_and_time_fit(self, X: TSCDataFrame):
        time_values = X.time_values(unique_values=True)
        features_in = X.columns

        self.time_values_in_ = (len(time_values), time_values)
        self._validate_time_values(self.time_values_in_[1])

        self.dt_ = X.delta_time
        if isinstance(self.dt_, pd.Series):
            raise NotImplementedError(
                "Currently, all methods assume a constant time "
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

        self._check_attributes_set_up(check_attributes=["time_values_in_"])

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

    def _validate_delta_time(self, delta_time):
        self._check_attributes_set_up(check_attributes=["dt_"])

        if isinstance(delta_time, pd.Series):
            raise NotImplementedError(
                "Currently, all methods assume that dt_ is const."
            )

        if delta_time != self.dt_:
            raise ValueError(
                f"delta_time during fit was {self.dt_}, " f"now it is {delta_time}"
            )

    def _validate_feature_names(self, X: TRANF_TYPES):
        self._check_attributes_set_up(check_attributes=["features_in_"])

        if isinstance(X, pd.Series):
            # if X is a Series, then the columns of the original data are in a Series
            # this usually happens if X.iloc[0, :] --> returns a Series
            pdtest.assert_index_equal(right=self.features_in_[1], left=X.index)
        else:
            pdtest.assert_index_equal(right=self.features_in_[1], left=X.columns)

    def _validate_features_and_time_values(
        self, X: FEATURE_NAME_TYPES, time_values: np.ndarray
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

    def fit(self, X: PRE_FIT_TYPES, **fit_params):
        raise NotImplementedError("base class")

    def reconstruct(self, X: TSCDataFrame):
        raise NotImplementedError("base class")

    def fit_reconstruct(self, X: TSCDataFrame, **fit_params):
        raise NotImplementedError("base class")

    def predict(self, X: PRE_IC_TYPES, time_values=None, **predict_params):
        # NOTE the definition of predict cannot be a TSC. Best is provided as a
        # pd.DataFrame with all the information...
        raise NotImplementedError("base class")
