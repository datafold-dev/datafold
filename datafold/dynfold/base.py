#!/usr/bin/env python3

from typing import Any, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import pandas.testing as pdtest
from sklearn.base import TransformerMixin
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_array, check_is_fitted

from datafold.pcfold import TSCDataFrame, TSCMetric, TSCScoring
from datafold.pcfold.timeseries.collection import TSCException
from datafold.utils.general import if1dim_rowvec

# types allowed for transformation
TransformType = Union[TSCDataFrame, np.ndarray]

# types allowed for time predictions
TimePredictType = TSCDataFrame
InitialConditionType = Union[TSCDataFrame, np.ndarray]


class TSCBaseMixin(object):
    """Base class for Mixin's in *datafold*.

    See Also
    --------
    :py:class:`.TSCTransformerMixin`
    :py:class:`.TSCPredictMixin`
    """

    def _has_feature_names(self, _obj):
        # True, for pandas.DataFrame or TSCDataFrame
        return isinstance(_obj, pd.DataFrame)

    def _read_fit_params(self, attrs: Optional[List[Tuple[str, Any]]], fit_params):
        return_values = []

        if attrs is not None:
            for attr in attrs:
                return_values.append(fit_params.pop(attr[0], attr[1]))

        if fit_params != {}:
            raise KeyError(f"fit_params.keys = {fit_params.keys()} are not supported")

        if len(return_values) == 0:
            return None
        elif len(return_values) == 1:
            return return_values[0]
        else:
            return return_values

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

    def _validate_datafold_data(
        self,
        X: Union[TSCDataFrame, np.ndarray],
        ensure_tsc: bool = False,
        validate_array_kwargs: Optional[dict] = None,
        validate_tsc_kwargs: Optional[dict] = None,
    ):
        """Provides a general function to validate data that is input to datafold
        functions -- it can be overwritten if a concrete implementation requires
        different checks.

        This function is very close to scikit-learn BaseEstimator._validate_data (which
        was introduced in  0.23.1).

        Parameters
        ----------
        X
        ensure_feature_name_type
        validate_array_kwargs
        validate_tsc_kwargs

        Returns
        -------

        """

        # defaults to empty dictionary if None
        validate_array_kwargs = validate_array_kwargs or {}
        validate_tsc_kwargs = validate_tsc_kwargs or {}

        if ensure_tsc and not isinstance(X, TSCDataFrame):
            raise TypeError(
                f"Input 'X' is of type {type(X)} but a TSCDataFrame is required."
            )

        if type(X) != TSCDataFrame:
            # Currently, everything that is not strictly a TSCDataFrame will go the
            # path of an usual array format. This includes:
            #  * sparse scipy matrices
            #  * numpy ndarray
            #  * memmap
            #  * pandas.DataFrame (Note a TSCDataFrame is also a pandas.DataFrame,
            #                      but not strictly)

            validate_tsc_kwargs = {}  # no need to check -> overwrite to empty dict

            if type(X) == pd.DataFrame:
                # special handling of pandas.DataFrame (strictly, not including
                # TSCDataFrame) --> keep the type (recover after validation).
                assert isinstance(X, pd.DataFrame)  # mypy checking
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

            validate_array_kwargs = {}  # no need to check -> overwrite to empty dict

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
                ensure_no_degenerate_ts=validate_tsc_kwargs.pop(
                    "ensure_no_degenerate_ts", False
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


class TSCTransformerMixin(TSCBaseMixin, TransformerMixin):
    """Mixin to provide functionality for point cloud and time series transformations.

    Generally, the following input/output types are supported.

    * :class:`numpy.ndarray`
    * :class:`pandas.DataFrame` no restriction on the frame's index and column format
    * :class:`.TSCDataFrame` as a special data frame for time series collections

    The parameters should be set in during `fit` in a subclass.

    .. note::

        The scikit-learn project heavily discusses on how to handle feature names. There
        are many proposed solutions. The solution that datafold uses is
        `SLEP007 <https://scikit-learn-enhancement-proposals.readthedocs.io/en/latest/slep007/proposal.html>`__

        However, this is only a proposal and may have to be changed later.

        Other resources:

        * `new array (SLEP012) <https://scikit-learn-enhancement-proposals.readthedocs.io/en/latest/slep012/proposal.html>`__
        * `discussion (SLEP008) <https://github.com/scikit-learn/enhancement_proposals/pull/18/>`__

    Parameters
    ----------

    n_features_in_: int
        Number of features of input during `fit`.

    n_features_in_: Optional[pd.Index]
        Feature names during `fit` if the input is indexed. The feature names
        are used as output in `inverse_transform` and for validation in `transform`.

    n_features_out_: int
        Number of features of output during `fit`.

    features_out_: Optional[pd.Index]
        Feature names during `fit` if the input is indexed. The feature names
        are used as output in `transform` and for validation in `inverse_transform`.
    """

    _feature_attrs = [
        "n_features_in_",
        "n_features_out_",
        "feature_names_in_",
        "feature_names_out_",
    ]

    def _setup_frame_input_fit(self, features_in: pd.Index, features_out: pd.Index):

        if features_in.has_duplicates or features_out.has_duplicates:
            raise ValueError(
                "duplicated indices detected. \n"
                f"features_in={features_in.duplicated()} \n"
                f"features_out={features_out.duplicated()}"
            )

        if features_in.ndim != 1 or features_out.ndim != 1:
            raise ValueError("feature names must be 1-dim.")

        self.n_features_in_ = len(features_in)
        self.n_features_out_ = len(features_out)
        self.feature_names_in_ = features_in
        self.feature_names_out_ = features_out

    def _setup_array_input_fit(self, features_in: int, features_out: int):
        # do not store names, because they are not available
        self.n_features_in_ = features_in
        self.n_features_out_ = features_out
        self.feature_names_in_ = None
        self.feature_names_out_ = None

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

            if isinstance(features_out, str) and features_out == "like_features_in":
                features_out = X.shape[1]
            elif isinstance(features_out, int):
                assert features_out > 0
            else:
                # if list or pd.Index use the number of features out
                features_out = len(features_out)

            self._setup_array_input_fit(
                features_in=X.shape[1], features_out=features_out
            )

    def _validate_feature_input(self, X: TransformType, direction):

        self._check_attributes_set_up(self._feature_attrs)

        if not self._has_feature_names(X) or self.feature_names_out_ is None:
            # Either
            # * X has no feature names, or
            # * during fit X had no feature names given.
            # --> Only check if shape is correct and trust user with the rest
            if self.n_features_in_ != X.shape[1]:
                raise ValueError(
                    f"Shape mismatch: expected {self.n_features_out_} "
                    f"features (number of columns in 'X') but got {X.shape[1]}."
                )
        else:  # self._has_feature_names(X)
            # Now X has features and during fit features were given. So now we can
            # check if feature names of X match with data during fit:

            if direction == "transform":
                _check_features = self.feature_names_in_
            elif direction == "inverse_transform":
                _check_features = self.feature_names_out_
            else:
                raise RuntimeError(
                    f"'direction'={direction} not known. Please report bug."
                )

            if isinstance(X, pd.Series):
                # if X is a Series, then the columns of the original data are in a
                # Series this usually happens if X.iloc[0, :] --> returns a Series
                pdtest.assert_index_equal(right=_check_features, left=X.index)
            else:
                pdtest.assert_index_equal(right=_check_features, left=X.columns)

    def _same_type_X(
        self, X: TransformType, values: np.ndarray, feature_names: pd.Index
    ) -> Union[pd.DataFrame, TransformType]:
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

        **fit_params: Dict[str, object]
            Additional fit parameters.

        Returns
        -------
        numpy.ndarray, pandas.DataFrame, TSCDataFrame
            Transformed array of shape `(n_samples, n_transformed_features)` and of same
            type as input `X`.
        """
        # This is only to overwrite the datafold documentation from scikit-learns docs
        return super(TSCTransformerMixin, self).fit_transform(X=X, y=y, **fit_params)


class TSCPredictMixin(TSCBaseMixin):
    """Mixin to provide functionality for models that train on time series data.

    The attribute should be set during `fit` and used to validate during `predict`.

    Parameters
    ----------

    n_features_in_: int
        Number of features during `fit`.

    feature_names_in_: pd.Index
        The feature names during `fit`.

    time_values_in_: numpy.ndarray
        Time values with all time values observed during `fit`. Note, that because in a
        time series collection not all time series must share the same time
        values, a time value to be recorded in `time_values_in_` must at least appear
        in one time series.

    dt_: Union[float, pd.Series]
        Time sampling rate in the time series data during `fit`.
    """

    _cls_feature_attrs = [
        "n_features_in_",
        "feature_names_in_",
        "time_values_in_",
        "dt_",
    ]

    @property
    def time_interval_(self):
        self._check_attributes_set_up(check_attributes="time_values_in_")
        return (self.time_values_in_[0], self.time_values_in_[-1])

    def _setup_default_tsc_metric_and_score(self):
        self.metric_eval = TSCMetric(metric="rmse", mode="feature", scaling="min-max")
        self._score_eval = TSCScoring(self.metric_eval)

    def _setup_features_and_time_fit(self, X: TSCDataFrame):

        if not isinstance(X, TSCDataFrame):
            raise TypeError("Only TSCDataFrame can be used for 'X'.")

        time_values = X.time_values()
        features_in = X.columns

        time_values = self._validate_time_values(time_values=time_values)
        self.time_values_in_ = time_values
        self.n_features_in_ = len(features_in)
        self.feature_names_in_ = features_in
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

        if not (np.diff(time_values).astype(np.float64) > 0).all():
            # as "float64" is required in case of datetime where the differences are in
            # terms of "np.timedelta"
            raise ValueError(
                "Parameter 'time_values' must be sorted with increasing unique values."
            )

        return time_values

    def _validate_delta_time(self, delta_time):
        self._check_attributes_set_up(check_attributes=["dt_"])

        if isinstance(delta_time, pd.Series):
            raise NotImplementedError(
                "Currently, all methods assume that dt_ is const."
            )

        if delta_time != self.dt_:
            raise TSCException(
                f"delta_time during fit was {self.dt_}, now it is {delta_time}"
            )

    def _validate_feature_names(self, X: TransformType, require_all=True):
        self._check_attributes_set_up(check_attributes=["feature_names_in_"])

        try:
            if require_all:
                pdtest.assert_index_equal(
                    right=self.feature_names_in_, left=X.columns, check_names=False
                )
            else:
                if not np.isin(X.columns, self.feature_names_in_).all():
                    raise AssertionError(
                        f"feature names in X are invalid "
                        f"{X.columns[np.isin(self.feature_names_in_,X.columns)]}"
                    )
        except AssertionError as e:
            raise ValueError(e.args[0])

    def _validate_features_and_time_values(
        self, X: TSCDataFrame, time_values: Optional[np.ndarray]
    ):

        self._check_attributes_set_up(check_attributes=["time_values_in_"])

        if time_values is None:
            time_values = self.time_values_in_

        if not self._has_feature_names(X):
            raise TypeError("only types that support feature names are supported")

        self._validate_time_values(time_values=time_values)
        self._validate_feature_names(X)

        return X, time_values

    def fit(self, X: TimePredictType, **fit_params):
        raise NotImplementedError("method not implemented")

    def reconstruct(
        self,
        X: TSCDataFrame,
        qois: Optional[Union[np.ndarray, pd.Index, List[str]]] = None,
    ):
        raise NotImplementedError("method not implemented")

    def predict(
        self,
        X: InitialConditionType,
        time_values: Optional[np.ndarray] = None,
        **predict_params,
    ):
        raise NotImplementedError("method not implemented")
