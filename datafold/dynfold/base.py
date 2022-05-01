#!/usr/bin/env python3

from typing import Any, Callable, List, Optional, Tuple, Union

import numpy as np
import numpy.testing as nptest
import pandas as pd
from pandas.api.types import is_datetime64_dtype
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_array, check_is_fitted

from datafold.pcfold import InitialCondition, TSCDataFrame, TSCMetric, TSCScoring
from datafold.pcfold.timeseries.collection import TSCException
from datafold.utils.general import if1dim_rowvec

# types allowed for transformation
TransformType = Union[TSCDataFrame, np.ndarray]

# types allowed for time predictions
TimePredictType = TSCDataFrame
InitialConditionType = Union[TSCDataFrame, np.ndarray]


class TSCBase(object):
    """Base class for Mixin's in *datafold*.

    See Also
    --------
    :py:class:`.TSCTransformerMixin`
    :py:class:`.TSCPredictMixin`
    """

    def get_feature_names_out(self, input_features=None):
        raise NotImplementedError(
            "class does not provide 'get_feature_names_out' method"
        )

    def _has_feature_names(self, _obj):
        # True, for pandas.DataFrame or TSCDataFrame
        return isinstance(_obj, pd.DataFrame)

    def _read_fit_params(self, attrs: Optional[List[Tuple[str, Any]]], fit_params):
        return_values = []

        if attrs is not None:
            for a in attrs:
                return_values.append(fit_params.pop(a[0], a[1]))

        if fit_params != {}:
            raise KeyError(f"fit_params.keys = {fit_params.keys()} are not supported")

        if len(return_values) == 0:
            return None
        elif len(return_values) == 1:
            return return_values[0]
        else:
            return return_values

    def _X_to_numpy(self, X):
        """Returns a numpy array of the data."""
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
                self,
                attributes=check_attributes,
            )
        except NotFittedError:
            raise RuntimeError(
                f"{check_attributes} are not available for estimator {self}. "
                f"Please report bug."
            )

    def _validate_datafold_data(
        self,
        X: Union[TSCDataFrame, np.ndarray],
        *,
        ensure_np: bool = False,
        ensure_tsc: bool = False,
        force_all_finite: bool = True,
        ensure_min_samples: int = 1,
        ensure_min_features: int = 1,
        array_kwargs: Optional[dict] = None,
        tsc_kwargs: Optional[dict] = None,
    ):
        """Provides a general function to validate data that is input to datafold
        functions -- it can be overwritten if a concrete implementation requires
        different checks.

        This function is very close to scikit-learn BaseEstimator._validate_data (which
        was introduced in  0.23.1).

        Parameters
        ----------
        X
        ensure_np
        ensure_tsc
        array_kwargs
        tsc_kwargs

        Returns
        -------

        """

        # defaults to empty dictionary if None
        array_kwargs = array_kwargs or {}
        tsc_kwargs = tsc_kwargs or {}

        if ensure_np + ensure_tsc == 2:
            raise ValueError("only 'ensure_np' or 'ensure_tsc' can be True")

        if self._has_feature_names(X):
            if X.columns.ndim != 1:
                raise ValueError("columns (features) must be 1-dim.")

        if type(X) != TSCDataFrame:
            # Currently, everything that is not strictly a TSCDataFrame will go the
            # path of a usual array format. This includes:
            #  * sparse scipy matrices
            #  * numpy ndarray
            #  * memmap
            #  * pd.DataFrame (Note that TSCDataFrame is also a pd.DataFrame,
            #                  but not in a strict sense)

            if ensure_tsc:
                raise TypeError(
                    f"Input 'X' is of type {type(X)} but type TSCDataFrame is required."
                )

            tsc_kwargs = {}  # no need to check -> overwrite to empty dict

            if type(X) == pd.DataFrame:

                if ensure_np:
                    TypeError(
                        f"Input 'X' is of type {type(X)} but a numpy format is required."
                    )

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
                accept_sparse=array_kwargs.pop("accept_sparse", False),
                accept_large_sparse=array_kwargs.pop("accept_large_sparse", False),
                dtype=array_kwargs.pop("dtype", "numeric"),
                order=array_kwargs.pop("order", None),
                copy=array_kwargs.pop("copy", False),
                force_all_finite=force_all_finite,
                ensure_2d=array_kwargs.pop("ensure_2d", True),
                allow_nd=array_kwargs.pop("allow_nd", False),
                ensure_min_samples=ensure_min_samples,
                ensure_min_features=ensure_min_features,
                estimator=self,
            )

            if revert_to_data_frame:
                X = pd.DataFrame(X, index=idx, columns=col)

        else:  # isinstance(X, TSCDataFrame)

            if ensure_np:
                raise TypeError(
                    f"Input 'X' is of type {type(X)} but a numpy format is required."
                )

            array_kwargs = {}  # no need to check -> overwrite to empty dict

            X = X.tsc.check_tsc(
                ensure_all_finite=force_all_finite,
                ensure_min_samples=ensure_min_samples,
                ensure_same_length=tsc_kwargs.pop("ensure_same_length", False),
                ensure_const_delta_time=tsc_kwargs.pop(
                    "ensure_const_delta_time", False
                ),
                ensure_delta_time=tsc_kwargs.pop("ensure_delta_time", None),
                ensure_same_time_values=tsc_kwargs.pop(
                    "ensure_same_time_values", False
                ),
                ensure_normalized_time=tsc_kwargs.pop("ensure_normalized_time", False),
                ensure_n_timeseries=tsc_kwargs.pop("ensure_n_timeseries", None),
                ensure_min_timesteps=tsc_kwargs.pop("ensure_min_timesteps", None),
                ensure_no_degenerate_ts=tsc_kwargs.pop(
                    "ensure_no_degenerate_ts", False
                ),
            )

        if array_kwargs != {} or tsc_kwargs != {}:
            # validate_kwargs have to be empty and must only contain key-values that can
            # be handled to check_array / check_tsc

            left_over_keys = list(array_kwargs.keys()) + list(tsc_kwargs.keys())
            raise ValueError(
                f"{left_over_keys} are no valid validation keys. Please report bug."
            )

        return X


class TSCTransformerMixin(TSCBase, TransformerMixin):
    """Mixin to provide functionality for point cloud and time series transformations.

    Generally, the following input/output types are supported.

    * :class:`numpy.ndarray`
    * :class:`pandas.DataFrame` no restriction on the frame's index and column format
    * :class:`.TSCDataFrame` as a special data frame for time series collections

    The parameters should be set in during `fit` in a subclass.

    Discussions in the scikit-learn project, which are followed in datafold:

    * `SLEP 007 <https://scikit-learn-enhancement-proposals.readthedocs.io/en/latest/slep007/proposal.html>`__
    * `SLEP 010 <https://scikit-learn-enhancement-proposals.readthedocs.io/en/latest/slep010/proposal.html>`__

    Other related discussions (also proposing different solutions):

    * `new array (SLEP012) <https://scikit-learn-enhancement-proposals.readthedocs.io/en/latest/slep012/proposal.html>`__
    * `discussion (SLEP008) <https://github.com/scikit-learn/enhancement_proposals/pull/18/>`__

    Parameters
    ----------

    n_features_in_: int
        Number of features passed in input `X` in `fit`. The same number of features are
        required for `transform`.

    feature_names_in_: Optional[np.array]
        Feature names passed in input `X` in `fit`. The attribute is only set if the input is
        a pandas object. The feature names are used for validation in `transform` and as
        output feature names in `inverse_transform`.

    n_features_out_: int
        Number of features in output of `transform`.
    """  # noqa: E501

    _feature_attrs = ["n_features_in_", "n_features_out_"]

    def _setup_feature_attrs_fit(
        self: Union[BaseEstimator, "TSCTransformerMixin"],
        X,
        n_features_out: Optional[int] = None,
    ):
        if not hasattr(self, "n_features_in_"):
            # sklearn function to set n_features_in_
            self._check_n_features(X, reset=True)

        if not hasattr(self, "feature_names_in_"):

            if isinstance(X, TSCDataFrame) and type(X.columns[0]) != str:
                # workaround for datafold to support non-str feature names
                # sklearn does only support feature names of type str
                # Note that there is a guarantee for TSCDataFrame that the feature names have
                # same type
                self.feature_names_in_ = X.columns.to_numpy()
            else:
                # sklearn function to set feature_names_in_
                self._check_feature_names(X, reset=True)

        # set features out (currently not supported by sklearn, but there is a proposal:
        # https://scikit-learn-enhancement-proposals.readthedocs.io/en/latest/slep013/proposal.html  # noqa
        if not hasattr(self, "n_features_out_"):
            if n_features_out is None:
                feature_out = self.get_feature_names_out()
                self.n_features_out_: int = len(feature_out)
            else:
                self.n_features_out_ = n_features_out

    def _validate_feature_input(
        self: Union[BaseEstimator, "TSCTransformerMixin"], X: TransformType, direction
    ):

        self._check_attributes_set_up(self._feature_attrs)

        if direction == "transform":
            self._check_n_features(X, reset=False)
            self._check_feature_names(X, reset=False)
        else:  # direction == inverse_transform
            should_n_features = self.n_features_out_

            if should_n_features != X.shape[1]:
                raise ValueError(
                    f"The number of features (={X.shape[1]}) do not match. "
                    f"Required: {should_n_features}"
                )

            if self._has_feature_names(X):
                should_features = self.get_feature_names_out()
                try:
                    nptest.assert_array_equal(should_features, X.columns.to_numpy())
                except AssertionError:
                    raise ValueError(
                        f"The features names do not match. "
                        f"Required: {should_features}."
                    )

    def _same_type_X(
        self,
        X: TransformType,
        values: np.ndarray,
        feature_names: Union[pd.Index, np.ndarray],
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


class TSCPredictMixin(TSCBase):
    """Mixin to provide functionality for models that train on time series data.

    The attribute should be set during `fit` and used to validate during `predict`.

    Parameters
    ----------

    n_features_in_: int
        Number of features during `fit`.

    feature_names_in_: np.ndarray
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

    def _setup_features_and_time_attrs_fit(
        self: Union[BaseEstimator, "TSCPredictMixin"], X: TSCDataFrame
    ):

        if not isinstance(X, TSCDataFrame):
            raise TypeError("Only TSCDataFrame can be used for 'X'.")

        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = X.columns

        time_values = X.time_values()
        time_values = self._validate_time_values(time_values=time_values)
        self.time_values_in_ = time_values
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

        if not is_datetime64_dtype(time_values) and (time_values < 0).any():
            # "datetime" cannot be negative and raises error when checked with "< 0"
            raise ValueError("In 'time_values' all values must be non-negative.")

        if not np.isfinite(time_values).all():
            raise ValueError("'time_values' contains invalid numbers (inf/nan).")

        if time_values.ndim != 1:
            raise ValueError("'time_values' must be be an 1-dim. array")

        if not (np.diff(time_values).astype(float) > 0).all():
            # as "float64" is required in case of datetime where the differences are in
            # terms of "np.timedelta"
            raise ValueError(
                "'time_values' must be sorted with increasing unique values"
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

    def _validate_feature_names(
        self: Union[BaseEstimator, "TSCPredictMixin"], X: TransformType
    ):

        self._check_n_features(X, reset=False)  # type: ignore

        try:
            # alternative check in datafold comared to sklearn
            # self._check_feature_names(reset=False)
            # Reason: in datafold there are also non-string feature names allowed
            nptest.assert_array_equal(
                np.asarray(X.columns), np.asarray(self.feature_names_in_)
            )
        except AssertionError:
            raise ValueError(
                f"model was fit with feature names\n{self.feature_names_in_.tolist()}\n"
                f"but got\n{X.columns.tolist()}"
            )

    def _validate_qois(self, qois, valid_feature_names) -> np.ndarray:

        if qois is not None:
            try:
                qois = np.asarray(qois)
            except Exception:
                raise TypeError("parameter 'qois' must be list-like")

            if qois.ndim != 1:
                raise ValueError(
                    f"'qois' must be a 1-dim. array. " f"Got qois.ndim={qois.ndim}"
                )

            mask_valid_qois = np.isin(qois, valid_feature_names)

            if not mask_valid_qois.all():
                raise ValueError(
                    f"The qois={qois[~mask_valid_qois]} are invalid. Valid "
                    f"feature names are {valid_feature_names}."
                )

        return qois

    def _validate_features_and_time_values(
        self, X: TSCDataFrame, time_values: Optional[np.ndarray]
    ):

        self._check_attributes_set_up(check_attributes=["time_values_in_"])

        if time_values is None:
            time_values = self.time_values_in_

        if not self._has_feature_names(X):
            raise TypeError("only types that support feature names are supported")

        time_values = self._validate_time_values(time_values=time_values)
        self._validate_feature_names(X)

        return X, time_values

    def predict(
        self,
        X: InitialConditionType,
        time_values: Optional[np.ndarray] = None,
        **predict_params,
    ):
        # intended for duck-typing, but provides method layout
        raise NotImplementedError("method not implemented")

    def fit_predict(
        self,
        X: InitialConditionType,
        y=None,
        **fit_params,
    ) -> TSCDataFrame:
        # overwrite if necessary
        self.fit: Callable
        return self.fit(X, **fit_params).predict(X.initial_states())

    def reconstruct(
        self,
        X: TSCDataFrame,
        qois: Optional[Union[np.ndarray, pd.Index, List[str]]] = None,
    ):
        X_reconstruct_ts = []

        for X_ic, time_values in InitialCondition.iter_reconstruct_ic(
            X, n_samples_ic=1
        ):
            X_ts = self.predict(X=X_ic, time_values=time_values)
            X_reconstruct_ts.append(X_ts)

        return pd.concat(X_reconstruct_ts, axis=0)
