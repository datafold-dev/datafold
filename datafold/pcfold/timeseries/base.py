#!/usr/bin/env python3

from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import pandas.testing as pdtest
from sklearn.base import TransformerMixin
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_array, check_is_fitted

from datafold.pcfold.timeseries.collection import TSCDataFrame
from datafold.pcfold.timeseries.metric import TSCMetric
from datafold.utils.datastructure import if1dim_rowvec

INDICES_TYPES = Union[TSCDataFrame, pd.DataFrame]

# types allowed for transformation
TRANF_TYPES = Union[INDICES_TYPES, np.ndarray]

# types allowed for predict
PRE_FIT_TYPES = TSCDataFrame
PRE_IC_TYPES = Union[pd.Series, pd.DataFrame, np.ndarray]


class TSCBaseMixIn:
    def _strictly_pandas_df(self, df):
        # This returns False for subclasses (TSCDataFrame)
        return type(df) == pd.DataFrame

    def _has_indices(self, _obj):
        return isinstance(_obj, pd.DataFrame)

    def _intern_X_to_numpy(self, X):
        if self._has_indices(X):
            X = X.to_numpy()
            # a row in a df is always a single sample (which requires to be
            # represented in a 2D matrix)
            return if1dim_rowvec(X)
        else:
            return X

    def _setup_array_based_fit(self, features_in, features_out):
        self.features_in_ = (features_in, None)
        self.features_out_ = (features_out, None)

    def _setup_indices_based_fit(
        self, features_in: pd.Index, features_out: Union[List[str], pd.Index]
    ):
        # TODO: checks about columns
        #  -- no MultiIndex
        #  -- no duplicates in columns
        #  -- features_names_in needs to match X.shape[1]

        self.features_in_ = (len(features_in), features_in)

        if isinstance(features_out, list):
            features_out = pd.Index(
                features_out, dtype=np.str, name=TSCDataFrame.IDX_QOI_NAME,
            )
        self.features_out_ = (len(features_out), features_out)

    def _is_indices_set_up(self):

        try:
            check_is_fitted(
                self, attributes=["features_in_", "features_out_"],
            )
        except NotFittedError:
            return False
        else:
            return True


class TSCTransformerMixIn(TSCBaseMixIn, TransformerMixin):
    def _validate(self, X, enforce_index_type=False, **validate_kwargs):
        """Provides a general function to check data -- can be overwritten if an
        implementation requires different checks."""

        if enforce_index_type and not self._has_indices(X):
            raise TypeError("")

        if self._has_indices(X):
            X_check = X.to_numpy()
        else:
            X_check = X

        X_check = check_array(
            X_check,
            accept_sparse=validate_kwargs.pop("accept_sparse", False),
            accept_large_sparse=validate_kwargs.pop("accept_large_sparse", False),
            dtype=validate_kwargs.pop("dtype", "numeric"),
            order=validate_kwargs.pop("order", None),
            copy=validate_kwargs.pop("copy", False),
            force_all_finite=validate_kwargs.pop("force_all_finite", True),
            ensure_2d=validate_kwargs.pop("ensure_2d", True),
            allow_nd=validate_kwargs.pop("allow_nd", False),
            ensure_min_samples=validate_kwargs.pop("ensure_min_samples", 1),
            ensure_min_features=validate_kwargs.pop("ensure_min_features", 1),
            estimator=self,
        )

        if validate_kwargs != {}:
            raise ValueError(
                f"{validate_kwargs.keys()} are no valid validation keys. "
                "Please report bug."
            )

        if self._has_indices(X):
            return X
        else:
            return X_check

    def _validate_features_transform(self, X: TRANF_TYPES):

        if not self._is_indices_set_up():
            raise RuntimeError("_transform_columns is not set. Please report bug.")

        if self._has_indices(X):
            if self.features_in_[1] is None:
                # TODO -- fit was called with array but now try with DataFrame
                raise ValueError("")
            self._validate_feature_names(X=X, direction="transform")
        else:
            if self.features_in_[0] != X.shape[1]:
                # TODO
                raise ValueError("")

    def _validate_features_inverse_transform(self, X: TRANF_TYPES):
        if not self._is_indices_set_up():
            raise RuntimeError("_transform_columns is not set. Please report bug.")

        if self._has_indices(X):
            if self.features_in_[1] is None:
                # TODO -- fit was called with array but now try with DataFrame
                raise ValueError("")

            # TODO: this error is also raised if user forgot to call fit, so there
            #  should be checked before

            self._validate_feature_names(X, direction="inverse")

    def _validate_feature_names(self, X: TRANF_TYPES, direction):

        _check_features: Tuple[int, pd.Index]

        if direction == "transform":
            _check_features = self.features_in_
        elif direction == "inverse":
            _check_features = self.features_out_
        else:
            raise RuntimeError("Please report bug.")

        if self._has_indices(X):
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
    def fit(self, X: PRE_FIT_TYPES, **fit_params):
        raise NotImplementedError

    def predict(self, X: PRE_IC_TYPES, t, **predict_params):
        # NOTE the definition of predict cannot be a TSC. Best is provided as a
        # pd.DataFrame with all the information...
        raise NotImplementedError

    def fit_predict(self, X: PRE_FIT_TYPES, y=None) -> TSCDataFrame:
        # TODO: to be consistent this would require **fit_params and **predict_params,
        #  no kwargs for now to handle this, in case this becomes an issue.

        # Note: this is an non-optimized way. To optimize this case, overwrite this.
        X_ic = X.initial_states_df()
        t = X.time_indices(unique_values=True)
        return self.fit(X=X, y=y).predict(X_ic, t)

    def score(
        self,
        X_true: PRE_FIT_TYPES,
        X_pred: PRE_FIT_TYPES,
        metric="rmse",
        mode="qoi",
        scaling="id",
        sample_weight: Optional[np.ndarray] = None,
        multi_qoi="raw_values",
    ):

        return TSCMetric(metric=metric, mode=mode, scaling=scaling).score(
            y_true=X_true,
            y_pred=X_pred,
            sample_weight=sample_weight,
            multi_qoi=multi_qoi,
        )
