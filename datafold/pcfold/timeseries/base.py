#!/usr/bin/env python3

from typing import Optional, Union

import numpy as np
import pandas as pd
import pandas.testing as pdtest

from datafold.pcfold.timeseries.collection import TSCDataFrame
from datafold.pcfold.timeseries.metric import TSCMetric

TRANF_TYPES = Union[TSCDataFrame, pd.DataFrame, np.ndarray]


class TSCTransformMixIn:
    def _has_indices(self, _obj):
        return isinstance(_obj, pd.DataFrame)

    def _save_columns(self, fit_columns: pd.Index, transform_columns: pd.Index = None):
        self._fit_columns = fit_columns

        if transform_columns is not None:
            self._transform_columns = transform_columns

    def fit(self, X: TRANF_TYPES, y=None, **fit_params):
        if self._has_indices(X):
            transform_columns = fit_params.pop("transform_columns", X.columns)

            if isinstance(transform_columns, list):
                transform_columns = pd.Index(
                    transform_columns, dtype=np.str, name=TSCDataFrame.IDX_QOI_NAME
                )

            self._save_columns(X.columns, transform_columns=transform_columns)
        else:
            self._fit_columns = None
            self._transform_columns = None

    def transform(self, X: TRANF_TYPES):
        if self._has_indices(X):
            self._check_fit_columns(X=X)

    def fit_transform(self, X: TRANF_TYPES, y=None, **fit_params):
        if self._has_indices(X):
            self._save_columns(fit_columns=X.columns, transform_columns=X.columns)
        return self.fit(X, y, **fit_params).transform(X=X)

    def inverse_transform(self, X: TRANF_TYPES):
        if self._has_indices(X):
            self._check_transform_columns(X=X)

    def _check_fit_columns(self, X: TRANF_TYPES):
        if not hasattr(self, "_fit_columns"):
            raise RuntimeError("_fit_columns is not set. Please report bug.")

        pdtest.assert_index_equal(right=self._fit_columns, left=X.columns)

    def _check_transform_columns(self, X: TRANF_TYPES):
        if not hasattr(self, "_transform_columns"):
            raise RuntimeError("_transform_columns is not set. Please report bug.")

        pdtest.assert_index_equal(right=self._transform_columns, left=X.columns)

    def _same_type_X(
        self, X: TRANF_TYPES, values: np.ndarray, columns=None
    ) -> TRANF_TYPES:

        _type = type(X)

        if self._has_indices(X) and columns is None:
            columns = X.columns

        if isinstance(X, TSCDataFrame):
            # NOTE: order is important here TSCDataFrame is also a DataFrame, so first
            # check for the special case, then for the more general case.

            return TSCDataFrame.from_same_indices_as(
                X, values=values, except_columns=columns
            )
        elif isinstance(X, pd.DataFrame):
            return pd.DataFrame(values, index=X.index, columns=columns)
        elif isinstance(X, np.ndarray):
            return values
        else:
            raise TypeError


PRE_FIT_TYPES = TSCDataFrame
PRE_IC_TYPES = Union[pd.DataFrame, np.ndarray]


class TSCPredictMixIn:
    def fit(self, X: PRE_FIT_TYPES, **fit_params):
        # NOTE: Currently, only TSCDataFrame is supported as input!
        raise NotImplementedError

    def predict(self, X: PRE_IC_TYPES, t, **predict_params):
        # NOTE the definition of predict cannot be a TSC. Best is provided as a
        # pd.DataFrame with all the information...
        raise NotImplementedError

    def fit_predict(self, X: PRE_FIT_TYPES) -> TSCDataFrame:
        # TODO: to be consistent this would require **fit_params and **predict_params,
        #  no kwargs for now to handle this, in case this becomes an issue.

        # Note: this is an non-optimized way. To optimize this case, overwrite this.
        X_ic = X.initial_states_df()
        t = X.time_indices(unique_values=True)
        return self.fit(X).predict(X_ic, t)

    def score(
        self,
        X_true: PRE_FIT_TYPES,
        X_pred: PRE_FIT_TYPES,
        metric="rmse",
        mode="qoi",
        scaling="id",
        sample_weight: Optional[np.ndarray] = None,
    ):

        TSCMetric(metric=metric, mode=mode, scaling=scaling).score(y_true)
