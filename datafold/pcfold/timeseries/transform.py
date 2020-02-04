#!/usr/bin/env python3

import itertools
from typing import Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.utils.validation import check_is_fitted

from datafold.pcfold.timeseries import TSCDataFrame
from datafold.pcfold.timeseries.base import (
    INDICES_TYPES,
    TRANF_TYPES,
    TSCTransformerMixIn,
)
from datafold.pcfold.timeseries.collection import TimeSeriesCollectionError


class TSCQoiPreprocess(BaseEstimator, TSCTransformerMixIn):
    def __init__(self, transform_cls, **kwargs):
        """
        See
        https://scikit-learn.org/stable/modules/classes.html#module-sklearn.preprocessing
        for a list of preprocessing classes that can be used. A mandatory requirement
        (for now) is that it supports an inverse mapping.
        """

        if not hasattr(transform_cls, "transform") or not hasattr(
            transform_cls, "inverse_transform"
        ):
            raise AttributeError(
                f"transform cls {transform_cls} must provide a 'transform' "
                f"and 'inverse_transform' attribute"
            )
        self.transform_cls_ = transform_cls(**kwargs)

    def fit(self, X: TRANF_TYPES, y=None, **fit_params):

        X = self._validate(X)

        if self._has_indices(X):
            self._setup_indices_based_fit(features_in=X.columns, features_out=X.columns)
        else:
            self._setup_array_based_fit(features_in=X.shape[1], features_out=X.shape[1])

        X_intern = self._intern_X_to_numpy(X)
        self.transform_cls_.fit(X_intern)

        self.is_fit_ = True

        return self

    def transform(self, X: TRANF_TYPES):
        check_is_fitted(self, "is_fit_")

        X = self._validate(X)
        self._validate_features_transform(X)

        X_intern = self._intern_X_to_numpy(X)
        values = self.transform_cls_.transform(X_intern)
        return self._same_type_X(X=X, values=values, set_columns=self.features_out_[1])

    def fit_transform(self, X: TRANF_TYPES, y=None, **fit_params):

        X = self._validate(X)

        if self._has_indices(X):
            self._setup_indices_based_fit(features_in=X.columns, features_out=X.columns)
        else:
            self._setup_array_based_fit(features_in=X.shape[1], features_out=X.shape[1])

        values = self.transform_cls_.fit_transform(X)
        self.is_fit_ = True

        return self._same_type_X(X=X, values=values, set_columns=self.features_out_[1])

    def inverse_transform(self, X: TRANF_TYPES):
        X_intern = self._intern_X_to_numpy(X)
        values = self.transform_cls_.inverse_transform(X_intern)
        return self._same_type_X(X=X, values=values, set_columns=self.features_in_[1])


class TSCIdentity(BaseEstimator, TSCTransformerMixIn):
    def __init__(self):
        pass

    def fit(self, X: TRANF_TYPES, y=None, **fit_params):

        X = self._validate(X)

        if self._has_indices(X):
            self._setup_indices_based_fit(features_in=X.columns, features_out=X.columns)
        else:
            self._setup_array_based_fit(features_in=X.shape[1], features_out=X.shape[1])

        # "dummy" attribute to indicate
        self.is_fit_ = True

        return self

    def transform(self, X: TRANF_TYPES):

        check_is_fitted(self, "is_fit_")

        X = self._validate(X)
        self._validate_features_transform(X)

        if self.features_in_[0] != X.shape[1]:
            raise ValueError("")  # TODO: make general error

        return X

    def inverse_transform(self, X: TRANF_TYPES):

        check_is_fitted(self, "is_fit_")
        X = self._validate(X)
        self._validate_features_inverse_transform(X)
        return X


class TSCQoiScale(TSCQoiPreprocess):

    VALID_NAMES = ["min-max", "standard"]

    def __init__(self, name):
        """Convenience wrapper to use often used """
        if name == "min-max":
            _cls = MinMaxScaler
            kwargs = dict(feature_range=(0, 1))
        elif name == "standard":
            _cls = StandardScaler
            kwargs = dict(with_mean=True, with_std=True)
        else:
            raise ValueError(
                f"name={name} is not known. Choose from {self.VALID_NAMES}"
            )

        super(TSCQoiScale, self).__init__(transform_cls=_cls, **kwargs)


class TSCPrincipalComponent(PCA, TSCTransformerMixIn):
    def __init__(
        self,
        n_components,
        copy=True,
        whiten=False,
        svd_solver="auto",
        tol=0.0,
        iterated_power="auto",
        random_state=None,
    ):
        super(TSCPrincipalComponent, self).__init__(
            n_components=n_components,
            copy=copy,
            whiten=whiten,
            svd_solver=svd_solver,
            tol=tol,
            iterated_power=iterated_power,
            random_state=random_state,
        )

    def fit(self, X: TRANF_TYPES, y=None, **fit_params):

        if self._has_indices(X):
            self._setup_indices_based_fit(
                features_in=X.columns,
                features_out=[f"pca{i}" for i in range(self.n_components)],
            )
        else:
            self._setup_array_based_fit(
                features_in=X.shape[1], features_out=self.n_components
            )

        X_intern = self._intern_X_to_numpy(X)

        # validate happens here:
        return super(TSCPrincipalComponent, self).fit(X_intern, y=y)

    def transform(self, X: TRANF_TYPES):

        self._validate_features_transform(X)

        X_intern = self._intern_X_to_numpy(X)
        pca_data = super(TSCPrincipalComponent, self).transform(X_intern)

        return self._same_type_X(X, values=pca_data, set_columns=self.features_out_[1])

    def fit_transform(self, X, y=None):
        if self._has_indices(X):
            self._setup_indices_based_fit(
                features_in=X.columns,
                features_out=[f"pca{i}" for i in range(self.n_components)],
            )
        else:
            self._setup_array_based_fit(
                features_in=X.shape[1], features_out=self.n_components
            )

        X_intern = self._intern_X_to_numpy(X)
        pca_values = super(TSCPrincipalComponent, self).fit_transform(X_intern, y=y)
        return self._same_type_X(
            X, values=pca_values, set_columns=self.features_out_[1]
        )

    def inverse_transform(self, X: TRANF_TYPES):
        self._validate_features_inverse_transform(X)

        X_intern = self._intern_X_to_numpy(X)
        data_orig_space = super(TSCPrincipalComponent, self).inverse_transform(X_intern)

        return self._same_type_X(
            X, values=data_orig_space, set_columns=self.features_in_[1]
        )


class TSCTakensEmbedding(BaseEstimator, TSCTransformerMixIn):

    VALID_TIME_DIRECTION = ["forward", "backward"]

    def __init__(
        self,
        lag: int = 0,
        delays: int = 10,
        frequency: int = 1,
        time_direction="backward",
        fillin_handle: Union[str, float] = "remove",
    ):
        """
        fillin_handle:
            If 'value': all fill-ins will be set to this value
            If 'remove': all rows that are affected of fill-ins are removed.
        """

        self.lag = lag
        self.delays = delays
        self.frequency = frequency
        self.time_direction = time_direction
        self.fillin_handle = fillin_handle

    def _validate_parameter(self):
        from sklearn.utils.validation import check_scalar

        check_scalar(
            self.lag, name="lag", target_type=(np.integer, int), min_val=0, max_val=None
        )

        check_scalar(
            self.delays,
            name="delays",
            target_type=(np.integer, int),
            min_val=1,
            max_val=None,
        )

        check_scalar(
            self.frequency,
            name="delays",
            target_type=(np.integer, int),
            min_val=1,
            max_val=None,
        )

        if self.frequency > 1 and self.delays <= 1:
            raise ValueError(
                f"if frequency (={self.frequency} is larger than 1, "
                f"then number for delays (={self.delays}) has to be larger "
                "than 1)"
            )

        if self.time_direction not in self.VALID_TIME_DIRECTION:
            raise ValueError(
                f"time_direction={self.time_direction} invalid. Valid choices: "
                f"{self.VALID_TIME_DIRECTION}"
            )

    def _precompute_delay_indices(self):
        # zero delay (original data) is not treated
        return self.lag + (
            np.arange(1, (self.delays * self.frequency) + 1, self.frequency)
        )

    def _expand_all_delay_columns(self, cols):
        def expand():
            delayed_columns = list()
            for didx in self.delay_indices_:
                delayed_columns.append(self._expand_single_delta_column(cols, didx))
            return delayed_columns

        # the name of the original indices is not changed, therefore append the delay
        # indices to
        columns_names = cols.tolist() + list(itertools.chain(*expand()))

        return pd.Index(
            columns_names, dtype=np.str, copy=False, name=TSCDataFrame.IDX_QOI_NAME
        )

    def _expand_single_delta_column(self, cols, delay_idx):
        return list(map(lambda q: ":d".join([q, str(delay_idx)]), cols))

    def _allocate_delayed_qois(self, tsc):

        # original columns + delayed columns
        total_nr_columns = tsc.shape[1] * (self.delays + 1)

        data = np.zeros([tsc.shape[0], total_nr_columns]) * np.nan

        delayed_tsc = TSCDataFrame(data, index=tsc.index, columns=self.features_out_[1])

        delayed_tsc.loc[:, tsc.columns] = tsc
        return delayed_tsc

    def _shift_timeseries(self, single_ts, delay_idx):

        if self.fillin_handle == "remove":
            fill_value = np.nan
        else:
            fill_value = self.fillin_handle

        if self.time_direction == "backward":
            shifted_timeseries = single_ts.shift(
                delay_idx, fill_value=fill_value,
            ).copy()
        elif self.time_direction == "forward":
            shifted_timeseries = single_ts.shift(
                -1 * delay_idx, fill_value=fill_value
            ).copy()
        else:
            raise ValueError(
                f"time_direction={self.time_direction} not known. "
                f"Please report bug."
            )

        shifted_timeseries.columns = self._expand_single_delta_column(
            single_ts.columns, delay_idx
        )

        return shifted_timeseries

    def _validate_tsc_properties(self, X):

        # TODO: ensure minimum of samples per ID, otherwise there are columns with only
        #  fill-in

        if not isinstance(X, TSCDataFrame):
            if X.index.nlevels == 1:
                # X is required to be TSC here
                X = TSCDataFrame.from_single_timeseries(X)
            else:
                X = TSCDataFrame(X)

        if not X.is_const_dt():
            raise TimeSeriesCollectionError("dt is not constant")

        return X

    def fit(self, X: INDICES_TYPES, y=None, **fit_params):

        self._validate_parameter()
        X = self._validate(X, ensure_index_type=True, ensure_min_samples=1)
        X = self._validate_tsc_properties(X)

        self.delay_indices_ = self._precompute_delay_indices()
        features_out = self._expand_all_delay_columns(X.columns)

        # only TSCDataFrame works here
        self._setup_indices_based_fit(
            features_in=X.columns, features_out=features_out,
        )
        return self

    def transform(self, X: INDICES_TYPES):

        X = self._validate(X, ensure_index_type=True)
        X = self._validate_tsc_properties(X)

        if (X.lengths_time_series <= self.delay_indices_.max()).any():
            raise TimeSeriesCollectionError(
                f"Mismatch of delay and time series length. Shortest time series has "
                f"length {np.array(X.lengths_time_series).min()} and maximum delay is "
                f"{self.delay_indices_.max()}."
            )
        self._validate_features_transform(X)

        X = self._allocate_delayed_qois(X)

        # Compute the shifts --> per single time series
        for i, ts in X.loc[:, self.features_in_[1]].itertimeseries():
            for delay_idx in self.delay_indices_:
                shifted_timeseries = self._shift_timeseries(ts, delay_idx)
                X.loc[i, shifted_timeseries.columns] = shifted_timeseries.values

        if self.fillin_handle == "remove":
            bool_idx = np.logical_not(np.sum(pd.isnull(X), axis=1).astype(np.bool))
            X = X.loc[bool_idx]

        return X

    def inverse_transform(self, X: TRANF_TYPES):
        X = self._validate(X, ensure_index_type=True)
        X = self._validate_tsc_properties(X)
        self._validate_features_inverse_transform(X)

        return X.loc[:, self.features_in_[1]]


@DeprecationWarning  # TODO: implement if required...
class TSCFiniteDifference(object):

    # TODO: provide longer shifts? This could give some average of slow and fast
    #  variables...

    def __init__(self, scheme="centered"):
        self.scheme = scheme

    def _get_shift_negative(self):
        pass

    def _get_shift_positive(self):
        pass

    def _finite_difference_backward(self):
        pass

    def _finite_difference_forward(self, tsc):
        pass

    def _finite_difference_centered(self):
        pass

    def apply(self, tsc: TSCDataFrame):

        for i, traj in tsc:
            pass
