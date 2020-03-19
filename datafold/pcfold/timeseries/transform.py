#!/usr/bin/env python3

import itertools
from typing import Union

import numpy as np
import pandas as pd
from scipy.interpolate import Rbf
from sklearn.base import BaseEstimator, clone
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.utils.validation import check_is_fitted

from datafold.pcfold.timeseries import TSCDataFrame
from datafold.pcfold.timeseries.base import (
    FEATURE_NAME_TYPES,
    TRANF_TYPES,
    TSCTransformerMixIn,
)
from datafold.pcfold.timeseries.collection import TSCException


class TSCQoiPreprocess(BaseEstimator, TSCTransformerMixIn):
    VALID_SCALE_NAMES = ["min-max", "standard"]

    def __init__(self, sklearn_transformer, **kwargs):
        """
        See
        https://scikit-learn.org/stable/modules/classes.html#module-sklearn.preprocessing
        for a list of preprocessing classes that can be used. A mandatory requirement
        (for now) is that it supports an inverse mapping.
        """

        self.sklearn_transformer = sklearn_transformer

        if not hasattr(self.sklearn_transformer, "transform") or not hasattr(
            sklearn_transformer, "inverse_transform"
        ):
            raise AttributeError(
                f"transform cls {self.sklearn_transformer} must provide a 'transform' "
                f"and 'inverse_transform' attribute"
            )

    @classmethod
    def from_name(cls, name):

        if name == "min-max":
            return cls(MinMaxScaler(feature_range=(0, 1), copy=True))
        elif name == "standard":
            return cls(StandardScaler(copy=True, with_mean=True, with_std=True))
        else:
            raise ValueError(
                f"name='{name}' is not known. Choose from {cls.VALID_SCALE_NAMES}"
            )

    def fit(self, X: TRANF_TYPES, y=None, **fit_params):

        X = self._validate_data(X)

        if self._has_feature_names(X):
            self._setup_features_input_fit(
                features_in=X.columns, features_out=X.columns
            )
        else:
            self._setup_array_input_fit(features_in=X.shape[1], features_out=X.shape[1])

        self.sklearn_transformer_fit_ = clone(
            estimator=self.sklearn_transformer, safe=True
        )

        X_intern = self._X_to_numpy(X)
        self.sklearn_transformer_fit_.fit(X_intern)

        return self

    def transform(self, X: TRANF_TYPES):
        check_is_fitted(self, "sklearn_transformer_fit_")

        X = self._validate_data(X)
        self._validate_features_transform(X)

        X_intern = self._X_to_numpy(X)
        values = self.sklearn_transformer_fit_.transform(X_intern)
        return self._same_type_X(X=X, values=values, set_columns=self.features_out_[1])

    def fit_transform(self, X: TRANF_TYPES, y=None, **fit_params):

        X = self._validate_data(X)

        if self._has_feature_names(X):
            self._setup_features_input_fit(
                features_in=X.columns, features_out=X.columns
            )
        else:
            self._setup_array_input_fit(features_in=X.shape[1], features_out=X.shape[1])

        self.sklearn_transformer_fit_ = clone(self.sklearn_transformer)
        values = self.sklearn_transformer_fit_.fit_transform(X)

        return self._same_type_X(X=X, values=values, set_columns=self.features_out_[1])

    def inverse_transform(self, X: TRANF_TYPES):
        X_intern = self._X_to_numpy(X)
        values = self.sklearn_transformer_fit_.inverse_transform(X_intern)
        return self._same_type_X(X=X, values=values, set_columns=self.features_in_[1])


class TSCIdentity(BaseEstimator, TSCTransformerMixIn):
    def __init__(self):
        pass

    def fit(self, X: TRANF_TYPES, y=None, **fit_params):

        X = self._validate_data(X)

        if self._has_feature_names(X):
            self._setup_features_input_fit(
                features_in=X.columns, features_out=X.columns
            )
        else:
            self._setup_array_input_fit(features_in=X.shape[1], features_out=X.shape[1])

        # "dummy" attribute to indicate
        self.is_fit_ = True

        return self

    def transform(self, X: TRANF_TYPES):

        check_is_fitted(self, "is_fit_")

        X = self._validate_data(X)
        self._validate_features_transform(X)

        if self.features_in_[0] != X.shape[1]:
            raise ValueError("")  # TODO: make general error

        return X

    def inverse_transform(self, X: TRANF_TYPES):

        check_is_fitted(self, "is_fit_")
        X = self._validate_data(X)
        self._validate_features_inverse_transform(X)
        return X


class TSCPrincipalComponent(PCA, TSCTransformerMixIn):
    def __init__(
        self,
        n_components=2,
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

        X = self._validate_data(X)

        if self._has_feature_names(X):
            self._setup_features_input_fit(
                features_in=X.columns,
                features_out=[f"pca{i}" for i in range(self.n_components)],
            )
        else:
            self._setup_array_input_fit(
                features_in=X.shape[1], features_out=self.n_components
            )

        X_intern = self._X_to_numpy(X)

        # validate happens here:
        return super(TSCPrincipalComponent, self).fit(X_intern, y=y)

    def transform(self, X: TRANF_TYPES):
        check_is_fitted(self)
        X = self._validate_data(X)

        self._validate_features_transform(X)

        X_intern = self._X_to_numpy(X)
        pca_data = super(TSCPrincipalComponent, self).transform(X_intern)
        return self._same_type_X(X, values=pca_data, set_columns=self.features_out_[1])

    def fit_transform(self, X, y=None):

        X = self._validate_data(X)

        if self._has_feature_names(X):
            self._setup_features_input_fit(
                features_in=X.columns,
                features_out=[f"pca{i}" for i in range(self.n_components)],
            )
        else:
            self._setup_array_input_fit(
                features_in=X.shape[1], features_out=self.n_components
            )

        X_intern = self._X_to_numpy(X)
        pca_values = super(TSCPrincipalComponent, self).fit_transform(X_intern, y=y)
        return self._same_type_X(
            X, values=pca_values, set_columns=self.features_out_[1]
        )

    def inverse_transform(self, X: TRANF_TYPES):
        self._validate_features_inverse_transform(X)

        X_intern = self._X_to_numpy(X)
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
        total_n_columns = tsc.shape[1] * (self.delays + 1)

        data = np.zeros([tsc.shape[0], total_n_columns]) * np.nan

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

        if not X.is_const_delta_time():
            raise TSCException.not_const_delta_time()

        return X

    def fit(self, X: FEATURE_NAME_TYPES, y=None, **fit_params):

        self._validate_parameter()
        X = self._validate_data(
            X,
            ensure_feature_name_type=True,
            validate_array_kwargs=dict(ensure_min_samples=1),
        )
        X = self._validate_tsc_properties(X)

        self.delay_indices_ = self._precompute_delay_indices()
        features_out = self._expand_all_delay_columns(X.columns)

        # only TSCDataFrame works here
        self._setup_features_input_fit(
            features_in=X.columns, features_out=features_out,
        )
        return self

    def transform(self, X: FEATURE_NAME_TYPES):

        X = self._validate_data(X, ensure_feature_name_type=True)
        X = self._validate_tsc_properties(X)

        if (X.n_timesteps <= self.delay_indices_.max()).any():
            raise TSCException(
                f"Mismatch of delay and time series length. Shortest time series has "
                f"length {np.array(X.n_timesteps).min()} and maximum delay is "
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
        X = self._validate_data(X, ensure_feature_name_type=True)
        X = self._validate_tsc_properties(X)
        self._validate_features_inverse_transform(X)

        return X.loc[:, self.features_in_[1]]


class TSCRadialBasis(BaseEstimator, TSCTransformerMixIn):
    def __init__(self, centers=None, **rbf_kwargs):
        """
        # TODO: docu note: centers are row-wise!
        Parameters
        ----------
        centers
        rbf_kwargs
        """

        if "mode" in rbf_kwargs:
            raise ValueError("parameter 'mode' is set during fit")

        self.centers = centers.T if centers is not None else None
        self.rbf_kwargs = rbf_kwargs

    def fit(self, X: TRANF_TYPES, y=None, **fit_kwargs):
        X = self._validate_data(X)

        if self.centers is None:
            if self._has_feature_names(X):
                # Due to scipy interpolate.RBF data is transposed.
                _centers = X.to_numpy().T
            else:
                _centers = X.T
        else:
            _centers = self.centers

        n_centers = _centers.shape[1]

        if self._has_feature_names(X):
            self._setup_features_input_fit(
                features_in=X.columns,
                features_out=[f"rbf{i}" for i in range(n_centers)],
            )
        else:
            self._setup_array_input_fit(features_in=X.shape[1], features_out=n_centers)

        if X.shape[1] > 1:
            self.rbf_kwargs["mode"] = "N-D"

        self.rbf_ = Rbf(*_centers, _centers.T, **self.rbf_kwargs)

        # more memory efficient to reference to internal RBF data
        self.centers = self.rbf_.xi
        return self

    def transform(self, X: TRANF_TYPES):
        check_is_fitted(self, attributes=["rbf_"])
        X = self._validate_data(X)
        self._validate_features_transform(X)

        X_intern = X.to_numpy()

        cdist_values = self.rbf_._call_norm(X_intern.T, self.centers)
        rbf_coeff = self.rbf_._function(cdist_values)

        return self._same_type_X(X, values=rbf_coeff, set_columns=self.features_out_[1])

    def inverse_transform(self, X: TRANF_TYPES):
        self._validate_features_inverse_transform(X)

        if self._has_feature_names(X):
            rbf_coeff = X.to_numpy()
        else:
            rbf_coeff = X

        X_inverse = rbf_coeff @ self.rbf_.nodes
        return self._same_type_X(X, values=X_inverse, set_columns=self.features_in_[1])


@NotImplementedError  # TODO: implement if required...
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
