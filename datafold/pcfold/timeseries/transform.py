#!/usr/bin/env python3

import itertools

import numpy as np
import pandas as pd
import pandas.testing as pdtest
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from datafold.pcfold.timeseries import TSCDataFrame
from datafold.pcfold.timeseries.collection import TimeSeriesCollectionError


class TSCTransformMixIn:
    def _save_columns(self, fit_columns: pd.Index, transform_columns: pd.Index = None):
        self._fit_columns = fit_columns

        if transform_columns is not None:
            self._transform_columns = transform_columns

    def fit(self, X_ts: TSCDataFrame, **fit_params):
        raise NotImplementedError

    def transform(self, X_ts: TSCDataFrame):
        raise NotImplementedError

    def fit_transform(self, X_ts: TSCDataFrame, **fit_params):
        return self.fit(X_ts, **fit_params).transform(X_ts=X_ts)

    def inverse_transform(self, X_ts: TSCDataFrame):
        raise NotImplementedError

    def _check_fit_columns(self, X_ts: TSCDataFrame):
        if not hasattr(self, "_fit_columns"):
            raise RuntimeError("_fit_columns is not set. Please report bug.")

        pdtest.assert_index_equal(right=self._fit_columns, left=X_ts.columns)

    def _check_transform_columns(self, X_ts: TSCDataFrame):
        if not hasattr(self, "_transform_columns"):
            raise RuntimeError("_transform_columns is not set. Please report bug.")

        pdtest.assert_index_equal(right=self._transform_columns, left=X_ts.columns)

    def _return_same_type_X(self, X_ts, values, columns=None):

        _type = type(X_ts)

        if columns is None:
            columns = X_ts.columns

        if isinstance(X_ts, TSCDataFrame):
            # NOTE: order is important here TSCDataFrame is also a DataFrame, so first
            # check for the special case, then for the more general case.
            return TSCDataFrame.from_same_indices_as(
                X_ts, values=values, except_columns=columns
            )
        if isinstance(X_ts, pd.DataFrame):
            return pd.DataFrame(values, index=X_ts.index, columns=columns)
        else:
            raise TypeError


class TSCQoiPreprocess(TSCTransformMixIn):
    def __init__(self, cls, **kwargs):
        """
        See
        https://scikit-learn.org/stable/modules/classes.html#module-sklearn.preprocessing
        for a list of preprocessing classes that can be used. A mandatory requirement
        (for now) is that it supports an inverse mapping.
        """

        if not hasattr(cls, "transform") or not hasattr(cls, "inverse_transform"):
            raise AttributeError(
                f"transform cls {cls} must provide a 'transform' "
                f"and 'inverse_transform' attribute"
            )
        self.transform_cls_ = cls(**kwargs)

    def fit(self, X_ts: TSCDataFrame, **fit_params):
        self._save_columns(X_ts.columns)
        self.transform_cls_.fit(X_ts.to_numpy())
        return self

    def transform(self, X_ts: TSCDataFrame):
        self._check_fit_columns(X_ts=X_ts)
        values = self.transform_cls_.transform(X_ts.to_numpy())
        return self._return_same_type_X(X_ts=X_ts, values=values)

    def fit_transform(self, X_ts: TSCDataFrame, **fit_params):
        self._save_columns(fit_columns=X_ts.columns)
        values = self.transform_cls_.fit_transform(X_ts)
        return self._return_same_type_X(X_ts=X_ts, values=values)

    def inverse_transform(self, X_ts: TSCDataFrame):
        self._check_fit_columns(X_ts=X_ts)
        values = self.transform_cls_.inverse_transform(X_ts.to_numpy())
        return self._return_same_type_X(X_ts=X_ts, values=values)


class TSCQoiScale(TSCTransformMixIn):

    VALID_NAMES = ["min-max", "standard"]

    def __init__(self, name):

        if name == "min-max":
            self._qoi_scaler = TSCQoiPreprocess(cls=MinMaxScaler, feature_range=(0, 1))

        elif name == "standard":
            self._qoi_scaler = TSCQoiPreprocess(
                cls=StandardScaler, with_mean=True, with_std=True
            )
        else:
            raise ValueError(
                f"name={name} is not known. Choose from {self.VALID_NAMES}"
            )

    def fit(self, X_ts: TSCDataFrame, **fit_params):
        self._qoi_scaler.fit(X_ts=X_ts, **fit_params)

    def transform(self, X_ts: TSCDataFrame):
        return self._qoi_scaler.transform(X_ts=X_ts)

    def fit_transform(self, X_ts: TSCDataFrame, **fit_params):
        return self._qoi_scaler.fit_transform(X_ts=X_ts, **fit_params)

    def inverse_transform(self, X_ts: TSCDataFrame):
        return self._qoi_scaler.inverse_transform(X_ts=X_ts)


class TSCPrincipalComponents(TSCTransformMixIn):
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

        self._pca = PCA(
            n_components=n_components,
            copy=copy,
            whiten=whiten,
            svd_solver=svd_solver,
            tol=tol,
            iterated_power=iterated_power,
            random_state=random_state,
        )

    def fit(self, X_ts: TSCDataFrame, **fit_params):
        self._pca.fit(X=X_ts, y=None)

        column_names = ["pca{i}" for i in range(self._pca.n_components_)]
        transform_columns = pd.Index(
            column_names, dtype=np.str, copy=False, name=TSCDataFrame.IDX_QOI_NAME
        )

        self._save_columns(
            fit_columns=X_ts.columns, transform_columns=transform_columns
        )
        return self

    def transform(self, X_ts: TSCDataFrame):
        self._check_fit_columns(X_ts=X_ts)
        pca_data = self._pca.transform(X_ts.to_numpy())

        pca_data_tsc = TSCDataFrame.from_same_indices_as(
            X_ts, pca_data, except_columns=self._transform_columns
        )

        return pca_data_tsc

    def inverse_transform(self, X_ts: TSCDataFrame):
        self._check_transform_columns(X_ts=X_ts)
        data_orig_space = self._pca.inverse_transform(X_ts.to_numpy())

        data_orig_space = TSCDataFrame.from_same_indices_as(
            X_ts, values=data_orig_space, except_columns=self._fit_columns
        )

        return data_orig_space


class TSCTakensEmbedding(TSCTransformMixIn):
    def __init__(
        self,
        lag: int,
        delays: int,
        frequency: int,
        time_direction="backward",
        fill_value: float = np.nan,
    ):

        if lag < 0:
            raise ValueError(f"Lag has to be non-negative. Got lag={lag}")

        if delays < 1:
            raise ValueError(
                f"delays has to be an integer larger than zero. Got delays={frequency}"
            )

        if frequency < 1:
            raise ValueError(
                f"frequency has to be an integer larger than zero. Got frequency"
                f"={frequency}"
            )

        if frequency > 1 and delays == 1:
            raise ValueError("frequency must be 1, if delays=1")

        if time_direction not in ["backward", "forward"]:
            raise ValueError(
                f"time_direction={time_direction} invalid. Valid choices: "
                f"{['backward', 'forward']}"
            )

        self.lag = lag
        self.delays = delays
        self.frequency = frequency
        self.time_direction = time_direction
        self.fill_value = fill_value

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

    def _allocate_delayed_qoi(self, tsc):

        # original columns + delayed columns
        total_nr_columns = tsc.shape[1] * (self.delays + 1)

        data = np.empty([tsc.shape[0], total_nr_columns])

        delayed_tsc = TSCDataFrame(
            data, index=tsc.index, columns=self._transform_columns
        )

        delayed_tsc.loc[:, tsc.columns] = tsc
        return delayed_tsc

    def _shift_timeseries(self, single_ts, delay_idx):

        if self.time_direction == "backward":
            shifted_timeseries = single_ts.shift(
                delay_idx, fill_value=self.fill_value,
            ).copy()
        elif self.time_direction == "forward":
            shifted_timeseries = single_ts.shift(
                -1 * delay_idx, fill_value=self.fill_value
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

    def fit(self, X_ts, **fit_params):

        # TODO: Check that there are no NaN values present. (the integrate the option
        #  to remove fill in rows.

        self.delay_indices_ = self._precompute_delay_indices()

        fit_columns = X_ts.columns
        transform_columns = self._expand_all_delay_columns(fit_columns)
        self._save_columns(
            fit_columns=X_ts.columns, transform_columns=transform_columns
        )

        return self

    def transform(self, X_ts: TSCDataFrame):

        self._check_fit_columns(X_ts=X_ts)

        if not X_ts.is_const_dt():
            raise TimeSeriesCollectionError("dt is not constant")

        if X_ts.is_contain_nans():
            raise ValueError("The TSCDataFrame must be NaN free")

        if (X_ts.lengths_time_series <= self.delay_indices_.max()).any():
            raise TimeSeriesCollectionError(
                f"Mismatch of delay and time series length. Shortest time series has "
                f"length {np.array(X_ts.lengths_time_series).min()} and maximum delay is "
                f"{self.delay_indices_.max()}"
            )

        X_ts = self._allocate_delayed_qoi(X_ts)

        # Compute the shifts --> per single time series
        for i, ts in X_ts.loc[:, self._fit_columns].itertimeseries():
            for delay_idx in self.delay_indices_:
                shifted_timeseries = self._shift_timeseries(ts, delay_idx)
                X_ts.loc[i, shifted_timeseries.columns] = shifted_timeseries.values

        return X_ts

    def inverse_transform(self, X_ts: TSCDataFrame):
        self._check_transform_columns(X_ts=X_ts)
        return X_ts.loc[:, self._fit_columns]


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
