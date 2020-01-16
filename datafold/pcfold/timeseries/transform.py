#!/usr/bin/env python3

import itertools
from typing import Union

import numpy as np
import pandas as pd

from datafold.pcfold.timeseries import TSCDataFrame
from datafold.pcfold.timeseries.collection import TimeSeriesCollectionError


class TimeSeriesTransformMixIn:
    def fit_transform(self, X_ts: TSCDataFrame, **fit_params):
        raise NotImplementedError

    def transform(self, X_ts: TSCDataFrame):
        raise NotImplementedError

    def inverse_transform(self, X_ts: TSCDataFrame):
        raise NotImplementedError


class TSCQoiPreprocess(TimeSeriesTransformMixIn):
    def __init__(self, cls, **kwargs):
        """
        See
        https://scikit-learn.org/stable/modules/classes.html#module-sklearn.preprocessing
        for scaling classes
        """

        if not hasattr(cls, "transform") or not hasattr(cls, "inverse_transform"):
            raise AttributeError(
                f"transform cls {cls} must provide a 'transform' "
                f"and 'inverse_transform' attribute"
            )
        self.transform_cls_ = cls(**kwargs)

    def fit_transform(self, X_ts: TSCDataFrame, **fit_params):
        data = self.transform_cls_.fit_transform(X_ts.to_numpy())
        return TSCDataFrame.from_same_indices_as(X_ts, data)

    def transform(self, X_ts: TSCDataFrame):
        data = self.transform_cls_.transform(X_ts.to_numpy())
        return TSCDataFrame.from_same_indices_as(X_ts, data)

    def inverse_transform(self, X_ts: TSCDataFrame):
        data = self.transform_cls_.inverse_transform(X_ts.to_numpy())
        return TSCDataFrame.from_same_indices_as(X_ts, data)


class TSCQoiScale(TimeSeriesTransformMixIn):

    VALID_NAMES = ["min-max", "standard"]

    def __init__(self, name):
        from sklearn.preprocessing import MinMaxScaler, StandardScaler

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

    def fit_transform(self, X_ts: TSCDataFrame, **fit_params):
        return self._qoi_scaler.fit_transform(X_ts=X_ts)

    def transform(self, X_ts: TSCDataFrame):
        return self._qoi_scaler.transform(X_ts=X_ts)

    def inverse_transform(self, X_ts: TSCDataFrame):
        return self._qoi_scaler.inverse_transform(X_ts=X_ts)


class TSCTakensEmbedding(TimeSeriesTransformMixIn):
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

        self.delay_indices_ = self._precompute_delay_indices()

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
        return cols.tolist() + list(itertools.chain(*expand()))

    def _expand_single_delta_column(self, cols, delay_idx):
        return list(map(lambda q: "d".join([q, str(delay_idx)]), cols))

    def _setup_delayed_timeseries_collection(self, tsc):

        nr_columns_incl_delays = tsc.shape[1] * (self.delays + 1)

        data = np.zeros([tsc.shape[0], nr_columns_incl_delays])
        columns = self._expand_all_delay_columns(tsc.columns)

        delayed_timeseries = TSCDataFrame(
            pd.DataFrame(data, index=tsc.index, columns=columns),
        )

        delayed_timeseries.loc[:, tsc.columns] = tsc
        return delayed_timeseries

    def _shift_timeseries(self, timeseries, delay_idx):

        if self.time_direction == "backward":
            shifted_timeseries = timeseries.shift(
                delay_idx, fill_value=self.fill_value,
            ).copy()
        elif self.time_direction == "forward":
            shifted_timeseries = timeseries.shift(
                -1 * delay_idx, fill_value=self.fill_value
            ).copy()
        else:
            raise ValueError(f"time_direction={self.time_direction} not known.")

        columns = self._expand_single_delta_column(timeseries.columns, delay_idx)
        shifted_timeseries.columns = columns

        return shifted_timeseries

    def fit_transform(self, X_ts: TSCDataFrame, **fit_params):
        return self.fit(X_ts).transform(X_ts=X_ts)

    def transform(self, X_ts: TSCDataFrame):

        if not X_ts.is_const_dt():
            raise TimeSeriesCollectionError("dt is not const")

        if (X_ts.lengths_time_series <= self.delay_indices_.max()).any():
            raise TimeSeriesCollectionError(
                f"Mismatch of delay and time series length. Shortest time series has "
                f"length "
                f"{np.array(X_ts.lengths_time_series).min()} and maximum delay is "
                f"{self.delay_indices_.max()}"
            )

        delayed_tsc = self._setup_delayed_timeseries_collection(X_ts)

        for i, ts in X_ts.itertimeseries():
            for delay_idx in self.delay_indices_:
                shifted_timeseries = self._shift_timeseries(ts, delay_idx)
                delayed_tsc.loc[
                    i, shifted_timeseries.columns
                ] = shifted_timeseries.values

        return delayed_tsc


@DeprecationWarning  # TODO: implement if required...
class TimeFiniteDifference(object):

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
