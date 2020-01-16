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


class TSCQoiTransform(TimeSeriesTransformMixIn):

    VALID_NAMES = ["min-max", "standard"]

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

    @classmethod
    def from_name(cls, name):
        from sklearn.preprocessing import MinMaxScaler, StandardScaler

        if name == "min-max":
            return cls(cls=MinMaxScaler, feature_range=(0, 1))
        elif name == "mean":
            return cls(cls=StandardScaler, with_mean=True, with_std=False)
        elif name == "standard":
            return cls(cls=StandardScaler, with_mean=True, with_std=True)
        else:
            raise ValueError(f"name={name} is not known. Choose from {cls.VALID_NAMES}")

    def fit_transform(self, X_ts: TSCDataFrame, **fit_params):
        data = self.transform_cls_.fit_transform(X_ts.to_numpy())
        return TSCDataFrame.from_same_indices_as(X_ts, data)

    def transform(self, X_ts: TSCDataFrame):
        data = self.transform_cls_.transform(X_ts.to_numpy())
        return TSCDataFrame.from_same_indices_as(X_ts, data)

    def inverse_transform(self, X_ts: TSCDataFrame):
        data = self.transform_cls_.inverse_transform(X_ts.to_numpy())
        return TSCDataFrame.from_same_indices_as(X_ts, data)


@DeprecationWarning
class NormalizeQoi(object):
    VALID_STRATEGIES = ["id", "min-max", "mean", "standard"]

    def __init__(self, normalize_strategy: Union[str, dict], undo: bool):

        if undo and not isinstance(normalize_strategy, dict):
            raise ValueError(
                "If revert=True, the parameter normalize_strategy has to be a dict, "
                "containing the corresponding normalization factors"
            )

        # if the normalize_strategy is a str, then compute based on the handled
        # TSCDataFrame else if a dict (as returned from normalize()) it uses the
        # information there
        self.compute_norm_factor = isinstance(normalize_strategy, str)

        normalize_strategy = self.check_normalize_qoi_strategy(normalize_strategy)

        if isinstance(normalize_strategy, str):
            # make a new dict with consisting information
            self.normalize_strategy = {"strategy": normalize_strategy}
        else:
            self.normalize_strategy = normalize_strategy

        self.undo = undo

    @staticmethod
    def check_normalize_qoi_strategy(strategy: Union[str, dict]):

        if isinstance(strategy, str) and strategy not in NormalizeQoi.VALID_STRATEGIES:
            raise ValueError(
                f"strategy={strategy} not valid. Choose from "
                f" {NormalizeQoi.VALID_STRATEGIES}"
            )

        elif isinstance(strategy, dict):
            if "strategy" not in strategy.keys():
                raise ValueError("Not a valid dict to describe normalization strategy.")

        return strategy

    def _id(self, df):
        return df

    def _min_max(self, df):

        if self.compute_norm_factor:
            self.normalize_strategy["min"], self.normalize_strategy["max"] = (
                df.min(),
                df.max(),
            )

        if not self.undo:

            return (df - self.normalize_strategy["min"]) / (
                self.normalize_strategy["max"] - self.normalize_strategy["min"]
            )
        else:
            self._check_columns_present(df.columns, ["min", "max"])

            return (
                df * (self.normalize_strategy["max"] - self.normalize_strategy["min"])
                + self.normalize_strategy["min"]
            )

    def _mean(self, df):

        if self.compute_norm_factor:
            (
                self.normalize_strategy["mean"],
                self.normalize_strategy["min"],
                self.normalize_strategy["max"],
            ) = (
                df.mean(),
                df.min(),
                df.max(),
            )

        if not self.undo:
            return (df - self.normalize_strategy["mean"]) / (
                self.normalize_strategy["max"] - self.normalize_strategy["min"]
            )
        else:
            self._check_columns_present(df.columns, ["min", "max", "mean"])
            return (
                df * (self.normalize_strategy["max"] - self.normalize_strategy["min"])
                + self.normalize_strategy["mean"]
            )

    def _standard(self, df):

        if self.compute_norm_factor:
            self.normalize_strategy["mean"], self.normalize_strategy["std"] = (
                df.mean(),
                df.std(),
            )

        if not self.undo:
            df = df - self.normalize_strategy["mean"]
            return df / self.normalize_strategy["std"]
        else:
            self._check_columns_present(df.columns, ["std", "mean"])
            return df * self.normalize_strategy["std"] + self.normalize_strategy["mean"]

    def _check_columns_present(self, df_columns, strategy_fields):
        for field in strategy_fields:
            element_columns = self.normalize_strategy[field].index
            if not np.isin(df_columns, element_columns).all():
                raise ValueError("TODO")

    def transform(self, df: pd.DataFrame):

        strategy = self.normalize_strategy["strategy"]

        if strategy == "id":
            norm_handle = self._id
        elif strategy == "min-max":
            norm_handle = self._min_max
        elif strategy == "mean":
            norm_handle = self._mean
        elif strategy == "standard":
            norm_handle = self._standard
        else:
            raise ValueError(f"strategy={self.normalize_strategy} not known")

        return norm_handle(df), self.normalize_strategy


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

    def fit(self, X_ts: TSCDataFrame):
        self.delay_indices_ = self._precompute_delay_indices()

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
