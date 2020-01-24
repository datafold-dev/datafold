#!/usr/bin/env python3

import itertools
from typing import Union

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from datafold.pcfold.timeseries import TSCDataFrame
from datafold.pcfold.timeseries.collection import TimeSeriesCollectionError
from datafold.pcfold.timeseries.base import TSCTransformMixIn, TRANF_TYPES


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

    def fit(self, X: TRANF_TYPES, y=None, **fit_params):
        super(TSCQoiPreprocess, self).fit(X, y, **fit_params)
        self.transform_cls_.fit(X.to_numpy())
        return self

    def transform(self, X: TRANF_TYPES):
        super(TSCQoiPreprocess, self).transform(X)

        values = self.transform_cls_.transform(X.to_numpy())
        return self._same_type_X(X=X, values=values)

    def fit_transform(self, X: TRANF_TYPES, y=None, **fit_params):
        super(TSCQoiPreprocess, self).fit_transform(X, y, **fit_params)
        values = self.transform_cls_.fit_transform(X)
        return self._same_type_X(X=X, values=values)

    def inverse_transform(self, X: TRANF_TYPES):
        super(TSCQoiPreprocess, self).inverse_transform(X)
        values = self.transform_cls_.inverse_transform(X.to_numpy())
        return self._same_type_X(X=X, values=values)


class TSCIdentity(TSCTransformMixIn):
    def __init__(self):
        pass

    def fit(self, X: TRANF_TYPES, y=None, **fit_params):
        super(TSCIdentity, self).fit(X, y, **fit_params)
        return self

    def transform(self, X: TRANF_TYPES):
        super(TSCIdentity, self).transform(X)
        return X

    def inverse_transform(self, X: TRANF_TYPES):
        super(TSCIdentity, self).inverse_transform(X)
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

        super(TSCQoiScale, self).__init__(cls=_cls, **kwargs)


class TSCPrincipalComponent(TSCTransformMixIn):
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

    def fit(self, X: TRANF_TYPES, y=None, **fit_params):
        self._pca.fit(X=X, y=None)

        super(TSCPrincipalComponent, self).fit(
            X, y, transform_columns=[f"pca{i}" for i in range(self._pca.n_components_)]
        )

        return self

    def transform(self, X: TRANF_TYPES):
        super(TSCPrincipalComponent, self).transform(X)

        pca_data = self._pca.transform(X.to_numpy())
        return self._same_type_X(X, values=pca_data, columns=self._transform_columns)

    def inverse_transform(self, X: TRANF_TYPES):
        super(TSCPrincipalComponent, self).inverse_transform(X)
        data_orig_space = self._pca.inverse_transform(X.to_numpy())

        return self._same_type_X(X, values=data_orig_space, columns=self._fit_columns)


class TSCTakensEmbedding(TSCTransformMixIn):
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
        self.fillin_handle = fillin_handle

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

        data = np.zeros([tsc.shape[0], total_nr_columns]) * np.nan

        delayed_tsc = TSCDataFrame(
            data, index=tsc.index, columns=self._transform_columns
        )

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

    def fit(self, X, y=None, **fit_params):

        # TODO: Check that there are no NaN values present. (the integrate the option
        #  to remove fill in rows.

        self.delay_indices_ = self._precompute_delay_indices()
        transform_columns = self._expand_all_delay_columns(X.columns)
        super(TSCTakensEmbedding, self).fit(X, y, transform_columns=transform_columns)
        return self

    def transform(self, X: TRANF_TYPES):

        self._check_fit_columns(X=X)

        if not X.is_const_dt():
            raise TimeSeriesCollectionError("dt is not constant")

        if not X.is_finite():
            raise ValueError("The TSCDataFrame must only consist of finite values.")

        if (X.lengths_time_series <= self.delay_indices_.max()).any():
            raise TimeSeriesCollectionError(
                f"Mismatch of delay and time series length. Shortest time series has "
                f"length {np.array(X.lengths_time_series).min()} and maximum delay is "
                f"{self.delay_indices_.max()}"
            )

        X = self._allocate_delayed_qoi(X)

        # Compute the shifts --> per single time series
        for i, ts in X.loc[:, self._fit_columns].itertimeseries():
            for delay_idx in self.delay_indices_:
                shifted_timeseries = self._shift_timeseries(ts, delay_idx)
                X.loc[i, shifted_timeseries.columns] = shifted_timeseries.values

        if self.fillin_handle == "remove":
            bool_idx = np.logical_not(np.sum(pd.isnull(X), axis=1).astype(np.bool))
            X = X.loc[bool_idx]

            if isinstance(X, TSCDataFrame):
                pass
            elif isinstance(X, pd.DataFrame):
                import warnings

                # TODO: irregular time series could be removed completely (per option)
                warnings.warn(
                    "During Takens delay embedding the time series collection is "
                    "not regular (due to large delays) and is returned as "
                    "pd.DataFrame"
                )

        return X

    def inverse_transform(self, X: TRANF_TYPES):
        self._check_transform_columns(X=X)
        return X.loc[:, self._fit_columns]


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
