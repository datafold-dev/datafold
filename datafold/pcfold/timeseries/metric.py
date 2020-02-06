#!/usr/bin/env python3

from typing import Callable, Union

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler, Normalizer, StandardScaler

from datafold.pcfold.timeseries import TSCDataFrame
from datafold.utils.datastructure import is_df_same_index, series_if_applicable


class TSCMetric:

    VALID_MODE = ["timeseries", "timestep", "qoi"]
    VALID_METRIC = ["rmse", "rrmse", "mse", "mae", "max"]
    VALID_SCALING = ["id", "min-max", "standard", "l2_normalize"]

    def __init__(self, metric: str, mode: str, scaling: str = "id"):

        mode = mode.lower()
        metric = metric.lower()

        if metric in self.VALID_METRIC:
            self.metric = self._metric_from_str_input(metric)
        else:
            raise ValueError(f"Invalid metric={mode}. Choose from {self.VALID_METRIC}")

        if mode in self.VALID_MODE:
            self.mode = mode
        else:
            raise ValueError(f"Invalid mode={mode}. Choose from {self.VALID_MODE}")

        self.scaling = self._select_scaling(name=scaling)

    @classmethod
    def make_tsc_metric(cls, metric: str, mode: str, scaling: str = "id") -> Callable:
        """Metric measures difference between two datasets. Metrics can be
        multidimensional. Returns a callable to the metric"""
        return cls(metric=metric, mode=mode, scaling=scaling).eval_metric

    def _select_scaling(self, name):

        if name == "id":
            return None
        elif name == "min-max":
            return MinMaxScaler()
        elif name == "standard":
            return StandardScaler()
        elif name == "l2_normalize":
            return Normalizer(norm="l2")
        else:
            raise ValueError(
                f"scaling={name} is not known. Choose from {self.VALID_SCALING}"
            )

    def _scaling(self, y_true: TSCDataFrame, y_pred: TSCDataFrame):

        # it is checked before that y_true and y_pred indices/columns are identical
        index, columns = y_true.index, y_true.columns

        # first normalize y_true, afterwards (with the same factors from y_true!) y_pred
        if self.scaling is not None:  # is None if scaling is identity
            y_true = self.scaling.fit_transform(y_true)
            y_pred = self.scaling.transform(y_pred.to_numpy())

            y_true = TSCDataFrame(y_true, index=index, columns=columns)
            y_pred = TSCDataFrame(y_pred, index=index, columns=columns)

        return y_true, y_pred

    def _rmse_metric(
        self, y_true, y_pred, sample_weight=None, multioutput="uniform_average"
    ):
        # TODO: [minor] when upgrading to scikit-learn 0.22 mean_squared error has a
        #  keyword "squared", if set to False, this computes the RMSE directly

        mse_error = metrics.mean_squared_error(
            y_true, y_pred, sample_weight=sample_weight, multioutput=multioutput
        )
        return np.sqrt(mse_error)

    def _rrmse_metric(
        self, y_true, y_pred, sample_weight=None, multioutput="uniform_average"
    ):
        """Metric from

        # TODO make proper citing
        Higher Order Dynamic Mode Decomposition, Le Clainche and Vega
        """

        if multioutput == "uniform_average":
            norm_ = np.sum(np.square(np.linalg.norm(y_true, axis=1)))
        else:  # multioutput == "raw_values":
            norm_ = np.sum(np.square(y_true), axis=0)

        if (np.asarray(norm_) <= 1e-14).any():
            raise RuntimeError(
                f"norm factor(s) are too small for rrmse \n norm_factor = {norm_}"
            )

        mse_error = metrics.mean_squared_error(
            y_true, y_pred, sample_weight=sample_weight, multioutput=multioutput
        )

        mse_error_relative = np.divide(mse_error, norm_)
        return np.sqrt(mse_error_relative)

    def _max_error(
        self, y_true, y_pred, sample_weight=None, multioutput="uniform_average"
    ):
        """ Only a wrapper for sklean.metrics.max_error to allow sample weight and
        multioutput arguments.
        """

        # fails if y is multioutput
        return metrics.max_error(y_true=y_true, y_pred=y_pred)

    def _metric_from_str_input(self, error_metric: str):

        error_metric = error_metric.lower()

        if error_metric == "rmse":  # root mean squared error
            error_metric_handle = self._rmse_metric
        elif error_metric == "rrmse":
            error_metric_handle = self._rrmse_metric
        elif error_metric == "mse":
            error_metric_handle = metrics.mean_squared_error
        elif error_metric == "mae":
            error_metric_handle = metrics.mean_absolute_error
        elif error_metric == "max":
            error_metric_handle = self._max_error
        else:
            raise ValueError(f"Metric {error_metric} not known. Please report bug.")

        return error_metric_handle

    def _is_scalar_multioutput(self, multioutput) -> bool:

        if (
            isinstance(multioutput, str) and multioutput == "uniform_average"
        ) or isinstance(multioutput, np.ndarray):
            scalar_score = True

        elif multioutput == "raw_values":
            scalar_score = False

        else:
            raise ValueError(f"Illegal argument of multioutput={multioutput}")
        return scalar_score

    def _single_column_name(self, multioutput) -> list:

        assert self._is_scalar_multioutput(multioutput)

        if isinstance(multioutput, str) and multioutput == "uniform_average":
            column = ["metric_uniform_average"]
        elif isinstance(multioutput, np.ndarray):
            column = ["metric_user_weights"]
        else:
            raise ValueError(f"Illegal argument of multioutput={multioutput}")

        return column

    def _metric_per_timeseries(
        self,
        y_true: TSCDataFrame,
        y_pred: TSCDataFrame,
        sample_weight=None,
        multi_qoi="uniform_average",
    ) -> Union[pd.Series, pd.DataFrame]:

        if self._is_scalar_multioutput(multioutput=multi_qoi):
            column = self._single_column_name(multioutput=multi_qoi)

            # Make in both cases a DataFrame and later convert to Series in the scalar
            # case this allows to use .loc[i, :] in the loop
            error_per_timeseries = pd.DataFrame(
                np.nan, index=y_true.ids, columns=column
            )
        else:
            error_per_timeseries = pd.DataFrame(
                np.nan, index=y_true.ids, columns=y_true.columns.to_list(),
            )

        for i, y_true_single in y_true.itertimeseries():
            y_pred_single = y_pred.loc[i, :]

            error_per_timeseries.loc[i, :] = self.metric(
                y_true_single,
                y_pred_single,
                sample_weight=sample_weight,
                multioutput=multi_qoi,
            )

        return series_if_applicable(error_per_timeseries)

    def _metric_per_qoi(
        self, y_true: TSCDataFrame, y_pred: TSCDataFrame, sample_weight=None
    ):
        # NOTE: score per qoi is never a multioutput, as a QoI is seen as a single scalar
        # quantity

        error_per_qoi = self.metric(
            y_true.to_numpy(),
            y_pred.to_numpy(),
            sample_weight=sample_weight,
            multioutput="raw_values",  # raw_values to tread every qoi separately
        )

        error_per_qoi = pd.Series(error_per_qoi, index=y_true.columns,)
        return error_per_qoi

    def _metric_per_timestep(
        self,
        y_true: TSCDataFrame,
        y_pred: TSCDataFrame,
        sample_weight=None,
        multi_qoi="uniform_average",
    ):

        time_indices = y_true.time_values(unique_values=True)

        if self._is_scalar_multioutput(multioutput=multi_qoi):
            column = self._single_column_name(multioutput=multi_qoi)

            # Make in both cases a DataFrame and later convert to Series in the scalar
            # case this allows to use .loc[i, :] in the loop
            error_per_time = pd.DataFrame(np.nan, index=time_indices, columns=column)

        else:
            error_per_time = pd.DataFrame(
                np.nan, index=time_indices, columns=y_true.columns.to_list()
            )

        error_per_time.index.name = "time"

        idx_slice = pd.IndexSlice
        for t in time_indices:

            y_true_t = pd.DataFrame(y_true.loc[idx_slice[:, t], :])
            y_pred_t = pd.DataFrame(y_pred.loc[idx_slice[:, t], :])

            error_per_time.loc[t, :] = self.metric(
                y_true_t, y_pred_t, sample_weight=sample_weight, multioutput=multi_qoi,
            )

        return series_if_applicable(error_per_time)

    def eval_metric(
        self,
        y_true: TSCDataFrame,
        y_pred: TSCDataFrame,
        sample_weight=None,
        multi_qoi="raw_samples",
    ) -> Union[pd.Series, pd.DataFrame]:
        """Note for docu:
        These are "special" metrics for TSC and allow different "modes" to compare
        errors between qoi, time or timeseries (see VALID_MODE).

        The "multi_qoi" argument steers how to weight the QoI (e.g. when looking on
        time).

        Usually, all outputs are multi-dimensional and measure error and therefore are
        not suited for scoring.
        """

        is_df_same_index(y_true, y_pred, handle="raise")

        self._scaling(y_true=y_true, y_pred=y_pred)

        # score depending on mode:
        if self.mode == "timeseries":
            error_result = self._metric_per_timeseries(
                y_true=y_true,
                y_pred=y_pred,
                sample_weight=sample_weight,
                multi_qoi=multi_qoi,
            )

        elif self.mode == "timestep":
            error_result = self._metric_per_timestep(
                y_true=y_true,
                y_pred=y_pred,
                sample_weight=sample_weight,
                multi_qoi=multi_qoi,
            )

        elif self.mode == "qoi":
            error_result = self._metric_per_qoi(
                y_true=y_true, y_pred=y_pred, sample_weight=sample_weight
            )

        else:
            raise ValueError(f"Invalid mode={self.mode}. Please report bug.")

        if isinstance(error_result, pd.Series):
            assert not error_result.isnull().any()
        elif isinstance(error_result, pd.DataFrame):
            assert not error_result.isnull().any().any()
        else:
            raise RuntimeError("Bug. Please report.")

        return error_result


class TSCKfoldSeries:
    def __init__(self, n_splits=3, shuffle=False, random_state=None):
        self.kfold_splitter = KFold(
            n_splits=n_splits, shuffle=shuffle, random_state=random_state
        )

    def split(self, X: TSCDataFrame, y=None, groups=None):
        if not X.is_same_ts_length():
            raise NotImplementedError(
                "Currently, all time series are required to have "
                "the same length for this method."
            )

        n_time_series = X.nr_timeseries
        len_time_series = X.lengths_time_series
        n_samples = X.shape[0]

        indices_matrix = np.arange(n_samples).reshape([n_time_series, len_time_series])

        # uses the indices as samples and splits along the time series
        # the indices (rows) are then collected and can be used to select from X
        for train, test in self.kfold_splitter.split(indices_matrix):
            train_indices = indices_matrix[train].flatten()
            test_indices = indices_matrix[test].flatten()

            yield train_indices, test_indices

    def get_n_splits(self, X, y=None, groups=None):
        return self.kfold_splitter.get_n_splits(X, y, groups=groups)


class TSCKFoldTime:
    def __init__(self, n_splits=3):
        self.kfold_splitter = KFold(n_splits=n_splits, shuffle=False, random_state=None)

    def split(self, X: TSCDataFrame, y=None, groups=None):
        if not X.is_equal_time_values():
            raise NotImplementedError(
                "Currently, each time series must have the same " "time indices."
            )

        n_timeseries = X.nr_timeseries
        len_timeseries = X.lengths_time_series
        n_samples = X.shape[0]

        indices_matrix = np.arange(n_samples).reshape(
            [len_timeseries, n_timeseries], order="F"
        )

        for train, test in self.kfold_splitter.split(indices_matrix):
            train_indices = indices_matrix[train].flatten()
            test_indices = indices_matrix[test].flatten()
            yield train_indices, test_indices

    def get_n_splits(self, X, y=None, groups=None):
        return self.kfold_splitter.get_n_splits(X, y, groups=groups)


def make_tsc_scorer(metric_func, **metric_kwargs) -> Callable:
    """
    As specified by sklearn a score returns a floating point. Therefore, the output
    of TSC metric needs to be further condensed.
    """

    def _tsc_scoring(y_true: TSCDataFrame, y_pred: TSCDataFrame, sample_weight=None):
        """useage of sample_weight depends on the selected metric_func
         * mode=qoi -> weight each qoi
         * mode=timeseries -> weight each time series
         * mode=time -> weight each time point
        """

        eval_tsc_metric: pd.Series = metric_func(
            y_true=y_true,
            y_pred=y_pred,
            sample_weight=metric_kwargs.pop("sample_weight", None),
            multi_qoi=metric_kwargs.pop("multi_qoi", "uniform_average"),
        )

        if isinstance(eval_tsc_metric, pd.DataFrame):
            raise ValueError(
                "The metric_func must be configured such that only a "
                "pd.Series is used for the scoring function."
            )

        if sample_weight is None:
            score = np.mean(eval_tsc_metric.to_numpy())
        elif isinstance(sample_weight, np.ndarray):
            assert len(sample_weight) == len(eval_tsc_metric)
            score = np.average(eval_tsc_metric.to_numpy(), weights=sample_weight)
        else:
            raise TypeError(f"sample_weight={sample_weight} is invalid.")

        # score is "greater is better", all TSC metrics measure the error
        return -1 * float(score)

    return _tsc_scoring
