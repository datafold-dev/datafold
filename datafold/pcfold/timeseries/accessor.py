#!/usr/bin/env python3

import itertools
from typing import Union, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal
from sklearn import metrics

from datafold.pcfold.timeseries.collection import (
    TimeSeriesCollectionError,
    TSCDataFrame,
)
from datafold.utils.datastructure import (
    is_df_same_index_columns,
    series_if_applicable,
    is_integer,
)


@pd.api.extensions.register_dataframe_accessor("tsc")
class TSCollectionMethods(object):
    def __init__(self, tsc_df):

        # NOTE: cannot call TSCDataFrame(tsc_df) here to transform in case it is a normal
        # DataFrame. This is because the accessor has to know when updating this object.
        if not isinstance(tsc_df, TSCDataFrame):
            raise ValueError(
                "Can use 'tsc' extension only for type TSCDataFrame (convert before)."
            )

        self._tsc_df = tsc_df

    def shift_time(self, shift_t):

        if shift_t == 0:
            return self._tsc_df

        convert_times = self._tsc_df.index.get_level_values(1) + shift_t
        convert_times = pd.Index(convert_times, self._tsc_df.index.names[1])

        new_tsc_index = pd.MultiIndex.from_arrays(
            [self._tsc_df.index.get_level_values(0), convert_times]
        )

        self._tsc_df.index = new_tsc_index
        self._tsc_df._validate()

        return self._tsc_df

    def normalize_time(self):

        convert_times = self._tsc_df.index.get_level_values(1)
        min_time, _ = self._tsc_df.time_interval()

        if not self._tsc_df.is_const_dt():
            raise TimeSeriesCollectionError(
                "To normalize the time index it is required that the time time delta is "
                "constant."
            )

        convert_times = np.array(
            (convert_times / self._tsc_df.dt) - min_time, dtype=np.int
        )
        convert_times = pd.Index(convert_times, name=self._tsc_df.index.names[1])

        new_tsc_index = pd.MultiIndex.from_arrays(
            [self._tsc_df.index.get_level_values(0), convert_times]
        )
        self._tsc_df.index = new_tsc_index
        self._tsc_df._validate()

        return self._tsc_df

    def shift_matrices(
        self, snapshot_orientation="column"
    ) -> Tuple[np.ndarray, np.ndarray]:

        if not self._tsc_df.is_const_dt():
            raise TimeSeriesCollectionError(
                "Cannot compute shift matrices: Time series are required to have the "
                "same time delta."
            )

        ts_counts = self._tsc_df.lengths_time_series

        if is_integer(ts_counts):
            ts_counts = pd.Series(
                np.ones(self._tsc_df.nr_timeseries, dtype=np.int) * ts_counts,
                index=self._tsc_df.ids,
            )

        nr_shift_snapshots = (ts_counts.subtract(1)).sum()
        insert_indices = np.append(0, (ts_counts.subtract(1)).cumsum().to_numpy())

        assert len(insert_indices) == self._tsc_df.nr_timeseries + 1

        if snapshot_orientation == "column":
            shift_left = np.zeros([self._tsc_df.nr_qoi, nr_shift_snapshots])
        elif snapshot_orientation == "row":
            shift_left = np.zeros([nr_shift_snapshots, self._tsc_df.nr_qoi])
        else:
            raise ValueError(f"snapshot_orientation={snapshot_orientation} not known")

        shift_right = np.zeros_like(shift_left)

        # NOTE: if this has performance issues or memory issues, then it may be beneficial
        # to do the whole thing with boolean indexing

        for i, (id_, ts_df) in enumerate(self._tsc_df.itertimeseries()):
            # transpose because snapshots are column-wise by convention, whereas here they
            # are row-wise

            if snapshot_orientation == "column":
                # start from 0 and exclude last snapshot
                shift_left[:, insert_indices[i] : insert_indices[i + 1]] = ts_df.iloc[
                    :-1, :
                ].T
                # exclude 0 and go to last snapshot
                shift_right[:, insert_indices[i] : insert_indices[i + 1]] = ts_df.iloc[
                    1:, :
                ].T
            else:  # "row"
                shift_left[insert_indices[i] : insert_indices[i + 1], :] = ts_df.iloc[
                    :-1, :
                ]
                shift_right[insert_indices[i] : insert_indices[i + 1], :] = ts_df.iloc[
                    1:, :
                ]

        return shift_left, shift_right

    def normalize_qoi(self, normalize_strategy: Union[str, dict]):
        return NormalizeQoi(
            normalize_strategy=normalize_strategy, undo=False
        ).transform(self._tsc_df)

    def undo_normalize_qoi(self, normalize_strategy: dict):
        return NormalizeQoi(normalize_strategy=normalize_strategy, undo=True).transform(
            df=self._tsc_df
        )

    def takens_embedding(
        self,
        lag=0,
        delays=3,
        frequency=1,
        time_direction="backward",
        attach=False,
        fill_value: float = np.nan,
    ):
        """Wrapper function for class TakensEmbedding"""

        takens = TakensEmbedding(lag, delays, frequency, time_direction, fill_value)
        return_df = takens.apply(self._tsc_df)

        if attach:
            return_df = return_df.drop(self._tsc_df.columns, axis=1)
            return_df = pd.concat([self._tsc_df, return_df], axis=1)

        return return_df

    def plot_density2d(self, time, xresolution: int, yresolution: int, covariance=None):
        """
        Plot the density for a given time_step. For this:
          - Take the first two columns of the underlying data frame and interpret them as
            x and y coordinates.
          - Then, place Gaussian bells onto these coordinates and sum up the values of the
            corresponding probability density functions (PDF).
          - The PDF must be evaluated on a fine-granular grid.

          Returns axis handle.
        """

        if len(self._tsc_df.columns) != 2:
            raise ValueError("Density can only be plotted for 2D time series.")

        if covariance is None:
            covariance = np.eye(2)

        df = self._tsc_df.single_time_df(time=time)

        xmin = df.iloc[:, 0].min()
        xmax = df.iloc[:, 0].max()
        ymin = df.iloc[:, 1].min()
        ymax = df.iloc[:, 1].max()

        xdim = (xmin, xmax, xresolution)
        ydim = (ymin, ymax, yresolution)

        grid = self._generate_2d_meshgrid(xdim, ydim)

        coordinates = list(list(item) for item in df.itertuples(index=False))
        summed_density_values_as_vector = self._sum_up_density_values_on_grid(
            grid, coordinates, covariance
        )

        reshaped_density_values = summed_density_values_as_vector.reshape(
            xresolution, yresolution
        )

        gridsize = (1, 1)
        ax1 = plt.subplot2grid(gridsize, (0, 0))

        heatmap_plot = ax1.imshow(
            reshaped_density_values, origin="lower", cmap="seismic"
        )

        ax1.set_xlabel(df.columns[0])
        ax1.set_ylabel(df.columns[1])

        # The plot contains [xy]resolution pixels starting from 0 to [xy]resolution.
        # Therefore, place ticks at first and last pixel.
        ax1.set_xticks([0, xresolution])
        ax1.set_yticks([0, yresolution])

        float_precision = 4
        ax1.set_xticklabels(
            [round(xmin, float_precision), round(xmax, float_precision)]
        )
        ax1.set_yticklabels(
            [round(ymin, float_precision), round(ymax, float_precision)]
        )

        return heatmap_plot

    def _generate_2d_meshgrid(self, xdim, ydim):
        """
        Both parameters must be a tuple consisting of three values (min, max, resolution
        Return a n x 2 vector representing the grid.
        """
        x = np.linspace(*xdim)
        y = np.linspace(*ydim)

        xx, yy = np.meshgrid(x, y)
        grid = np.c_[xx.ravel(), yy.ravel()]

        return grid

    def _sum_up_density_values_on_grid(self, grid, coordinates, covariance):
        # Evaluate probability density functions of Gaussian bells on grid.
        normal_distributions_at_given_coordinates = [
            multivariate_normal(coordinate, cov=covariance)
            for coordinate in coordinates
        ]
        evaluated_density_functions_on_grid = [
            distribution.pdf(grid)
            for distribution in normal_distributions_at_given_coordinates
        ]

        # Stack all 1D-vectors and sum them up vertically.
        stacked_density_values_on_grid = np.vstack(evaluated_density_functions_on_grid)
        summed_density_values_as_vector = np.sum(stacked_density_values_on_grid, axis=0)

        return summed_density_values_as_vector


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


class TimeSeriesError(object):

    VALID_MODES = ["timeseries", "timestep", "qoi"]
    VALID_METRIC = ["rmse", "rrmse", "mse", "mae", "max"]

    def __init__(self, metric: str, mode: str, normalize_strategy: str = "id"):

        mode = mode.lower()
        metric = metric.lower()

        if metric in self.VALID_METRIC:
            self.metric = self._metric_from_str_input(metric)
        else:
            raise ValueError(f"Invalid metric={mode}. Choose from {self.VALID_METRIC}")

        if mode in self.VALID_MODES:
            self.mode = mode
        else:
            raise ValueError(f"Invalid mode={mode}. Choose from {self.VALID_MODES}")

        self.normalize_strategy = normalize_strategy

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

    def _is_scalar_score(self, multioutput) -> bool:
        """
        Determines if the error is scalar, i.e. if there are multiple quantity of
        interests, then these are combined to a single number (depending on the
        multioutput parameter)

        Parameters
        ----------
        multioutput
            See sklearn.metrics documentation for valid arguments

        Returns
        -------
        bool
            Whether it is a scalar error.

        """

        if (
            isinstance(multioutput, str) and multioutput == "uniform_average"
        ) or isinstance(multioutput, np.ndarray):
            scalar_score = True

        elif multioutput == "raw_values":
            scalar_score = False

        else:
            raise ValueError(f"Illegal argument of multioutput={multioutput}")
        return scalar_score

    def _scalar_column_name(self, multioutput) -> list:

        assert self._is_scalar_score(multioutput)

        if isinstance(multioutput, str) and multioutput == "uniform_average":
            column = ["error_uniform_average"]
        elif isinstance(multioutput, np.ndarray):
            column = ["error_user_weights"]
        else:
            raise ValueError(f"Illegal argument of multioutput={multioutput}")

        return column

    def _error_per_timeseries(
        self,
        y_true: TSCDataFrame,
        y_pred: TSCDataFrame,
        sample_weight=None,
        multioutput="uniform_average",
    ) -> Union[pd.Series, pd.DataFrame]:

        if self._is_scalar_score(multioutput=multioutput):
            column = self._scalar_column_name(multioutput=multioutput)

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
                multioutput=multioutput,
            )

        return series_if_applicable(error_per_timeseries)

    def _error_per_qoi(
        self, y_true: TSCDataFrame, y_pred: TSCDataFrame, sample_weight=None
    ):
        # NOTE: score per qoi is never a multiouput, as a QoI is seen as a single scalar
        # quantity

        error_per_qoi = self.metric(
            y_true.to_numpy(),
            y_pred.to_numpy(),
            sample_weight=sample_weight,
            multioutput="raw_values",  # raw_values to tread every qoi separately
        )

        error_per_qoi = pd.Series(error_per_qoi, index=y_true.columns,)
        assert not error_per_qoi.isna().any()
        return error_per_qoi

    def _error_per_timestep(
        self,
        y_true: TSCDataFrame,
        y_pred: TSCDataFrame,
        sample_weight=None,
        multioutput="uniform_average",
    ):

        time_indices = y_true.time_indices(unique_values=True)

        if self._is_scalar_score(multioutput=multioutput):
            column = self._scalar_column_name(multioutput=multioutput)

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
                y_true_t,
                y_pred_t,
                sample_weight=sample_weight,
                multioutput=multioutput,
            )

        return series_if_applicable(error_per_time)

    def score(
        self,
        y_true: TSCDataFrame,
        y_pred: TSCDataFrame,
        sample_weight=None,
        multi_qoi="uniform_average",
    ) -> Union[pd.Series, pd.DataFrame]:

        if not is_df_same_index_columns(y_true, y_pred):
            raise ValueError("y_true and y_pred must have the same index and columns")

        # first normalize y_true, then with the same factors the y_pred
        y_true, norm_factors = NormalizeQoi(
            self.normalize_strategy, undo=False
        ).transform(y_true)

        y_pred, _ = NormalizeQoi(norm_factors, undo=False).transform(y_pred)

        if self.mode == "timeseries":
            error_result = self._error_per_timeseries(
                y_true=y_true,
                y_pred=y_pred,
                sample_weight=sample_weight,
                multioutput=multi_qoi,
            )

        elif self.mode == "timestep":
            error_result = self._error_per_timestep(
                y_true=y_true,
                y_pred=y_pred,
                sample_weight=sample_weight,
                multioutput=multi_qoi,
            )

        elif self.mode == "qoi":
            error_result = self._error_per_qoi(
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


class TakensEmbedding(object):
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

        self.delay_indices = self._precompute_delay_indices()

    def _precompute_delay_indices(self):
        # zero delay (original data) is not treated
        return self.lag + (
            np.arange(1, (self.delays * self.frequency) + 1, self.frequency)
        )

    def _expand_all_delay_columns(self, cols):
        def expand():
            delayed_columns = list()
            for didx in self.delay_indices:
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

    def apply(self, tsc: TSCDataFrame):

        if not tsc.is_const_dt():
            raise TimeSeriesCollectionError("dt is not const")

        if (tsc.lengths_time_series <= self.delay_indices.max()).any():
            raise TimeSeriesCollectionError(
                f"Mismatch of delay and time series length. Shortest time series has "
                f"length "
                f"{np.array(tsc.lengths_time_series).min()} and maximum delay is "
                f"{self.delay_indices.max()}"
            )

        delayed_tsc = self._setup_delayed_timeseries_collection(tsc)

        for i, ts in tsc.itertimeseries():
            for delay_idx in self.delay_indices:
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


if __name__ == "__main__":
    pass
